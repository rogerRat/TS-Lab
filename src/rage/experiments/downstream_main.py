"""Downstream main experiment: does RAGE selection bring training gains?

Three arms with identical training budget from the same base checkpoint:

  rage     fine-tune on the B windows picked by the full two-stage
              pipeline (adaptive quadrature + block perturbations +
              stability-constrained lower-quantile selection);
  random      fine-tune on B uniformly random windows from the same
              candidate pool (selection replaced by chance);
  unfiltered  fine-tune on B uniformly random windows from the entire
              source pool (no candidate preprocessing, no selection).

Each arm is trained with several seeds; evaluation is the mean per-window
cross-entropy and expected-value MAE on the held-out test region of the
reference series (the last ``test_fraction`` of each trajectory, never used
for gradients, perturbations, or selection).
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from rage.core import RAGEConfig, TimeSeriesFragment
from rage.quadrature import AdaptiveQuadratureEngine
from rage.perturbation import ReferencePerturbationEngine
from rage.selection import StableRankingSelector

from .core_experiments import BatchedScanner, cosine_scores, mean_gradient, build_engine
from .metrics import mase_sql, quantile_forecasts, seasonality_for
from .data import (
    build_candidate_fragments,
    build_reference_fragments,
    reference_load_dirs,
    sample_pool_windows,
    split_reference_fragment,
)


# ---------------------------------------------------------------------------
# Selection phase (shared with EXP-C logic, on the selection region only)
# ---------------------------------------------------------------------------

@dataclass
class TestGroup:
    """Held-out test windows of one reference series."""

    entity_id: str
    windows: np.ndarray


def evaluate_forecast(engine, groups: list[TestGroup], *, batch_size: int = 128) -> dict:
    """All downstream metrics over grouped test windows.

    CE and MAE are computed on the full stack; MASE and SQL need each
    window's dataset seasonality, so they are aggregated per group weighted
    by window count.
    """
    stack = np.concatenate([group.windows for group in groups])
    ce_mae = evaluate(engine, stack, batch_size=batch_size)
    counts = np.array([len(group.windows) for group in groups], dtype=np.float64)
    mase_total = 0.0
    sql_total = 0.0
    quantile_chunks = []
    for group in groups:
        quantiles, _scales = quantile_forecasts(engine, group.windows, batch_size=batch_size)
        quantile_chunks.append(quantiles)
        metrics = mase_sql(
            group.windows, quantiles,
            context_length=engine.config.context_length,
            seasonality=seasonality_for(group.entity_id),
        )
        mase_total += metrics["mase"] * len(group.windows)
        sql_total += metrics["sql"] * len(group.windows)
    total = counts.sum()
    return {
        "test_ce": ce_mae["test_ce"],
        "test_mae": ce_mae["test_mae"],
        "test_mase": mase_total / total,
        "test_sql": sql_total / total,
        "test_windows": int(total),
    }


@dataclass
class SelectionOutcome:
    selected_windows: np.ndarray       # [B, window_length]
    candidate_windows: np.ndarray      # [N, window_length] full pool
    candidate_ids: list[tuple[int, int]]
    selected_ids: list[tuple[int, int]]
    anchors: int
    total_starts: int
    versions: int
    stability: list[float]
    full_gradient: np.ndarray          # full-scan reference gradient (selection region)


def run_selection(
    scanner: BatchedScanner,
    selection_fragments: list[TimeSeriesFragment],
    candidate_fragments: list[TimeSeriesFragment],
    candidate_windows: np.ndarray,
    candidate_ids: list[tuple[int, int]],
    *,
    budget: int,
    tolerance: float,
    versions: int,
    block_length: int,
    stability_threshold: float,
    lower_quantile: float,
    candidate_sketches: np.ndarray,
) -> SelectionOutcome:
    config = RAGEConfig(
        context_length=scanner.context_length,
        prediction_length=scanner.prediction_length,
        direction_tolerance=tolerance,
        magnitude_tolerance=tolerance,
        min_interval_size=4,
        block_length=block_length,
        num_reference_versions=versions,
        selection_budget=budget,
        ranking_budget=budget,
        stability_threshold=stability_threshold,
        lower_quantile=lower_quantile,
        projection_dimension=scanner.dimension,
    )
    quadrature = AdaptiveQuadratureEngine(config, scanner.anchor_computer())
    anchor_cache: dict[tuple[int, int], np.ndarray] = {}
    fragment_means = []
    for fragment in selection_fragments:
        starts = fragment.valid_starts(config.context_length, config.prediction_length)
        sketches = scanner.scan_starts(fragment, starts)
        for i, start in enumerate(starts):
            anchor_cache[(fragment.fragment_id, int(start))] = sketches[i]
        fragment_means.append(mean_gradient(sketches))
    quadrature.anchor_cache = anchor_cache
    full_gradient = np.mean(np.stack(fragment_means), axis=0)

    quadrature_results = [
        quadrature.process_fragment(fragment, fragment.fragment_id)
        for fragment in selection_fragments
    ]
    anchors = sum(r.computed_gradients for r in quadrature_results)
    total_starts = sum(r.total_positions for r in quadrature_results)
    valid_starts_list = [
        fragment.valid_starts(config.context_length, config.prediction_length)
        for fragment in selection_fragments
    ]

    perturbation = ReferencePerturbationEngine(config)
    reference_versions = perturbation.generate_reference_versions(
        quadrature_results, selection_fragments, valid_starts_list
    )
    selector = StableRankingSelector(config)
    candidate_scores = selector.score_candidates(candidate_sketches, candidate_ids, reference_versions)
    selection = selector.select(candidate_scores)

    chosen = [candidate_ids[score.candidate_id] for score in selection.selected]
    fragment_by_id = {f.fragment_id: f for f in candidate_fragments}
    windows = np.stack([
        fragment_by_id[fragment_id].extract_window(
            start, scanner.context_length, scanner.prediction_length
        )
        for fragment_id, start in chosen
    ])
    return SelectionOutcome(
        selected_windows=windows,
        candidate_windows=candidate_windows,
        candidate_ids=candidate_ids,
        selected_ids=chosen,
        anchors=anchors,
        total_starts=total_starts,
        versions=len(reference_versions),
        stability=[s.stability_frequency for s in selection.selected],
        full_gradient=full_gradient,
    )


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def fine_tune(
    engine,
    windows: np.ndarray,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    eval_windows: np.ndarray | None = None,
    eval_every: int = 0,
) -> dict:
    """AdamW fine-tuning on the given windows; base state restored by caller.

    With ``eval_every`` > 0 and a held-out ``eval_windows`` stack, the test
    cross-entropy is logged during training so the short-horizon behaviour
    (where fine-tuning gains appear) is visible alongside the fixed final
    budget used for reporting.
    """
    model = engine.model
    base_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    windows = np.asarray(windows, dtype=np.float32)
    losses: list[float] = []
    eval_curve: list[dict] = []
    model.train()
    for step in range(steps):
        indices = rng.integers(0, len(windows), size=batch_size)
        tensors = engine.batch_tensors(windows[indices])
        optimizer.zero_grad(set_to_none=True)
        output = model(**tensors, return_dict=True)
        loss = output.loss if getattr(output, "loss", None) is not None else None
        if loss is None:
            from gradient.chronos import per_sample_cross_entropy
            loss = per_sample_cross_entropy(output.logits, tensors["labels"]).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if eval_every > 0 and eval_windows is not None and (step + 1) % eval_every == 0:
            model.eval()
            metrics = evaluate(engine, eval_windows, compute_mae=False)
            model.train()
            eval_curve.append({"step": step + 1, "test_ce": metrics["test_ce"]})
    model.eval()
    curve = {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "tail_mean": float(np.mean(losses[-50:])),
    }
    # keep the fine-tuned weights; caller evaluates before restoring
    return {"curve": curve, "eval_curve": eval_curve, "base_state": base_state}


def evaluate(
    engine,
    test_windows: np.ndarray,
    *,
    batch_size: int = 256,
    compute_mae: bool = True,
) -> dict:
    """Mean per-window cross-entropy (and expected-value MAE) on test windows."""
    _losses, sketches = engine.batch_gradient_sketch(test_windows, batch_size=batch_size)
    ce_loss = float(np.mean(_losses))
    result = {"test_ce": ce_loss, "test_windows": int(len(test_windows))}
    if compute_mae:
        result["test_mae"] = expected_value_mae(engine, test_windows, batch_size=batch_size)
    return result


@torch.no_grad()
def expected_value_mae(engine, windows: np.ndarray, *, batch_size: int = 256) -> float:
    """MAE of the expected-value forecast: scale * sum_i p_i * centers_i."""
    tokenizer = engine.tokenizer
    centers = torch.as_tensor(tokenizer.centers, dtype=torch.float32, device=engine.device)
    # Vocabulary layout: [PAD=0, EOS=1, bin tokens (len(centers)), unused].
    vocab_centers = torch.zeros(engine.config.n_tokens, dtype=torch.float32, device=engine.device)
    vocab_centers[2 : 2 + len(centers)] = centers
    errors = []
    windows = np.asarray(windows, dtype=np.float32)
    context_length = engine.config.context_length
    for begin in range(0, len(windows), batch_size):
        chunk = torch.from_numpy(windows[begin : begin + batch_size])
        contexts = chunk[:, :context_length]
        futures = chunk[:, context_length:]
        input_ids, attention_mask, scale = tokenizer.context_input_transform(contexts)
        labels, label_mask = tokenizer.label_input_transform(futures, scale)
        input_ids = input_ids.to(engine.device)
        attention_mask = attention_mask.to(engine.device)
        labels = labels.masked_fill(label_mask == 0, 0).to(engine.device)
        output = engine.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        logits = output.logits[:, :-1, :].float()  # drop the EOS-position column
        probabilities = torch.softmax(logits, dim=-1)
        expected_scaled = probabilities @ vocab_centers
        expected = expected_scaled * scale.to(engine.device).view(-1, 1)
        errors.append((expected - futures.to(engine.device)).abs().mean(dim=1))
    return float(torch.cat(errors).mean())


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+",
                        default=["ETT_15T", "ETT_1H", "bizitobs_l2c_5T", "jena_weather_1H"])
    parser.add_argument("--series-cap", type=int, default=6)
    parser.add_argument("--slice-length", type=int, default=16384)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--candidate-series", type=int, default=32)
    parser.add_argument("--candidate-slice", type=int, default=4096)
    parser.add_argument("--candidate-cap", type=int, default=4096)
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--versions", type=int, default=20)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--stability-threshold", type=float, default=0.6)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--prediction-length", type=int, default=64)
    parser.add_argument("--projection-dimension", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--train-batch", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--out", default="outputs/rage_downstream")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    window_length = args.context_length + args.prediction_length
    started = time.time()

    print("loading and splitting reference series ...", flush=True)
    load_dirs = reference_load_dirs(keywords=args.datasets)
    full_fragments = build_reference_fragments(
        load_dirs, window_length=window_length,
        slice_length=args.slice_length, series_cap=args.series_cap,
    )
    pairs = [split_reference_fragment(f, test_fraction=args.test_fraction, window_length=window_length)
             for f in full_fragments]
    selection_fragments = [selection for selection, _test in pairs]
    test_fragments = [test for _selection, test in pairs]
    test_groups = [
        TestGroup(
            entity_id=test.entity_id,
            windows=np.stack([
                test.extract_window(int(start), args.context_length, args.prediction_length)
                for start in test.valid_starts(args.context_length, args.prediction_length)
            ]),
        )
        for test in test_fragments
    ]
    test_windows = np.concatenate([group.windows for group in test_groups])
    print("selection region: %d fragments; test windows: %d" % (
        len(selection_fragments), len(test_windows)), flush=True)

    print("loading candidate pool ...", flush=True)
    candidate_fragments = build_candidate_fragments(
        window_length=window_length, slice_length=args.candidate_slice,
        num_series=args.candidate_series, seed=0,
    )
    rng = np.random.default_rng(1)
    all_candidates = [
        (fragment, int(start))
        for fragment in candidate_fragments
        for start in fragment.valid_starts(args.context_length, args.prediction_length)
    ]
    chosen_idx = rng.choice(len(all_candidates), size=min(args.candidate_cap, len(all_candidates)),
                            replace=False)
    pool = [all_candidates[i] for i in chosen_idx]
    candidate_windows = np.stack([
        fragment.extract_window(start, args.context_length, args.prediction_length)
        for fragment, start in pool
    ])
    candidate_ids = [(fragment.fragment_id, start) for fragment, start in pool]
    print("candidate pool: %d windows from %d series" % (len(pool), len(candidate_fragments)), flush=True)

    engine = build_engine(args)
    scanner = BatchedScanner(engine, args.context_length, args.prediction_length, log_every=50000)
    print("engine ready on %s" % engine.device, flush=True)

    print("computing candidate gradients ...", flush=True)
    candidate_sketches = scanner.scan_windows(candidate_windows).astype(np.float64)

    print("running RAGE selection on the selection region ...", flush=True)
    selection = run_selection(
        scanner, selection_fragments, candidate_fragments, candidate_windows, candidate_ids,
        budget=args.budget, tolerance=args.tolerance, versions=args.versions,
        block_length=args.block_length, stability_threshold=args.stability_threshold,
        lower_quantile=0.25, candidate_sketches=candidate_sketches,
    )
    print("selection done: %d anchors / %d starts (%.2fx), %d versions, mean stability %.3f" % (
        selection.anchors, selection.total_starts,
        selection.total_starts / max(selection.anchors, 1), selection.versions,
        float(np.mean(selection.stability))), flush=True)

    print("sampling baseline arms ...", flush=True)
    pool_rng = np.random.default_rng(2)
    random_idx = pool_rng.choice(len(pool), size=args.budget, replace=False)
    random_windows = candidate_windows[random_idx]
    unfiltered_windows, _origins = sample_pool_windows(
        args.budget, window_length=window_length, seed=3,
    )
    # Diagnostic arm: the ideal full-scan ranking (top-B by plain cosine to
    # the full reference gradient) separates approximation error of the
    # quadrature stage from differences of the stability-constrained rule.
    plain_scores = cosine_scores(candidate_sketches, selection.full_gradient)
    fullscan_windows = candidate_windows[np.argsort(-plain_scores)[: args.budget]]

    arms = {
        "rage": selection.selected_windows,
        "fullscan": fullscan_windows,
        "random": random_windows,
        "unfiltered": unfiltered_windows,
    }

    results = {"protocol": vars(args)}
    results["selection"] = {
        "anchors": selection.anchors,
        "total_reference_starts": selection.total_starts,
        "anchor_compression": round(selection.total_starts / max(selection.anchors, 1), 2),
        "versions": selection.versions,
        "mean_stability": round(float(np.mean(selection.stability)), 3),
        "budget": args.budget,
    }

    # base (untrained) evaluation for reference
    base_eval = evaluate(engine, test_windows)
    results["base_model"] = base_eval
    print("base model: CE %.4f MAE %.4f" % (base_eval["test_ce"], base_eval["test_mae"]), flush=True)

    for arm_name, windows in arms.items():
        arm_records = []
        for seed in args.seeds:
            run_start = time.time()
            outcome = fine_tune(
                engine, windows, steps=args.train_steps, batch_size=args.train_batch,
                learning_rate=args.learning_rate, seed=seed,
                eval_windows=test_windows, eval_every=args.eval_every,
            )
            metrics = evaluate(engine, test_windows)
            metrics["train"] = outcome["curve"]
            metrics["eval_curve"] = outcome["eval_curve"]
            metrics["seed"] = seed
            metrics["elapsed_s"] = round(time.time() - run_start, 1)
            arm_records.append(metrics)
            print("arm=%-10s seed=%d CE %.4f MAE %.4f (train %.4f -> %.4f) [%.0fs]" % (
                arm_name, seed, metrics["test_ce"], metrics["test_mae"],
                outcome["curve"]["initial_loss"], outcome["curve"]["final_loss"],
                metrics["elapsed_s"]), flush=True)
            # restore base weights for the next run
            engine.model.load_state_dict(outcome["base_state"], strict=True)
            engine.model.eval()
        results[arm_name] = arm_records

    summary = summarize(results, arms=list(arms))
    results["summary"] = summary
    results["finished"] = time.strftime("%F %T")
    results["total_minutes"] = round((time.time() - started) / 60, 1)
    (out_dir / "downstream_main.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print("DOWNSTREAM MAIN DONE in %.1f minutes" % results["total_minutes"], flush=True)
    return 0


def summarize(results: dict, arms: list[str]) -> dict:
    base = results["base_model"]
    rows = {}
    for arm in arms:
        records = results[arm]
        ce = np.array([r["test_ce"] for r in records])
        mae = np.array([r["test_mae"] for r in records])
        rows[arm] = {
            "test_ce_mean": round(float(ce.mean()), 5),
            "test_ce_std": round(float(ce.std()), 5),
            "test_mae_mean": round(float(mae.mean()), 5),
            "test_mae_std": round(float(mae.std()), 5),
            "ce_gain_vs_base": round(float(base["test_ce"] - ce.mean()), 5),
            "mae_gain_vs_base": round(float(base["test_mae"] - mae.mean()), 5),
        }
    comparison = {}
    for arm in arms:
        if arm == "rage":
            continue
        ce_req = np.array([r["test_ce"] for r in results["rage"]])
        ce_arm = np.array([r["test_ce"] for r in results[arm]])
        mae_req = np.array([r["test_mae"] for r in results["rage"]])
        mae_arm = np.array([r["test_mae"] for r in results[arm]])
        comparison["rage_vs_%s" % arm] = {
            "ce_delta": round(float(ce_arm.mean() - ce_req.mean()), 5),
            "ce_delta_std": round(float((ce_arm - ce_req).std()), 5),
            "mae_delta": round(float(mae_arm.mean() - mae_req.mean()), 5),
            "mae_delta_std": round(float((mae_arm - mae_req).std()), 5),
            "ce_positive_seeds": int(np.sum(ce_arm > ce_req)),
            "mae_positive_seeds": int(np.sum(mae_arm > mae_req)),
        }
    return {"arms": rows, "paired_comparison": comparison, "base_model": base}


if __name__ == "__main__":
    raise SystemExit(main())
