"""Core RAGS experiments on real profile10b data with ChronosTiny.

EXP-A  Temporal reference fragility (Section 5.2): does changing the
       reference construction (stride, boundary, block deletion, noise,
       reference length, random subset) change candidate rankings?
EXP-B  Adaptive quadrature recovery (Section 5.3): can a fraction of the
       reference anchors recover the full-scan ranking?
EXP-C  Main selection experiment: the full two-stage pipeline with
       stability-constrained selection versus plain full-scan ranking.

All gradient sketches come from the ChronosTiny engine (lm_head fast path,
projected). Reference-side scans are batched to keep the GPU saturated.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

from rags.core import RAGSConfig, TimeSeriesFragment, compute_cosine_similarity
from rags.quadrature import AdaptiveQuadratureEngine
from rags.perturbation import ReferencePerturbationEngine
from rags.selection import StableRankingSelector

from .data import build_candidate_fragments, build_reference_fragments, reference_load_dirs


# ---------------------------------------------------------------------------
# Gradient scanning
# ---------------------------------------------------------------------------

class BatchedScanner:
    """Batched window-gradient scans over fragments."""

    def __init__(self, engine, context_length: int, prediction_length: int, log_every: int = 0):
        self.engine = engine
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.log_every = log_every
        self.windows_scanned = 0

    @property
    def dimension(self) -> int:
        return int(self.engine.config.projection_dimension)

    def scan_starts(self, fragment: TimeSeriesFragment, starts) -> np.ndarray:
        """Sketches for the given start positions, shape [len(starts), d]."""
        windows = np.stack([
            fragment.extract_window(int(start), self.context_length, self.prediction_length)
            for start in starts
        ])
        return self.scan_windows(windows)

    def scan_windows(self, windows: np.ndarray) -> np.ndarray:
        windows = np.asarray(windows, dtype=np.float32)
        _losses, sketches = self.engine.batch_gradient_sketch(windows)
        self.windows_scanned += len(windows)
        if self.log_every and self.windows_scanned // self.log_every != (self.windows_scanned - len(windows)) // self.log_every:
            print("[%s] scanned %d windows" % (time.strftime("%H:%M:%S"), self.windows_scanned), flush=True)
        return np.asarray(sketches, dtype=np.float32)

    def anchor_computer(self):
        """Gradient-computer contract for AdaptiveQuadratureEngine (batch 1)."""
        def computer(window: np.ndarray, context_length: int, prediction_length: int) -> np.ndarray:
            sketches = self.scan_windows(np.asarray(window, dtype=np.float32).reshape(1, -1))
            self.windows_scanned -= 1  # anchor scans count separately in reports
            return sketches[0]
        return computer


def mean_gradient(sketches: np.ndarray) -> np.ndarray:
    return sketches.astype(np.float64).mean(axis=0)


def cosine_scores(candidate_matrix: np.ndarray, gradient: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    normalized = candidate_matrix / (np.linalg.norm(candidate_matrix, axis=1, keepdims=True) + epsilon)
    norm = np.linalg.norm(gradient)
    if norm < epsilon:
        return np.zeros(len(normalized))
    return normalized @ (gradient / (norm + epsilon))


def ranking_agreement(scores_a: np.ndarray, scores_b: np.ndarray, k: int) -> dict:
    spearman = stats.spearmanr(scores_a, scores_b).statistic
    kendall = stats.kendalltau(scores_a, scores_b).statistic
    top_a = set(np.argsort(-scores_a)[:k].tolist())
    top_b = set(np.argsort(-scores_b)[:k].tolist())
    jaccard = len(top_a & top_b) / max(len(top_a | top_b), 1)
    return {
        "spearman": float(spearman),
        "kendall": float(kendall),
        "topk_jaccard": float(jaccard),
        "topk_overlap": len(top_a & top_b),
    }


# ---------------------------------------------------------------------------
# Reference constructions (EXP-A variants)
# ---------------------------------------------------------------------------

def variant_starts(fragment: TimeSeriesFragment, window_length: int, name: str, rng: np.random.Generator):
    max_start = fragment.length - window_length + 1
    base = np.arange(1, max_start + 1)
    if name == "baseline":
        return base, fragment
    if name.startswith("stride"):
        step = int(name.split("-")[1])
        return base[::step], fragment
    if name.startswith("shift"):
        delta = int(name[len("shift"):])  # e.g. "shift-32" -> -32, "shift+32" -> +32
        shifted = base + delta
        shifted = shifted[(shifted >= 1) & (shifted <= max_start)]
        return shifted, fragment
    if name.startswith("len"):
        fraction = float(name.split("-")[1])
        cut = int(fragment.length * (1.0 - fraction))
        trimmed = TimeSeriesFragment(
            data=fragment.data[cut:], entity_id=fragment.entity_id, fragment_id=fragment.fragment_id,
        )
        max_start_t = trimmed.length - window_length + 1
        return np.arange(1, max_start_t + 1), trimmed
    if name.startswith("delete"):
        fraction = float(name.split("-")[1])
        block = max(int(len(base) * fraction / 4), 1)
        drop = set()
        for _ in range(4):
            begin = int(rng.integers(0, len(base) - block))
            drop.update(range(begin, begin + block))
        keep = np.array([i for i in range(len(base)) if i not in drop])
        return base[keep], fragment
    if name.startswith("random"):
        fraction = float(name.split("-")[1])
        count = max(int(len(base) * fraction), 1)
        return base[np.sort(rng.choice(len(base), size=count, replace=False))], fragment
    if name.startswith("noise"):
        sigma = float(name.split("-")[1]) * float(fragment.data.std())
        noised = TimeSeriesFragment(
            data=(fragment.data + rng.normal(0.0, sigma, size=fragment.length)).astype(np.float32),
            entity_id=fragment.entity_id, fragment_id=fragment.fragment_id,
        )
        return base, noised
    raise ValueError("unknown variant: %s" % name)


VARIANTS = [
    "baseline",
    "stride-2", "stride-4", "stride-8",
    "shift-32", "shift+32",
    "len-0.5", "len-0.75",
    "delete-0.1",
    "random-0.5",
    "noise-0.05",
]


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def run_experiment_a(scanner, fragments, window_length, candidate_matrix, k, seed):
    """EXP-A: reference-construction sensitivity of candidate rankings.

    For each construction variant we rebuild the full reference gradient from
    batched scans and score the same candidate pool; agreement is measured
    against the baseline construction (stride 1, full tail, no noise).
    """
    print("EXP-A: temporal reference fragility", flush=True)
    rng = np.random.default_rng(seed)
    results = []
    baseline_scores = None
    for name in VARIANTS:
        label = name.replace("-", " ")
        start_time = time.time()
        gradients = []
        total_starts = 0
        for fragment in fragments:
            starts, target = variant_starts(fragment, window_length, name, rng)
            if len(starts) == 0:
                continue
            sketches = scanner.scan_starts(target, starts)
            gradients.append(mean_gradient(sketches))
            total_starts += len(starts)
        gradient = np.mean(np.stack(gradients), axis=0)
        scores = cosine_scores(candidate_matrix, gradient)
        record = {
            "variant": label,
            "starts_scanned": total_starts,
            "gradient_norm": float(np.linalg.norm(gradient)),
            "elapsed_s": round(time.time() - start_time, 1),
        }
        if baseline_scores is None:
            baseline_scores = scores
            record.update({"spearman": 1.0, "kendall": 1.0, "topk_jaccard": 1.0, "topk_overlap": k})
        else:
            record.update(ranking_agreement(baseline_scores, scores, k))
        results.append(record)
        print("  %-12s spearman=%.3f jaccard=%.3f starts=%d" % (
            label, record["spearman"], record["topk_jaccard"], total_starts), flush=True)
    return results


def run_experiment_b(scanner, fragments, tolerances, epsilon_guard=1e-8):
    """EXP-B: adaptive quadrature versus full scan per fragment.

    All start gradients are precomputed in one batched scan per fragment and
    shared across the tolerance sweep through the engine's anchor cache; the
    adaptive stage then costs no additional gradient evaluations.
    """
    print("EXP-B: adaptive quadrature recovery", flush=True)
    results = []
    for fragment in fragments:
        max_start = fragment.length - (scanner.context_length + scanner.prediction_length) + 1
        starts = np.arange(1, max_start + 1)
        full_sketches = scanner.scan_starts(fragment, starts)
        full_G = mean_gradient(full_sketches)
        anchor_cache = {
            (fragment.fragment_id, int(start)): full_sketches[i]
            for i, start in enumerate(starts)
        }
        record = {
            "entity": fragment.entity_id,
            "total_starts": int(len(starts)),
            "full_gradient_norm": float(np.linalg.norm(full_G)),
            "tolerances": [],
        }
        for tolerance in tolerances:
            config = RAGSConfig(
                context_length=scanner.context_length,
                prediction_length=scanner.prediction_length,
                direction_tolerance=tolerance,
                magnitude_tolerance=tolerance,
                min_interval_size=4,
                projection_dimension=scanner.dimension,
            )
            engine = AdaptiveQuadratureEngine(config, None, anchor_cache=anchor_cache)
            result = engine.process_fragment(fragment, fragment.fragment_id)
            estimated, _ = engine.compute_full_reference_gradient([result], [fragment])
            record["tolerances"].append({
                "tolerance": tolerance,
                "anchors": int(result.computed_gradients),
                "anchor_ratio": round(result.computed_gradients / max(len(starts), 1), 4),
                "relative_gradient_error": float(
                    np.linalg.norm(full_G - estimated) / (np.linalg.norm(full_G) + epsilon_guard)
                ),
            })
        results.append(record)
        print("  %s: %d starts, tolerance sweep done" % (fragment.entity_id, len(starts)), flush=True)
    return results


def run_experiment_c(scanner, fragments, candidate_matrix, candidate_ids, *, tolerance,
                     versions, block_length, budget, stability_threshold, lower_quantile, k):
    """EXP-C: full two-stage pipeline with stability-constrained selection."""
    print("EXP-C: main selection experiment", flush=True)
    config = RAGSConfig(
        context_length=scanner.context_length,
        prediction_length=scanner.prediction_length,
        direction_tolerance=tolerance,
        magnitude_tolerance=tolerance,
        min_interval_size=4,
        block_length=block_length,
        num_reference_versions=versions,
        selection_budget=budget,
        ranking_budget=k,
        stability_threshold=stability_threshold,
        lower_quantile=lower_quantile,
        projection_dimension=scanner.dimension,
    )
    quadrature = AdaptiveQuadratureEngine(config, scanner.anchor_computer())
    # Precompute all reference start gradients batched; the adaptive stage and
    # both perturbation stages read them through the shared anchor cache.
    anchor_cache: dict[tuple[int, int], np.ndarray] = {}
    fragment_means = []
    for fragment in fragments:
        starts = fragment.valid_starts(config.context_length, config.prediction_length)
        sketches = scanner.scan_starts(fragment, starts)
        for i, start in enumerate(starts):
            anchor_cache[(fragment.fragment_id, int(start))] = sketches[i]
        fragment_means.append(mean_gradient(sketches))
    quadrature.anchor_cache = anchor_cache
    full_G = np.mean(np.stack(fragment_means), axis=0)

    quadrature_results = [
        quadrature.process_fragment(fragment, fragment.fragment_id) for fragment in fragments
    ]
    anchors = sum(r.computed_gradients for r in quadrature_results)
    total_starts = sum(r.total_positions for r in quadrature_results)
    valid_starts_list = [
        fragment.valid_starts(config.context_length, config.prediction_length) for fragment in fragments
    ]
    estimated_G, _ = quadrature.compute_full_reference_gradient(quadrature_results, fragments)

    perturbation = ReferencePerturbationEngine(config)
    reference_versions = perturbation.generate_reference_versions(
        quadrature_results, fragments, valid_starts_list
    )

    selector = StableRankingSelector(config)
    candidate_scores = selector.score_candidates(candidate_matrix, candidate_ids, reference_versions)
    selection = selector.select(candidate_scores)

    # Plain full-scan ranking for comparison
    plain_scores = cosine_scores(candidate_matrix, full_G)
    plain_top = set(np.argsort(-plain_scores)[:budget].tolist())
    rags_top = {score.candidate_id for score in selection.selected}

    baseline_scores_matrix = np.stack([
        cosine_scores(candidate_matrix, version.reference_gradient) for version in reference_versions
    ])
    random_baseline = float(np.mean(plain_scores))
    selected_utility = float(np.mean([plain_scores[score.candidate_id] for score in selection.selected]))
    plain_utility = float(np.mean(np.sort(plain_scores)[-budget:]))

    return {
        "anchors": int(anchors),
        "total_reference_starts": int(total_starts),
        "anchor_compression": round(total_starts / max(anchors, 1), 2),
        "versions": len(reference_versions),
        "gradient_norm_full": float(np.linalg.norm(full_G)),
        "gradient_norm_quadrature": float(np.linalg.norm(estimated_G)),
        "relative_gradient_error": float(
            np.linalg.norm(full_G - estimated_G) / (np.linalg.norm(full_G) + 1e-8)
        ),
        "selection": {
            "budget": budget,
            "selected": len(selection.selected),
            "coverage": round(selection.coverage, 3),
            "backfilled": selection.num_backfilled,
            "mean_stability": round(float(np.mean([s.stability_frequency for s in selection.selected])), 3),
            "overlap_with_plain_topk": len(plain_top & rags_top),
            "selected_plain_utility": round(selected_utility, 5),
            "plain_topk_utility": round(plain_utility, 5),
            "random_utility": round(random_baseline, 5),
        },
        "version_score_spread": {
            "mean_std_across_versions": round(float(baseline_scores_matrix.std(axis=0).mean()), 5),
        },
        "selected_ids": [
            {
                "fragment": score.fragment_id,
                "start": score.position,
                "stability": round(score.stability_frequency, 2),
                "quantile_score": round(score.lower_quantile_score, 5),
            }
            for score in selection.selected[: budget]
        ],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_engine(args):
    from gradient.chronos import ChronosTinyConfig, ChronosTinyGradientEngine

    checkpoint = getattr(args, "model_checkpoint", None)
    common = dict(
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        projection_dimension=args.projection_dimension,
        projection_repeats=1,
        token_chunk_size=8,
        gradient_batch_size=args.batch_size,
    )
    if checkpoint:
        # Main experiments must consume a completed checkpoint.  Accept the
        # project's 9M Tiny and 46M Small families, but never construct a
        # randomly initialized fallback when a path is missing or invalid.
        config = ChronosTinyConfig(
            **common,
            model_path=str(checkpoint),
            require_warmup_models=False,
            initialization_mode="pretrained",
            parameter_count_min=8_000_000,
            parameter_count_max=47_000_000,
        )
    else:
        config = ChronosTinyConfig(
            **common,
            warmup_model_root="outputs/models/warmup/chronos-tiny-from-scratch",
            warmup_seeds=(str(args.model_seed),),
            warmup_model_count=1,
            require_warmup_models=True,
        )
    return ChronosTinyGradientEngine(config, device="cuda")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+",
                        default=["ETT_15T", "ETT_1H", "bizitobs_l2c_5T", "jena_weather_1H"])
    parser.add_argument("--series-cap", type=int, default=6)
    parser.add_argument("--slice-length", type=int, default=16384)
    parser.add_argument("--candidate-series", type=int, default=32)
    parser.add_argument("--candidate-slice", type=int, default=4096)
    parser.add_argument("--candidate-cap", type=int, default=4096)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--prediction-length", type=int, default=64)
    parser.add_argument("--projection-dimension", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--tolerances", nargs="+", type=float, default=[0.02, 0.05, 0.1, 0.2])
    parser.add_argument("--versions", type=int, default=20)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--stability-threshold", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/rags_core")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    window_length = args.context_length + args.prediction_length
    started = time.time()

    print("loading reference fragments ...", flush=True)
    load_dirs = reference_load_dirs(keywords=args.datasets)
    reference_fragments = build_reference_fragments(
        load_dirs, window_length=window_length,
        slice_length=args.slice_length, series_cap=args.series_cap,
    )
    print("reference fragments: %d from %d datasets" % (len(reference_fragments), len(load_dirs)), flush=True)

    print("loading candidate fragments ...", flush=True)
    candidate_fragments = build_candidate_fragments(
        window_length=window_length, slice_length=args.candidate_slice,
        num_series=args.candidate_series, seed=args.seed,
    )
    rng = np.random.default_rng(args.seed + 1)
    all_candidates = []
    for fragment in candidate_fragments:
        for start in fragment.valid_starts(args.context_length, args.prediction_length):
            all_candidates.append((fragment, int(start)))
    chosen = [all_candidates[i] for i in rng.choice(len(all_candidates),
                                                    size=min(args.candidate_cap, len(all_candidates)),
                                                    replace=False)]
    print("candidate pool: %d starts from %d series" % (len(chosen), len(candidate_fragments)), flush=True)

    engine = build_engine(args)
    print("engine ready: layer=%s fast=%s device=%s d=%d" % (
        engine.layer_name, engine.fast_lm_head, engine.device, args.projection_dimension), flush=True)
    scanner = BatchedScanner(engine, args.context_length, args.prediction_length, log_every=20000)

    print("computing candidate gradients (batched) ...", flush=True)
    windows = np.stack([
        fragment.extract_window(start, args.context_length, args.prediction_length)
        for fragment, start in chosen
    ])
    candidate_sketches = scanner.scan_windows(windows).astype(np.float64)
    candidate_ids = [(fragment.fragment_id, start) for fragment, start in chosen]

    results = {"protocol": vars(args), "started": time.strftime("%F %T")}

    exp_a = run_experiment_a(scanner, reference_fragments, window_length,
                             candidate_sketches, args.topk, args.seed + 2)
    results["exp_a_fragility"] = exp_a
    (out_dir / "exp_a_fragility.json").write_text(json.dumps(exp_a, indent=2))
    print("EXP-A done: %d variants" % len(exp_a), flush=True)

    exp_b = run_experiment_b(scanner, reference_fragments, args.tolerances)
    results["exp_b_quadrature"] = exp_b
    (out_dir / "exp_b_quadrature.json").write_text(json.dumps(exp_b, indent=2))
    print("EXP-B done: %d fragments" % len(exp_b), flush=True)

    exp_c = run_experiment_c(
        scanner, reference_fragments, candidate_sketches, candidate_ids,
        tolerance=args.tolerances[1], versions=args.versions, block_length=args.block_length,
        budget=args.topk, stability_threshold=args.stability_threshold,
        lower_quantile=0.25, k=args.topk,
    )
    results["exp_c_main"] = exp_c
    (out_dir / "exp_c_main.json").write_text(json.dumps(exp_c, indent=2))

    results["finished"] = time.strftime("%F %T")
    results["total_minutes"] = round((time.time() - started) / 60, 1)
    results["windows_scanned"] = scanner.windows_scanned
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2, default=str))
    print("ALL EXPERIMENTS DONE in %.1f minutes, %d windows scanned" % (
        results["total_minutes"], scanner.windows_scanned), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
