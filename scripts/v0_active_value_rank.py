"""Replay expanded value ranking and active correction."""
from __future__ import annotations
import argparse
import csv
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from configs.logger import configure_logger, format_duration
from configs.paths import PROJECT_ROOT, output_path

NAME = "v0-active-value-rank"
SOURCE = PROJECT_ROOT / "outputs/train/v0-gradient-proxy-test"
TRUTH = PROJECT_ROOT / "outputs/train/v0-uni-data-evaluation"
FRACTIONS = (0.05, 0.10, 0.20)
BUDGETS = (8, 16, 32, 64)
HEAD_MS = 14.21
FULL_MS = 42.80

def write_csv(path, rows):
    if not rows:
        raise ValueError("empty table")
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def metrics(ref, true, predicted, method, budget, model_seconds=0.0):
    n = len(true)
    oracle_order = np.argsort(-true, kind="stable")
    selected_order = np.argsort(-predicted, kind="stable")
    rows = []
    for fraction in FRACTIONS:
        k = math.ceil(n * fraction)
        oracle = oracle_order[:k]
        selected = selected_order[:k]
        oracle_mean = float(np.mean(true[oracle]))
        selected_mean = float(np.mean(true[selected]))
        rows.append(dict(reference_id=ref["reference_id"], reference_track=ref["track"],
                         candidates=n, method=method, budget=budget, fraction=fraction,
                         k=k, recall=float(np.intersect1d(oracle, selected).size / k),
                         selected_true_mean=selected_mean, oracle_true_mean=oracle_mean,
                         oracle_regret=max(0.0, oracle_mean - selected_mean),
                         estimated_head_calls=n + 1, estimated_full_calls=budget + 1,
                         estimated_call_ms=(n + 1) * HEAD_MS + (budget + 1) * FULL_MS,
                         replay_model_seconds=model_seconds))
    return rows

def load_sources():
    refs = json.loads((TRUTH / "references.json").read_text(encoding="ascii"))
    if len(refs) != 100 or sorted(r["reference_id"] for r in refs) != list(range(100)):
        raise ValueError("invalid references")
    tracks = []
    for track in range(10):
        directory = SOURCE / ("trajectory-%02d" % track)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="ascii"))
        n = int(manifest["windows"])
        head = np.load(directory / "head_gradients.npy", mmap_mode="r")
        feat = np.load(directory / "window_features.npz")
        full = np.load(TRUTH / ("trajectory-%02d-scores.npy" % track), mmap_mode="r")
        if not (directory / ".complete").is_file() or head.shape != (n, 1048576):
            raise ValueError("invalid head cache")
        if full.shape != (n, 100, 2) or not np.array_equal(feat["start"], np.arange(n)):
            raise ValueError("invalid score or feature alignment")
        tracks.append(dict(head=head, feat=feat, full=full, manifest=manifest, n=n))
    return refs, tracks

def batch_one(out, log):
    if (out / ".complete").exists():
        raise ValueError("batch 1 already complete")
    out.mkdir(parents=True, exist_ok=True)
    refs, tracks = load_sources()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    ref_head = np.stack([tracks[r["track"]]["head"][r["start"]] for r in refs])
    ref_embed = np.stack([tracks[r["track"]]["feat"]["embedding"][r["start"]] for r in refs])
    ref_loss = np.array([tracks[r["track"]]["feat"]["loss"][r["start"]] for r in refs])
    ref_tensor = torch.from_numpy(ref_head).to(device)
    ref_norm = torch.linalg.vector_norm(ref_tensor, dim=1)
    ref_embed_norm = np.linalg.norm(ref_embed, axis=1)
    started = time.monotonic()
    log.info("Stage start: batch 1; references=100 stride=16 output=%s", out)
    for track, item in enumerate(tracks):
        reference_starts = {r["start"] for r in refs if r["track"] == track}
        starts = np.array([i for i in range(0, item["n"], 16)
                           if i not in reference_starts], dtype=np.int32)
        head_score = np.empty((len(starts), 100), dtype=np.float32)
        head_norm = np.empty(len(starts), dtype=np.float32)
        for left in range(0, len(starts), 32):
            chosen = starts[left:left + 32]
            query = torch.from_numpy(np.array(item["head"][chosen], copy=True)).to(device)
            norm = torch.linalg.vector_norm(query, dim=1)
            score = (query @ ref_tensor.T) / (norm[:, None] * ref_norm[None, :]).clamp_min(1e-30)
            head_score[left:left + len(chosen)] = score.clamp(-1, 1).cpu().numpy()
            head_norm[left:left + len(chosen)] = norm.cpu().numpy()
            del query, norm, score
        embed = np.asarray(item["feat"]["embedding"][starts], dtype=np.float32)
        embed_score = (embed @ ref_embed.T) / np.maximum(
            np.linalg.norm(embed, axis=1)[:, None] * ref_embed_norm[None, :], 1e-30)
        loss = np.asarray(item["feat"]["loss"][starts], dtype=np.float32)
        loss_difference = np.abs(loss[:, None] - ref_loss[None, :])
        full_score = np.asarray(item["full"][starts, :, 0], dtype=np.float32)
        if not all(np.isfinite(x).all() for x in (head_score, head_norm, embed_score,
                                                  loss_difference)):
            raise ValueError("nonfinite proxy or feature score")
        cross_reference_ids = [r["reference_id"] for r in refs if r["track"] != track]
        if not np.isfinite(full_score[:, cross_reference_ids]).all():
            raise ValueError("nonfinite cross-track full score")
        if np.any(head_norm <= 0):
            raise ValueError("zero head norm")
        np.savez(out / ("trajectory-%02d.npz" % track), starts=starts,
                 head=head_score, full=full_score, head_norm=head_norm,
                 embedding=embed_score.astype(np.float32),
                 loss_difference=loss_difference.astype(np.float32))
        elapsed = time.monotonic() - started
        log.info("Track complete: %02d candidates=%d elapsed=%s eta=%s", track,
                 len(starts), format_duration(elapsed),
                 format_duration(elapsed * (9 - track) / (track + 1)))
    del ref_tensor, ref_head
    arrays = [np.load(out / ("trajectory-%02d.npz" % t)) for t in range(10)]
    rows, ranking = [], []
    for ref in refs:
        rid = ref["reference_id"]
        source = [a for t, a in enumerate(arrays) if t != ref["track"]]
        true = np.concatenate([a["full"][:, rid] for a in source])
        proxy = np.concatenate([a["head"][:, rid] for a in source])
        ranking.append(dict(reference_id=rid, reference_track=ref["track"],
                            candidates=len(true), spearman=float(spearmanr(true, proxy).statistic),
                            proxy_mae=float(np.mean(np.abs(true - proxy)))))
        rows.extend(metrics(ref, true, proxy, "head_proxy", 0))
    write_csv(out / "proxy_metrics.csv", rows)
    write_csv(out / "proxy_ranking.csv", ranking)
    write_csv(out / "proxy_summary.csv", [
        dict(fraction=f, references=100, mean_recall=float(np.mean(
            [r["recall"] for r in rows if r["fraction"] == f])),
             mean_regret=float(np.mean(
            [r["oracle_regret"] for r in rows if r["fraction"] == f])))
        for f in FRACTIONS])
    (out / "input_manifest.json").write_text(json.dumps(
        dict(experiment=NAME, batch="1-expanded-cross-rank", stride=16,
             references=refs, top_fractions=FRACTIONS,
             source_manifests=[t["manifest"] for t in tracks],
             head_cost_ms=HEAD_MS, full_cost_ms=FULL_MS), indent=2) + "\n",
        encoding="ascii")
    (out / ".complete").write_text("complete\n", encoding="ascii")
    log.info("Stage complete: batch 1; mean_spearman=%.4f duration=%s output=%s",
             np.mean([r["spearman"] for r in ranking]),
             format_duration(time.monotonic() - started), out)

def reference_data(arrays, rid, track):
    source = [a for t, a in enumerate(arrays) if t != track]
    true = np.concatenate([a["full"][:, rid] for a in source]).astype(np.float64)
    proxy = np.concatenate([a["head"][:, rid] for a in source]).astype(np.float64)
    norm = np.concatenate([a["head_norm"] for a in source]).astype(np.float64)
    embed = np.concatenate([a["embedding"][:, rid] for a in source]).astype(np.float64)
    loss = np.concatenate([a["loss_difference"][:, rid] for a in source]).astype(np.float64)
    raw = np.stack([proxy, proxy * proxy, np.log(norm), embed, loss], axis=1)
    scale = raw.std(axis=0)
    scale[scale < 1e-12] = 1.0
    design = np.column_stack([np.ones(len(proxy)), (raw - raw.mean(axis=0)) / scale])
    return true, proxy, design

def fit_predict(design, true, proxy, labels):
    x = design[labels]
    residual = true[labels] - proxy[labels]
    penalty = np.diag([1e-6, 1.0, 1.0, 1.0, 1.0, 1.0])
    precision = x.T @ x + penalty
    coefficient = np.linalg.solve(precision, x.T @ residual)
    corrected = proxy + design @ coefficient
    error = residual - x @ coefficient
    sigma = max(0.02, float(np.sqrt(np.mean(error * error))))
    covariance = np.linalg.inv(precision)
    leverage = np.einsum("ij,jk,ik->i", design, covariance, design, optimize=True)
    uncertainty = sigma * np.sqrt(1.0 + np.maximum(leverage, 0.0))
    corrected[labels] = true[labels]
    return corrected, uncertainty

def batch_two(out, log):
    source = out.parent / "batch-1"
    if not (source / ".complete").is_file():
        raise ValueError("batch 1 incomplete")
    if (out / ".complete").exists():
        raise ValueError("batch 2 already complete")
    out.mkdir(parents=True, exist_ok=True)
    refs = json.loads((TRUTH / "references.json").read_text(encoding="ascii"))
    arrays = [np.load(source / ("trajectory-%02d.npz" % t)) for t in range(10)]
    rows, query_rows = [], []
    started = time.monotonic()
    log.info("Stage start: batch 2; references=100 budgets=%s output=%s", BUDGETS, out)
    for ref in refs:
        rid = ref["reference_id"]
        true, proxy, design = reference_data(arrays, rid, ref["track"])
        n = len(true)
        order = np.argsort(proxy, kind="stable")
        initial = order[np.linspace(0, n - 1, 8, dtype=int)].tolist()
        if len(set(initial)) != 8:
            raise ValueError("duplicate initial labels")
        for method in ("random", "proxy_high", "uncertainty", "boundary"):
            selected = list(initial)
            selected_set = set(selected)
            random_order = np.random.default_rng(rid + 1000).permutation(n)
            model_seconds = 0.0
            for budget in range(8, 65):
                fit_started = time.monotonic()
                corrected, uncertainty = fit_predict(design, true, proxy, np.array(selected))
                model_seconds += time.monotonic() - fit_started
                if budget in BUDGETS:
                    rows.extend(metrics(ref, true, corrected, method, budget, model_seconds))
                if budget == 64:
                    break
                if method == "random":
                    chosen = next(int(i) for i in random_order if int(i) not in selected_set)
                elif method == "proxy_high":
                    chosen = next(int(i) for i in order[::-1] if int(i) not in selected_set)
                elif method == "uncertainty":
                    utility = uncertainty.copy()
                    utility[selected] = -np.inf
                    chosen = int(np.argmax(utility))
                else:
                    k = math.ceil(0.10 * n)
                    cutoff = float(np.partition(corrected, n - k)[n - k])
                    utility = uncertainty / (np.abs(corrected - cutoff) + 0.01)
                    utility[selected] = -np.inf
                    chosen = int(np.argmax(utility))
                selected.append(chosen)
                selected_set.add(chosen)
            query_rows.append(dict(reference_id=rid, method=method,
                                   initial=initial, query_order=selected))
        if (rid + 1) % 10 == 0:
            elapsed = time.monotonic() - started
            log.info("References complete: %d/100 elapsed=%s eta=%s", rid + 1,
                     format_duration(elapsed),
                     format_duration(elapsed * (99 - rid) / (rid + 1)))
    write_csv(out / "selection_metrics.csv", rows)
    (out / "query_manifest.json").write_text(json.dumps(query_rows, indent=2) + "\n",
                                              encoding="ascii")
    summary = []
    for method in ("random", "proxy_high", "uncertainty", "boundary"):
        for budget in BUDGETS:
            for fraction in FRACTIONS:
                group = [r for r in rows if r["method"] == method
                         and r["budget"] == budget and r["fraction"] == fraction]
                if len(group) != 100:
                    raise ValueError("missing reference result")
                summary.append(dict(method=method, budget=budget, fraction=fraction,
                                    references=100,
                                    mean_recall=float(np.mean([r["recall"] for r in group])),
                                    mean_regret=float(np.mean([r["oracle_regret"] for r in group])),
                                    estimated_call_ms=float(np.mean(
                                        [r["estimated_call_ms"] for r in group])),
                                    mean_replay_model_seconds=float(np.mean(
                                        [r["replay_model_seconds"] for r in group]))))
    write_csv(out / "summary.csv", summary)
    (out / "input_manifest.json").write_text(json.dumps(
        dict(experiment=NAME, batch="2-active-correction", budgets=BUDGETS,
             top_fractions=FRACTIONS, seed=0, initial_labels=8,
             ridge_penalty=1.0, uncertainty_floor=0.02, boundary_offset=0.01,
             boundary_fraction=0.10, head_cost_ms=HEAD_MS, full_cost_ms=FULL_MS),
        indent=2) + "\n", encoding="ascii")
    (out / ".complete").write_text("complete\n", encoding="ascii")
    log.info("Stage complete: batch 2; rows=%d duration=%s output=%s",
             len(rows), format_duration(time.monotonic() - started), out)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    base = output_path("train", "outputs/train/v0-active-value-rank")
    base.mkdir(parents=True, exist_ok=True)
    log = configure_logger("v0.active.value.rank", NAME,
                           "batch-%d" % args.batch, "initialization")
    log.info("Configuration: batch=%d stride=16 fractions=%s budgets=%s output=%s",
             args.batch, FRACTIONS, BUDGETS, base)
    if args.batch == 1:
        batch_one(base / "batch-1", log)
    else:
        batch_two(base / "batch-2", log)

if __name__ == "__main__":
    main()
