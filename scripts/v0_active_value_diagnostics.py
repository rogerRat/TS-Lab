"""Diagnose proxy coverage and replay equal-budget query baselines."""
from __future__ import annotations
import argparse
import csv
import json
import math
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from configs.logger import configure_logger, format_duration
from configs.paths import PROJECT_ROOT, output_path
from scripts.v0_active_value_rank import TRUTH, reference_data

BASE = output_path("train", "outputs/train/v0-active-value-rank")
BUDGETS = (0, 8, 16, 32, 64)
SEEDS = tuple(range(30))


def write_csv(path, rows):
    if not rows:
        raise ValueError("empty table")
    with path.open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def inputs():
    source = BASE / "batch-1"
    if not (source / ".complete").is_file():
        raise ValueError("source batch incomplete")
    refs = json.loads((TRUTH / "references.json").read_text(encoding="ascii"))
    arrays = [np.load(source / ("trajectory-%02d.npz" % t)) for t in range(10)]
    if len(refs) != 100:
        raise ValueError("reference count changed")
    return refs, arrays


def orders(ref, arrays):
    true, proxy, _ = reference_data(arrays, ref["reference_id"], ref["track"])
    if not np.isfinite(true).all() or not np.isfinite(proxy).all():
        raise ValueError("nonfinite score")
    n = len(true)
    k = math.ceil(n * 0.10)
    true_order = np.argsort(-true, kind="stable")
    proxy_order = np.argsort(-proxy, kind="stable")
    return true, proxy, n, k, true_order, proxy_order


def diagnose(out, log):
    refs, arrays = inputs()
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    rank_rows, coverage_rows, bin_rows = [], [], []
    started = time.monotonic()
    log.info("Stage start: proxy coverage; references=100 output=%s", out)
    for ref in refs:
        rid = ref["reference_id"]
        true, proxy, n, k, true_order, proxy_order = orders(ref, arrays)
        proxy_rank = np.empty(n, dtype=int)
        proxy_rank[proxy_order] = np.arange(1, n + 1)
        good_ranks = proxy_rank[true_order[:k]]
        for true_rank, idx in enumerate(true_order[:k], 1):
            rank_rows.append(dict(reference_id=rid, reference_track=ref["track"],
                                  candidates=n, k=k, candidate_index=int(idx),
                                  true_rank=true_rank, proxy_rank=int(proxy_rank[idx]),
                                  true_score=float(true[idx]), proxy_score=float(proxy[idx])))
        counts = np.bincount(good_ranks, minlength=n + 1)
        cumulative = np.cumsum(counts)
        positions = sorted(set([0, n] + list(range(10, n + 1, 10)) +
                               [min(n, j * k) for j in (1, 2, 3, 4, 5, 7)]))
        for m in positions:
            coverage_rows.append(dict(reference_id=rid, reference_track=ref["track"],
                                      candidates=n, k=k, proxy_candidates=m,
                                      recovered=int(cumulative[m]), recall=float(cumulative[m] / k)))
        edges = [0, k, 2 * k, 3 * k, 4 * k, 5 * k, n]
        for b in range(6):
            bin_rows.append(dict(reference_id=rid, reference_track=ref["track"],
                                 candidates=n, k=k, rank_start=edges[b] + 1,
                                 rank_end=edges[b + 1], true_top_count=int(
                                     np.count_nonzero((good_ranks > edges[b]) &
                                                      (good_ranks <= edges[b + 1])))))
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        axes[0].plot(np.arange(n + 1), cumulative / k)
        axes[0].axvline(k, color="gray", linestyle="--")
        axes[0].set(xlabel="Proxy candidate count", ylabel="True top-k recall",
                    title="Reference %02d: coverage" % rid, ylim=(0, 1.02))
        axes[1].bar(range(6), [bin_rows[-6 + j]["true_top_count"] for j in range(6)])
        axes[1].set(xticks=range(6), xticklabels=["1k", "2k", "3k", "4k", "5k", "rest"],
                    xlabel="Proxy rank bin", ylabel="True top-k windows",
                    title="Reference %02d: rank location" % rid)
        fig.tight_layout()
        fig.savefig(out / "figures" / ("reference-%02d.png" % rid), dpi=130)
        plt.close(fig)
        if (rid + 1) % 10 == 0:
            elapsed = time.monotonic() - started
            log.info("Coverage complete: %d/100 elapsed=%s eta=%s", rid + 1,
                     format_duration(elapsed), format_duration(elapsed * (99 - rid) / (rid + 1)))
    write_csv(out / "true_top_proxy_ranks.csv", rank_rows)
    write_csv(out / "coverage.csv", coverage_rows)
    write_csv(out / "rank_bins.csv", bin_rows)
    summary = []
    for multiplier in (1, 2, 3, 4, 5, 7):
        rows = [r for r in coverage_rows if r["proxy_candidates"] == min(r["candidates"], multiplier * r["k"])]
        summary.append(dict(proxy_k_multiple=multiplier, references=len(rows),
                            mean_recall=float(np.mean([r["recall"] for r in rows])),
                            min_recall=float(np.min([r["recall"] for r in rows])),
                            median_recall=float(np.median([r["recall"] for r in rows]))))
    write_csv(out / "coverage_summary.csv", summary)
    (out / "input_manifest.json").write_text(json.dumps(dict(batch="3-proxy-coverage",
        source="batch-1", target_fraction=0.10, rounding="ceil", tie_break="stable_input_order",
        rank_bins="k,2k,3k,4k,5k,n", reference_count=100), indent=2) + "\n", encoding="ascii")
    (out / ".complete").write_text("complete\n", encoding="ascii")
    log.info("Stage complete: proxy coverage; rows=%d duration=%s output=%s",
             len(coverage_rows), format_duration(time.monotonic() - started), out)


def boundary_order(proxy_order, k):
    n = len(proxy_order)
    ranks = []
    for distance in range(n):
        if distance == 0:
            ranks.append(k - 1)
        else:
            if k - 1 + distance < n:
                ranks.append(k - 1 + distance)
            if k - 1 - distance >= 0:
                ranks.append(k - 1 - distance)
    return proxy_order[np.asarray(ranks, dtype=int)]


def evaluate(true, proxy, oracle, selected):
    predicted = proxy.copy()
    predicted[selected] = true[selected]
    chosen = np.argsort(-predicted, kind="stable")[:len(oracle)]
    return float(np.intersect1d(chosen, oracle).size / len(oracle))


def compare(out, log):
    refs, arrays = inputs()
    source = BASE / "batch-2"
    if not (source / ".complete").is_file():
        raise ValueError("complex method source incomplete")
    out.mkdir(parents=True, exist_ok=True)
    with (source / "selection_metrics.csv").open(newline="", encoding="ascii") as stream:
        prior = list(csv.DictReader(stream))
    complex_rows = {(int(r["reference_id"]), int(r["budget"])): r for r in prior
                    if r["method"] == "boundary" and float(r["fraction"]) == 0.10}
    query_rows = json.loads((source / "query_manifest.json").read_text(encoding="ascii"))
    complex_queries = {r["reference_id"]: r for r in query_rows if r["method"] == "boundary"}
    rows, query_records = [], []
    started = time.monotonic()
    log.info("Stage start: equal-budget replay; references=100 budgets=%s random_seeds=%d output=%s",
             BUDGETS, len(SEEDS), out)
    for ref in refs:
        rid = ref["reference_id"]
        true, proxy, n, k, true_order, proxy_order = orders(ref, arrays)
        oracle = true_order[:k]
        initial = np.argsort(proxy, kind="stable")[np.linspace(0, n - 1, 8, dtype=int)]
        if list(map(int, initial)) != complex_queries[rid]["initial"]:
            raise ValueError("initial labels differ from existing complex replay")
        base_recall = evaluate(true, proxy, oracle, np.array([], dtype=int))
        rows.append(dict(reference_id=rid, reference_track=ref["track"], candidates=n, k=k,
                         method="proxy_only", budget=0, seed=-1, recall=base_recall))
        for budget in BUDGETS[1:]:
            old = complex_rows[(rid, budget)]
            rows.append(dict(reference_id=rid, reference_track=ref["track"], candidates=n, k=k,
                             method="complex_boundary", budget=budget, seed=-1,
                             recall=float(old["recall"])))
        policies = dict(pure_boundary=boundary_order(proxy_order, k),
                        proxy_high=proxy_order)
        for method, order in policies.items():
            selected = list(map(int, initial))
            selected_set = set(selected)
            selected.extend(int(i) for i in order if int(i) not in selected_set)
            query_records.append(dict(reference_id=rid, method=method, seed=-1,
                                      query_order=selected[:64]))
            for budget in BUDGETS[1:]:
                recall = evaluate(true, proxy, oracle, np.asarray(selected[:budget]))
                rows.append(dict(reference_id=rid, reference_track=ref["track"], candidates=n,
                                 k=k, method=method, budget=budget, seed=-1, recall=recall))
        for seed in SEEDS:
            random_order = np.random.default_rng(100000 * seed + rid + 1000).permutation(n)
            selected = list(map(int, initial))
            selected_set = set(selected)
            selected.extend(int(i) for i in random_order if int(i) not in selected_set)
            query_records.append(dict(reference_id=rid, method="random", seed=seed,
                                      query_order=selected[:64]))
            for budget in BUDGETS[1:]:
                recall = evaluate(true, proxy, oracle, np.asarray(selected[:budget]))
                rows.append(dict(reference_id=rid, reference_track=ref["track"], candidates=n,
                                 k=k, method="random", budget=budget, seed=seed, recall=recall))
        if (rid + 1) % 10 == 0:
            elapsed = time.monotonic() - started
            log.info("Comparison complete: %d/100 elapsed=%s eta=%s", rid + 1,
                     format_duration(elapsed), format_duration(elapsed * (99 - rid) / (rid + 1)))
    write_csv(out / "per_reference_recall.csv", rows)
    (out / "query_manifest.json").write_text(json.dumps(query_records, indent=2) + "\n", encoding="ascii")
    summary = []
    for method in ("proxy_only", "pure_boundary", "proxy_high", "random", "complex_boundary"):
        for budget in (BUDGETS if method == "proxy_only" else BUDGETS[1:]):
            group = [r for r in rows if r["method"] == method and r["budget"] == budget]
            if not group:
                continue
            if method == "random":
                reference_values = [np.mean([r["recall"] for r in group if r["reference_id"] == rid])
                                    for rid in range(100)]
                seed_means = [np.mean([r["recall"] for r in group if r["seed"] == seed])
                              for seed in SEEDS]
                seed_sd = float(np.std(seed_means, ddof=1))
            else:
                reference_values = [r["recall"] for r in group]
                seed_sd = 0.0
            summary.append(dict(method=method, budget=budget, references=100,
                                mean_recall=float(np.mean(reference_values)),
                                min_reference_recall=float(np.min(reference_values)),
                                median_reference_recall=float(np.median(reference_values)),
                                seed_mean_sd=seed_sd))
    write_csv(out / "summary.csv", summary)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    base = next(r["mean_recall"] for r in summary if r["method"] == "proxy_only")
    for method, label in (("pure_boundary", "Pure boundary"), ("proxy_high", "Proxy high"),
                          ("random", "Random (30 seeds)"), ("complex_boundary", "Existing complex")):
        group = [r for r in summary if r["method"] == method]
        ax.plot([0] + [r["budget"] for r in group], [base] + [r["mean_recall"] for r in group],
                marker="o", label=label)
    ax.set(xlabel="Full-score query budget", ylabel="Recall@k", xticks=BUDGETS, ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "method_comparison.png", dpi=160)
    plt.close(fig)
    (out / "input_manifest.json").write_text(json.dumps(dict(batch="4-equal-budget-query",
        source="batch-1", complex_source="batch-2 boundary policy", budgets=BUDGETS,
        random_seeds=SEEDS, initial_labels="same eight proxy-quantile labels as batch 2; counted in budget",
        unqueried_scores="raw head cosine for simple policies; existing ridge correction for complex policy",
        queried_scores="cached full cosine", target_fraction=0.10, rounding="ceil",
        tie_break="stable_input_order", scale="both scores are cosine; no additional calibration for simple policies"),
        indent=2) + "\n", encoding="ascii")
    (out / ".complete").write_text("complete\n", encoding="ascii")
    log.info("Stage complete: equal-budget replay; rows=%d duration=%s output=%s",
             len(rows), format_duration(time.monotonic() - started), out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, choices=(3, 4), required=True)
    args = parser.parse_args()
    out = BASE / ("batch-%d" % args.batch)
    if (out / ".complete").exists():
        raise ValueError("batch already complete")
    log = configure_logger("v0.active.value.diagnostics", "v0-active-value-rank",
                           "batch-%d" % args.batch, "initialization")
    log.info("Configuration: batch=%d output=%s", args.batch, out)
    (diagnose if args.batch == 3 else compare)(out, log)


if __name__ == "__main__":
    main()
