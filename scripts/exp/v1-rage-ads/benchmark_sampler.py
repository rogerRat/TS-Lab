#!/usr/bin/env python3
"""Measure dense versus compact source-distribution sampling overhead."""
from __future__ import annotations

import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

from configs.logger import configure_logger, format_duration
from configs.paths import PROJECT_ROOT, output_path

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT / "scripts/exp/v1-rage-ads"))
from train_eval import load_source
from rage.ads import SourceProbabilitySampler


def main():
    logger = configure_logger("v1_rage_ads_sampler_benchmark", "v1-rage-ads",
                              "sampler-benchmark", "sampling-profile")
    started = time.monotonic()
    values, legal, value_offsets, legal_offsets = load_source()
    task_dir = ROOT / "outputs/selection/v1-rage-ads/rolling/fev-09/rage"
    manifest_path = task_dir / "probability.json"
    descriptor = json.loads(manifest_path.read_text(encoding="ascii"))
    repetitions, batch_size = 400, 32
    total = int(legal_offsets[-1])
    logger.info("stage=config mode=fev-09-rage total_windows=%d repetitions=%d batch=%d",
                total, repetitions, batch_size)

    build_started = time.monotonic()
    compact_for_dense = SourceProbabilitySampler(descriptor)
    dense = np.zeros(total, dtype=np.float64)
    if compact_for_dense.mode == "uniform":
        dense.fill(1.0 / total)
    else:
        probabilities = np.diff(np.r_[0.0, compact_for_dense.cumulative_mass])
        dense[compact_for_dense.support_rows] = probabilities
    dense_build_seconds = time.monotonic() - build_started
    if len(dense) != total or not np.isclose(dense.sum(), 1.0):
        raise ValueError("dense ADS probability baseline is invalid")
    dense_rng = np.random.default_rng(42)
    dense_started = time.monotonic()
    for _ in range(repetitions):
        dense_rng.choice(total, size=batch_size, replace=True, p=dense)
    dense_draw_seconds = time.monotonic() - dense_started

    compact_started = time.monotonic()
    compact = SourceProbabilitySampler(descriptor)
    compact_build_seconds = time.monotonic() - compact_started
    compact_rng = np.random.default_rng(42)
    compact_started = time.monotonic()
    for _ in range(repetitions):
        fragment_ids, starts = compact.sample(compact_rng, batch_size)
        owners = np.searchsorted(compact.fragment_ids, fragment_ids)
        if (len(owners) != batch_size or np.any(owners >= len(compact.counts))
                or np.any(starts < 1)):
            raise ValueError("compact ADS draws failed source identity checks")
    compact_draw_seconds = time.monotonic() - compact_started
    dense_mib = dense.nbytes / (1024 ** 2)
    compact_bytes = (compact.support_rows.nbytes + compact.cumulative_mass.nbytes
                     + compact.counts.nbytes + compact.fragment_ids.nbytes
                     + compact.first_starts.nbytes + compact.strides.nbytes)
    compact_mib = compact_bytes / (1024 ** 2)
    result = dict(
        experiment="v1-rage-ads", phase="training-sampler-cpu",
        task="fev-09", arm="rage", source_windows=total,
        repetitions=repetitions, batch_size=batch_size,
        dense_probability_build_seconds=dense_build_seconds,
        dense_probability_vector_mib=dense_mib,
        dense_draw_seconds=dense_draw_seconds,
        dense_draws_per_second=repetitions * batch_size / dense_draw_seconds,
        compact_table_build_seconds=compact_build_seconds,
        compact_table_mib=compact_mib,
        compact_draw_seconds=compact_draw_seconds,
        compact_draws_per_second=repetitions * batch_size / compact_draw_seconds,
        draw_speedup=dense_draw_seconds / compact_draw_seconds,
        max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        seed=42,
    )
    output = output_path("selection", ROOT / "outputs/selection/v1-rage-ads/parallelism")
    output.mkdir(parents=True, exist_ok=True)
    target = output / "sampler-benchmark.json"
    temporary = target.with_name(target.name + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                         encoding="ascii")
    os.replace(temporary, target)
    (output / "sampler-benchmark.complete").write_text("complete\n", encoding="ascii")
    logger.info("stage=complete dense_draws_per_second=%.1f compact_draws_per_second=%.1f speedup=%.2fx dense_mib=%.1f compact_mib=%.1f elapsed=%s output=%s",
                result["dense_draws_per_second"], result["compact_draws_per_second"],
                result["draw_speedup"], dense_mib, compact_mib,
                format_duration(time.monotonic() - started), target)


if __name__ == "__main__":
    main()
