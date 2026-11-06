"""Middle-layer W/S long-context scoring over a fixed candidate matrix."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from adapters.foundation.components import cosine_to_reference, grouped_mean_sketches

from .deploy import cut_disjoint


@dataclass(frozen=True)
class LongContextSelection:
    selected_rows: np.ndarray
    selected_ids: np.ndarray
    event_scores: np.ndarray
    aggregation: str
    diagnostics: dict[str, object]


def _gradient_sketches(engine, windows: np.ndarray, *, batch_size: int) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for begin in range(0, len(windows), batch_size):
        _loss, sketch = engine.batch_gradient_sketch(windows[begin:begin + batch_size])
        values = np.asarray(sketch, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("long-context engine must return one [N,D] sketch matrix")
        chunks.append(values)
    if not chunks:
        raise ValueError("long-context scoring requires non-empty windows")
    return np.concatenate(chunks)


def select_long_context_candidates(
    engine,
    *,
    source_windows: np.ndarray,
    source_ids: np.ndarray,
    source_groups: np.ndarray,
    reference_windows: np.ndarray,
    reference_groups: np.ndarray,
    n_select: int,
    deployment_window_length: int = 576,
    aggregation: str = "window",
    batch_size: int = 1,
    seed: int = 5,
) -> LongContextSelection:
    """Select supplied static rows under W (window) or S (segment) scoring."""
    if n_select <= 0 or deployment_window_length <= 0 or batch_size <= 0:
        raise ValueError("selection sizes and batch_size must be positive")
    windows = np.asarray(source_windows, dtype=np.float32)
    identities = np.asarray(source_ids, dtype=np.int64)
    groups = np.asarray(source_groups)
    references = np.asarray(reference_windows, dtype=np.float32)
    reference_group_ids = np.asarray(reference_groups)
    if windows.ndim != 2 or references.ndim != 2 or windows.shape[1] != references.shape[1]:
        raise ValueError("source/reference long-context windows must share [N,L] geometry")
    if identities.shape != (len(windows), 2) or groups.shape != (len(windows),):
        raise ValueError("source IDs/groups must align one-to-one with source windows")
    if reference_group_ids.shape != (len(references),):
        raise ValueError("reference groups must align one-to-one with reference windows")
    if len(np.unique(identities, axis=0)) != len(identities):
        raise ValueError("static source identities must be unique")
    if aggregation not in {"window", "segment"}:
        raise ValueError("aggregation must be window or segment")

    source_sketches = _gradient_sketches(engine, windows, batch_size=batch_size)
    reference_sketches = _gradient_sketches(engine, references, batch_size=batch_size)
    if aggregation == "window":
        event_scores = cosine_to_reference(source_sketches, reference_sketches)
        group_count = len(windows)
    else:
        labels, means = grouped_mean_sketches(source_sketches, groups)
        _reference_labels, reference_means = grouped_mean_sketches(
            reference_sketches, reference_group_ids,
        )
        group_scores = cosine_to_reference(means, reference_means)
        score_by_group = {label.item() if hasattr(label, "item") else label: score
                          for label, score in zip(labels, group_scores)}
        event_scores = np.asarray([score_by_group[item] for item in groups], dtype=np.float32)
        group_count = len(labels)

    rng = np.random.default_rng(seed)
    chosen = cut_disjoint(
        identities[:, 0], identities[:, 1], event_scores,
        window_length=deployment_window_length, n=n_select,
        tiebreak=rng.random(len(event_scores)),
    )
    return LongContextSelection(
        selected_rows=chosen,
        selected_ids=identities[chosen].copy(),
        event_scores=event_scores,
        aggregation=aggregation,
        diagnostics={
            "source_events": len(windows),
            "reference_events": len(references),
            "reference_groups": len(dict.fromkeys(reference_group_ids.tolist())),
            "aggregation_groups": group_count,
            "selected": len(chosen),
            "gradient_computations": len(windows) + len(references),
        },
    )


__all__ = ["LongContextSelection", "select_long_context_candidates"]
