"""Whole-unit delivery and lazy, uniform sampling of legal window occurrences."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

REVISION = "point-budget-regions-v1"


def check_run_revision(out_dir, args=None):
    path = Path(out_dir) / "protocol.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get("delivery_revision") != REVISION:
            raise ValueError("Delivery protocol changed; use a new run directory")
        if args is not None and any(previous.get(key) != getattr(args, key) for key in
                                    ("cover_density", "error_density", "split_density")):
            raise ValueError("Sampling budget changed; use a new run directory")


class RegionWindows:
    """Packed delivered regions; duplicate regions remain distinct occurrences.

    Region identity is (source fragment ID, zero-based start, exclusive stop).
    Legal windows never cross a region boundary. No dense window matrix exists.
    """
    def __init__(self, values, offsets, regions, *, window_length, budget_windows):
        self.values = np.asarray(values, dtype=np.float32)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.regions = np.asarray(regions, dtype=np.int64).reshape(-1, 3)
        self.window_length = int(window_length)
        self.budget_windows = int(budget_windows)
        self.window_offsets = np.cumsum(np.r_[0, np.diff(self.offsets) - self.window_length + 1])

    def __len__(self):
        return int(self.window_offsets[-1])

    def identities(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        owners = np.searchsorted(self.window_offsets[1:], rows, side="right")
        starts = rows - self.window_offsets[owners]
        return owners, starts

    def __getitem__(self, rows):
        if isinstance(rows, slice):
            rows = np.arange(*rows.indices(len(self)))
        owners, starts = self.identities(rows)
        locations = self.offsets[owners] + starts
        return self.values[locations[..., None] + np.arange(self.window_length)]

    def prefix(self, n):
        count = prefix_count(np.diff(self.offsets), n * self.window_length)
        return RegionWindows(self.values[:self.offsets[count]], self.offsets[:count + 1],
                             self.regions[:count], window_length=self.window_length, budget_windows=n)

    def coverage(self):
        budget = self.budget_windows * self.window_length
        delivered = int(self.offsets[-1])
        unique = 0
        previous, end = None, 0
        for fid, start, stop in sorted(map(tuple, self.regions.tolist())):
            if fid != previous:
                end = start
            unique += max(0, stop - max(start, end))
            end, previous = max(end, stop), fid
        return {"budget_windows": self.budget_windows, "budget_points": budget,
                "delivered_units": len(self.regions), "delivered_points": delivered,
                "unique_points": unique, "repeated_points": delivered - unique,
                "overflow_points": max(0, delivered - budget),
                "overflow_ratio": max(0, delivered - budget) / budget,
                "legal_window_occurrences": len(self)}


def prefix_count(lengths, budget):
    cumulative = np.cumsum(lengths)
    if not len(cumulative) or cumulative[-1] < budget:
        raise ValueError(f"Delivered candidates cannot fill point budget {budget}")
    return int(np.searchsorted(cumulative, budget, side="left")) + 1


def deliver_regions(index, regions, *, n):
    rows = np.asarray(regions, dtype=np.int64).reshape(-1, 3)
    count = prefix_count(rows[:, 2] - rows[:, 1], n * (index.context_length + index.prediction_length))
    rows = rows[:count]
    arrays = [index.by_fragment[int(fid)].data[start:stop] for fid, start, stop in rows]
    offsets = np.cumsum([0] + [len(a) for a in arrays])
    return RegionWindows(np.concatenate(arrays), offsets, rows,
                         window_length=index.context_length + index.prediction_length, budget_windows=n)


def deliver_trajectories(index, trajectory_order, *, n):
    return deliver_regions(index, [(fid, 0, index.by_fragment[int(fid)].length)
                                   for fid in trajectory_order], n=n)


def block_regions(index, positions_by_block):
    width = index.context_length + index.prediction_length
    return [(int(index.fragment_ids[p[0]]), int(index.starts[p[0]]) - 1,
             int(index.starts[p[-1]]) - 1 + width) for p in positions_by_block]


def save_selection(path, output):
    from .resampling import ResampledWindows, REVISION as RESAMPLING_REVISION
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"name": output.name, "schema": REVISION, "cost": json.dumps(vars(output.cost)),
            "diagnostics": json.dumps(output.diagnostics)}
    if isinstance(output.selected, ResampledWindows):
        data.update(output.selected.payload())
        data["schema"] = RESAMPLING_REVISION
    elif isinstance(output.selected, RegionWindows):
        selection = output.selected
        data.update(values=selection.values, offsets=selection.offsets, regions=selection.regions,
                    window_length=selection.window_length, budget_windows=selection.budget_windows)
    else:
        from .deploy import selection_coverage
        if len(output.selected):
            output.diagnostics["delivery"] = selection_coverage(output.window_ids, output.selected.shape[1])
            data["diagnostics"] = json.dumps(output.diagnostics)
        data.update(windows=output.selected, ids=np.asarray(output.window_ids, dtype=np.int64))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **data)
    temporary.replace(path)


def load_selection(path):
    from .arms import ArmOutput, SelectionCost
    from .resampling import ResampledWindows, REVISION as RESAMPLING_REVISION
    with np.load(path, allow_pickle=False) as payload:
        if "schema" in payload and str(payload["schema"]) == RESAMPLING_REVISION:
            return ArmOutput(str(payload["name"]), ResampledWindows.from_payload(payload), [],
                             SelectionCost(**json.loads(str(payload["cost"]))),
                             json.loads(str(payload["diagnostics"])))
        if "schema" not in payload or str(payload["schema"]) != REVISION:
            raise ValueError("Old selection protocol: create a new point-budget run")
        if "regions" in payload:
            selected = RegionWindows(payload["values"], payload["offsets"], payload["regions"],
                                     window_length=int(payload["window_length"]),
                                     budget_windows=int(payload["budget_windows"]))
            ids = []
        else:
            selected, ids = payload["windows"], [tuple(map(int, row)) for row in payload["ids"]]
        return ArmOutput(str(payload["name"]), selected, ids,
                         SelectionCost(**json.loads(str(payload["cost"]))),
                         json.loads(str(payload["diagnostics"])))
