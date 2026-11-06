"""Unified dispatch only; implementations live in per-baseline packages."""
from __future__ import annotations
import time
from rage.experiments.arms import ArmOutput, WindowIndex
from .contracts import baseline_contract, verify_static_identities
from .less.runner import less_select
from .tsds.runner import tsds_select
from .lp_mirror.runner import lp_mirror_select
from .ltsv.runner import ltsv_select
from .common import reference_windows

BASELINE_RUNNERS = {
    "less": less_select, "tsds": tsds_select,
    "lp_mirror": lp_mirror_select, "ltsv": ltsv_select,
}

def run_baseline(name: str, **kwargs) -> ArmOutput:
    import inspect

    runner = BASELINE_RUNNERS.get(name)
    if runner is None:
        raise ValueError("unknown baseline: %s" % name)
    accepted = inspect.signature(runner).parameters
    filtered = {key: value for key, value in kwargs.items() if key in accepted}
    started = time.perf_counter()
    output = runner(**filtered)
    output.cost.wall_seconds = time.perf_counter() - started
    contract = baseline_contract(name)
    index = kwargs.get("index")
    if not isinstance(index, WindowIndex):
        raise TypeError("external baselines require the shared static WindowIndex")
    from rage.experiments.regions import RegionWindows
    if isinstance(output.selected, RegionWindows):
        for fid, start, stop in output.selected.regions:
            fragment = index.by_fragment[int(fid)]
            if start < 0 or stop > fragment.length:
                raise ValueError("baseline region is outside the authorized source pool")
    elif output.selected.shape[0] != len(output.window_ids):
        raise ValueError("baseline output rows and window identities disagree")
    verify_static_identities(output.window_ids, index.locate)
    output.diagnostics["baseline_contract"] = contract.as_mapping()
    output.diagnostics["static_framework_verified"] = True
    return output
