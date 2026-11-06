"""Shared selection dispatch for student-native development experiments."""
from .arms import rage_crb_select, streaming_random_select, trajectory_level_select
from .train_eval import crb_gradient_computer


def full_pool_control(pool, index):
    import numpy as np
    from .arms import ArmOutput, SelectionCost
    from .regions import RegionWindows
    length = index.context_length + index.prediction_length
    eligible = [f for f in pool if f.length >= length]
    offsets = np.cumsum([0] + [f.length for f in eligible])
    regions = [(f.fragment_id, 0, f.length) for f in eligible]
    windows = RegionWindows(np.concatenate([f.data for f in eligible]), offsets, regions,
        window_length=length, budget_windows=int(np.ceil(offsets[-1]/length)))
    return ArmOutput('full_pool', windows, [], SelectionCost(),
        dict(control='unselected full source pool; uniform legal window sampling',
             selection_budget_matched=False, update_budget_matched=True,
             actual_pool_points=int(offsets[-1])))


ARMS = ("rage", "random", "trajectory_level", "less", "tsds", "lp_mirror", "ltsv")


def select_arm(name, probe, reference, pool, index, *, n_select, **overrides):
    """Use existing algorithms and delivery; no persistent feature cache."""
    common = dict(context_length=index.context_length, prediction_length=index.prediction_length,
                  n_select=n_select)
    if name == 'full_pool':
        return full_pool_control(pool, index)
    if name == "random":
        return streaming_random_select(index, n_select=n_select)
    if name == "rage":
        return rage_crb_select(reference, pool, index,
            gradient_computer=crb_gradient_computer(probe, index.context_length, index.prediction_length),
            window_stride=index.stride, cover_density=.02, error_density=.02, split_density=.005,
            block_budget=8, mean_epsilon=1e-8, endpoint_epsilon=1e-8, tie_seed=5, **common)
    if name == "trajectory_level":
        return trajectory_level_select(reference, pool, index, engine=probe, **common)
    from baselines.runners import run_baseline
    return run_baseline(name, engine=probe, reference_fragments=reference, pool=pool,
                        index=index, seed=7, **common, **overrides)
