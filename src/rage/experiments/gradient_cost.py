"""Separate independent gradient requirements from shared-cache incremental work."""
import math


def gradient_cost_record(*, required_source, required_reference, benchmark,
                         preparation_new_windows, preparation_seconds,
                         preparation_gradient_seconds, selection_seconds,
                         method_preparation_nongrad_seconds=None,
                         preparation_scope='shared_union', selection_cached=False):
    counts = [required_source, required_reference, preparation_new_windows]
    if any(int(x) != x or x < 0 for x in counts):
        raise ValueError('gradient cost counts must be nonnegative integers')
    times = [preparation_seconds, preparation_gradient_seconds, selection_seconds]
    if method_preparation_nongrad_seconds is not None:
        times.append(method_preparation_nongrad_seconds)
    if any(not math.isfinite(t) or t < 0 for t in times):
        raise ValueError('gradient cost durations must be finite and nonnegative')
    if preparation_gradient_seconds > preparation_seconds + 1e-6:
        raise ValueError('gradient-call time cannot exceed enclosing preparation time')
    required = int(required_source + required_reference)
    average = benchmark.get('mean_seconds_per_window') if benchmark else None
    if average is not None and (not math.isfinite(average) or average <= 0):
        raise ValueError('observed mean gradient latency must be positive')
    gradient_estimate = required * average if average is not None else (0. if required == 0 else None)
    nongrad = None if selection_cached else selection_seconds + (method_preparation_nongrad_seconds or 0.)
    estimate = gradient_estimate + nongrad if gradient_estimate is not None and nongrad is not None else None
    return dict(schema='gradient-selection-cost-v1', required_source=int(required_source),
        required_reference=int(required_reference), required_total=required,
        benchmark=benchmark, estimated_required_gradient_seconds=gradient_estimate,
        measured_selection_body_seconds=selection_seconds,
        method_preparation_nongrad_seconds=method_preparation_nongrad_seconds,
        estimated_cold_selection_seconds=estimate,
        estimate_scope='required gradient plus measured method preparation and selection; shared initialization excluded',
        estimate_complete_method_stages=method_preparation_nongrad_seconds is not None and not selection_cached,
        unavailable_reason=('selection cache load is not a selection recomputation' if selection_cached else
            'no matching observed gradient latency' if gradient_estimate is None else None),
        preparation=dict(scope=preparation_scope, new_windows=int(preparation_new_windows),
            elapsed_seconds=preparation_seconds, gradient_seconds=preparation_gradient_seconds,
            nongrad_seconds=max(0., preparation_seconds - preparation_gradient_seconds)),
        accounting='required cost is per method; preparation increment is shared and must not be summed across methods')
