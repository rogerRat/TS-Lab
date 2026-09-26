"""Record baseline-specific acceleration separately from comparison runtime."""
import json
import math
from pathlib import Path


def load_restore_timing(cache_dir, model):
    if cache_dir is None:
        return None
    path = Path(cache_dir) / 'ltsv_restore_timing.json'
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if data['schema'] != 'ltsv-restore-timing-v1' or data['model'] != model:
        raise ValueError('LTSV timing calibration/model fields differ')
    for key in ('cpu_seconds', 'gpu_seconds'):
        if not math.isfinite(data[key]) or data[key] <= 0:
            raise ValueError('LTSV timing calibration must be finite and positive')
    return data


def accounting(calibration, blocks, setup_seconds):
    """Only the restored-weight component is adjusted, never the whole method."""
    if calibration is None:
        return None
    return dict(schema='baseline-runtime-adjustment-v1',
        optimization='ltsv_gpu_weight_restore', optimized_blocks=int(blocks),
        extra_setup_seconds=float(setup_seconds), calibration=calibration,
        add_back_seconds=blocks * (calibration['cpu_seconds'] - calibration['gpu_seconds'])
                         - setup_seconds,
        report_basis='estimated_without_baseline_specific_optimization',
        framework='same CUDA MPS and concurrency; not reversed',
        scope='this framework adapter without GPU restore, not an upstream-code wall-time measurement',
        formula='reported = actual - extra_setup + blocks * (CPU_restore - GPU_restore)',
        uncertainty='small-sample component estimate; assumes unchanged non-restore work and contention')


def comparison_seconds(actual_seconds, adjustment):
    if adjustment is None:
        return float(actual_seconds)
    result = actual_seconds + adjustment['add_back_seconds']
    if not math.isfinite(result) or result < 0:
        raise ValueError('Invalid baseline counterfactual runtime')
    return float(result)
