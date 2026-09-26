"""Observed gradient latency records, isolated by explicit execution fields."""
from dataclasses import asdict
import platform
import torch


def timing_identity(manifest, engine):
    from rags.execution import execution_policy
    device = torch.device(engine.device)
    return dict(model=manifest['model'], probe=manifest['probe'],
        gradient_protocol=manifest['gradient_protocol'],
        context_length=manifest['context_length'], prediction_length=manifest['prediction_length'],
        execution=asdict(execution_policy(engine)), device=str(device),
        device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
        host=platform.node(), torch_version=torch.__version__,
        engine_type=type(engine).__module__ + '.' + type(engine).__qualname__)


def benchmark_summary(record):
    windows, seconds = record['gradient_windows'], record['gradient_seconds']
    return dict(record, mean_seconds_per_window=seconds/windows if windows and seconds > 0 else None,
        boundary='batch_gradient_sketch call through materialized NumPy return; cache hits excluded')
