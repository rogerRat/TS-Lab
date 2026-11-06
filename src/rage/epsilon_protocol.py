"""Frozen p10 classification thresholds shared by production and probes."""
import json
import math
from pathlib import Path


def load_p10_calibration(path, identity):
    report = json.loads(Path(path).read_text())
    if report.get('percentile') != 10 or report.get('count', 0) <= 0:
        raise ValueError('Classification requires a frozen p10 calibration')
    epsilon = float(report['epsilon'])
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError('Invalid p10 threshold')
    for key in ('model_id', 'revision', 'projection_dimension', 'projection_seed',
                'precision_policy', 'ckpt'):
        if key not in report['model'] or key not in identity or report['model'][key] != identity[key]:
            raise ValueError(f'P10 calibration model/projection/precision mismatch: {key}')
    if not math.isclose(float(report['distribution']['10']), epsilon, rel_tol=1e-12, abs_tol=0):
        raise ValueError('Classification threshold differs from recorded p10')
    return report
