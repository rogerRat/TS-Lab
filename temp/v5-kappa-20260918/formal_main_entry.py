"""Run the reviewed shared-ten matrix after queue prerequisites succeed."""
import argparse
import json
from pathlib import Path
from .full_pool_run import run_suite, validate_config
from rage.epsilon_protocol import load_p10_calibration


def preflight(config, readiness):
    if not readiness.get('ready') or readiness.get('config') != config:
        raise ValueError('Queued formal config differs from the reviewed ready configuration')
    validate_config(config)
    if config['crb'].get('epsilon_policy') in ('trajectory_p10', 'none'):
        return  # Computed independently inside each CRB trajectory, no external file.
    for student in config['students']:
        path = Path(config['settings'][student]['epsilon_calibration'])
        report = json.loads(path.read_text())
        if student == 'chronos-base':
            model_id,revision = 'amazon/chronos-t5-base','ad294eaacead15db499b740ea4122266dd2a81a2'
        else:
            from adapters.foundation.spec import get_model_spec
            spec = get_model_spec(student)
            model_id,revision = spec.model_id,spec.revision
        identity = dict(student=student,model_id=model_id,revision=revision,
                        ckpt=config['model_paths'][student],projection_dimension=4096,
                        projection_seed=42,precision_policy='bf16-model-gradient-adamw-v2')
        load_p10_calibration(path, identity)
        if report['model'].get('student') != student or report['model'].get('loaded_parameter_dtypes') != ['torch.bfloat16']:
            raise ValueError('Formal P10 student or BF16 identity mismatch')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    ready=json.loads((args.run/'readiness.json').read_text())
    preflight(config,ready)
    return run_suite(config,args.run)


if __name__=='__main__':
    raise SystemExit(main())
