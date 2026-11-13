"""Generate the explicit user-approved training amendment, preserving asset identity."""
import json
from pathlib import Path
from rage.experiments.v7_protocol import make_reduced_config, validate

if __name__ == '__main__':
    path=Path('configs/rage/main_v7_curve4.json')
    config=make_reduced_config(json.loads(Path('configs/rage/main_v7.json').read_text()))
    validate(config)
    if path.exists() and json.loads(path.read_text())!=config:
        raise ValueError('Refusing to overwrite a different existing training amendment')
    path.write_text(json.dumps(config,indent=2)+'\n')
