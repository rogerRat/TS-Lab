"""Discard completed-cell training state only after all metrics are durable."""
import json
import re

from .students import atomic_json


def cleanup_completed_weights(result_path, checkpoint_dir):
    record = json.loads(result_path.read_text())
    for path in checkpoint_dir.iterdir() if checkpoint_dir.exists() else ():
        if re.fullmatch(r"epoch-[0-9]+(?:\.[0-9]+)?\.pt|resume\.pt(?:\.tmp)?", path.name):
            path.unlink()
    record["checkpoint_retention"] = "completed_cell_cleanup"
    record["training_weights_retained"] = False
    atomic_json(result_path, record)
