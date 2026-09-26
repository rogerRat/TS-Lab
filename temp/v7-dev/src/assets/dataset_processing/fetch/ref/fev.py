"""Materialize FEV task data into the spec-owned Hugging Face cache root."""

from __future__ import annotations

import os
from pathlib import Path

from ts3.spec.ref_extraction import ReferenceExtractionSpecification


def fetch(spec: ReferenceExtractionSpecification, declaration_path: Path, data_root: Path) -> None:
    """Use FEV's official task loader so its declared datasets enter one local cache."""
    data_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(data_root)
    from evaluate.fev import load_task_declarations
    import fev

    tasks = load_task_declarations(declaration_path)
    selection = spec.mapping["task_selection"]
    indices = range(len(tasks)) if selection["mode"] == "all" else [int(item) for item in selection["items"]]
    for index in indices:
        task = fev.Task(**tasks[index])
        task.get_window(0, num_proc=1).get_input_data()
