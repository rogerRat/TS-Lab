"""Per-invocation selection timing and PyTorch peak memory, shared by all arms."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time
from uuid import uuid4

import torch


class SelectionMetrics:
    """Append observations so resume never overwrites the original measurement.

    Memory is the process's peak allocated tensor memory on the probe device,
    including the resident probe. It excludes CUDA allocator reserves and other
    processes. Only coarse timing boundaries synchronize CUDA.
    """

    def __init__(self, out_dir, *, device, protocol):
        self.path = Path(out_dir) / "selection_metrics.jsonl"
        self.device = torch.device(device)
        self.cuda = self.device.type == "cuda"
        self.common = {
            "schema": "rags.selection-metrics.v1",
            "invocation_id": uuid4().hex,
            "run_directory": str(Path(out_dir).resolve()),
            "host": platform.node(),
            "device": str(self.device),
            "gpu_name": torch.cuda.get_device_name(self.device) if self.cuda else None,
            "torch_version": torch.__version__,
            "protocol": protocol,
        }

    def _write(self, dataset, arm, *, cache_state, seconds, peak_gib, gradient_cost=None):
        row = dict(self.common, recorded_at=datetime.now(timezone.utc).isoformat(),
                   dataset=dataset, arm=arm, cache_state=cache_state,
                   selection_seconds=seconds, peak_gpu_memory_gib=peak_gib)
        if gradient_cost is not None:
            row["gradient_cost"] = gradient_cost
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")

    def original(self, dataset, *, arm="original"):
        # No selection operation takes place for the pretrained reference arm.
        self._write(dataset, arm, cache_state="not_applicable",
                    seconds=0.0, peak_gib=0.0)

    def measure(self, dataset, arm, build, *, selection_cached=False, gradient_cost=None):
        if self.cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        output = build()
        if self.cuda:
            torch.cuda.synchronize(self.device)
        seconds = time.perf_counter() - started
        peak_gib = (torch.cuda.max_memory_allocated(self.device) / 2**30
                    if self.cuda else 0.0)
        if selection_cached:
            state = "selection_cache_hit"
        else:
            state = "warm" if any(output.cost.notes.get(key, False) for key in
                                  ("pool_cached", "proxy_cached", "scores_cached")) else "cold"
            output.cost.wall_seconds = seconds
        cost_record = None
        if gradient_cost is not None:
            from .gradient_cost import gradient_cost_record
            cost_record = gradient_cost_record(**gradient_cost, selection_seconds=seconds,
                                               selection_cached=selection_cached)
            if not selection_cached:
                output.cost.notes['gradient_cost'] = cost_record
        self._write(dataset, arm, cache_state=state, seconds=seconds, peak_gib=peak_gib,
                    gradient_cost=cost_record)
        return output
