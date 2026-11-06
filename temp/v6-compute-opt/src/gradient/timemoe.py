"""TimeMoE native per-window gradients: persistent GPU workers or CUDA streams."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from gradient.base import NativeProbe, IndependentGradientWorkers


class TimeMoEGradientEngine(IndependentGradientWorkers, NativeProbe):
    worker_student = "timemoe-200m"
    # BF16 eager attention changes substantially with batch shape. Independent
    # CUDA streams preserve each native window's forward/backward geometry.
    def __init__(self, engine, *, model_key="Maple728/TimeMoE-200M", checkpoint_path=None, **kwargs):
        super().__init__(engine, model_key=model_key, **kwargs)
        self.checkpoint_path = checkpoint_path
        self._gradient_pool = None
        self._pool_size = 0

    def batch_gradient_sketch(self, windows, batch_size=None):
        from rage.execution import execution_policy
        values = np.asarray(windows, dtype=np.float32)
        if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
            raise ValueError("gradient windows must be a non-empty finite matrix")
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be positive")
        size = batch_size or execution_policy(self).batch_size(self, "gradient", fallback=4, count=len(values))
        size = min(size, len(values), 8)
        if self.device.type != "cuda" or size <= 1:
            return super().batch_gradient_sketch(values, batch_size=1)
        if self.checkpoint_path and execution_policy(self).gradient_workers > 1:
            return self._process_gradients(values, size)
        self.model.eval()
        parent = torch.cuda.current_stream(self.device)
        streams = [torch.cuda.Stream(device=self.device) for _ in range(size)]
        for stream in streams:
            stream.wait_stream(parent)

        def compute(item):
            row, window = item
            with torch.cuda.device(self.device), torch.cuda.stream(streams[row % size]):
                loss, gradients = self.raw_gradient(window)
                sketch = self.projector.project(gradients).cpu().numpy()
                return loss, sketch

        losses, sketches = [], []
        # Dispatch in fixed groups: a stream is never shared by active tasks.
        try:
            with ThreadPoolExecutor(max_workers=size, thread_name_prefix="timemoe-gradient") as executor:
                for begin in range(0, len(values), size):
                    results = list(executor.map(compute, enumerate(values[begin:begin + size])))
                    for loss, sketch in results:
                        losses.append(loss)
                        sketches.append(sketch)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            result = self.batch_gradient_sketch(values, batch_size=max(1, size // 2))
            self.last_gradient_execution["oom_retries"] = self.last_gradient_execution.get("oom_retries", 0) + 1
            return result
        self.last_gradient_execution = dict(batch_size=size, backend="independent-cuda-streams",
                                            oom_retries=0)
        return np.asarray(losses), np.asarray(sketches, dtype=np.float32)
