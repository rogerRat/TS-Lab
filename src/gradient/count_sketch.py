"""Fused signed hash accumulation; same Count-Sketch matrix as the torch path."""
import torch


def add_count_sketch(output, values, hashes, signs, *, fused=None):
    if fused is None:
        fused = values.shape[0] >= 4 and values.numel() >= 1048576
    if not fused or values.device.type != "cuda":
        # Keep parameter gradients BF16; only the bounded reduction workspace
        # uses the accumulator dtype required by index_add/Count-Sketch.
        for begin in range(0, values.shape[1], 262144):
            stop = begin + 262144
            signed = (signs[begin:stop] * values[:, begin:stop]).to(output.dtype)
            output.index_add_(1, hashes[begin:stop], signed)
        return
    # Triton is shipped with the CUDA PyTorch environment. Import lazily so
    # CPU-only baseline packages do not need a GPU compiler.
    from gradient.count_sketch_kernel import count_sketch_kernel
    rows, columns = values.shape
    count_sketch_kernel[((rows * columns + 1023) // 1024,)](
        output, values, hashes, signs, rows * columns, columns,
        values.stride(0), values.stride(1), output.shape[1], BLOCK=1024)
