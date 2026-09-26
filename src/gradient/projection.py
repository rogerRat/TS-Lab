"""Bounded-memory random projections for per-example gradients.

TensorSketchProjector is the fast path for a linear output head: it sketches
the two factors of the weight-gradient outer product and never materializes
that outer product. ParameterHashProjector is the fallback for an arbitrary
parameter layer and accepts one parameter gradient at a time.
"""

from __future__ import annotations

import math

import numpy as np
import torch


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("%s must be a positive integer" % name)
    return value


class TensorSketchProjector:
    """Project a linear-layer gradient without making its outer product.

    token_chunk_size is the important OOM guard. The FFT workspace is bounded
    by batch_size * token_chunk_size * projection_dimension rather than by the
    complete decoder sequence.
    """

    def __init__(
        self,
        vocabulary_size: int,
        hidden_size: int,
        projection_dimension: int = 4096,
        repeats: int = 1,
        seed: int = 42,
        device: torch.device | str = "cpu",
        token_chunk_size: int = 8,
    ) -> None:
        self.vocabulary_size = _positive_int(vocabulary_size, "vocabulary_size")
        self.hidden_size = _positive_int(hidden_size, "hidden_size")
        self.dimension = _positive_int(projection_dimension, "projection_dimension")
        self.repeats = _positive_int(repeats, "repeats")
        self.seed = int(seed)
        self.device = torch.device(device)
        self.token_chunk_size = _positive_int(token_chunk_size, "token_chunk_size")
        self._weight_hashes: list[torch.Tensor] = []
        self._weight_signs: list[torch.Tensor] = []
        self._hidden_hashes: list[torch.Tensor] = []
        self._hidden_signs: list[torch.Tensor] = []
        for repeat in range(self.repeats):
            rng = np.random.default_rng(self.seed + 1009 * (repeat + 1))
            self._weight_hashes.append(torch.as_tensor(
                rng.integers(0, self.dimension, size=self.vocabulary_size),
                dtype=torch.long, device=self.device,
            ))
            self._weight_signs.append(torch.as_tensor(
                rng.choice((-1.0, 1.0), size=self.vocabulary_size),
                dtype=torch.float32, device=self.device,
            ))
            self._hidden_hashes.append(torch.as_tensor(
                rng.integers(0, self.dimension, size=self.hidden_size),
                dtype=torch.long, device=self.device,
            ))
            self._hidden_signs.append(torch.as_tensor(
                rng.choice((-1.0, 1.0), size=self.hidden_size),
                dtype=torch.float32, device=self.device,
            ))

    def _sketch_last_dimension(
        self, values: torch.Tensor, hashes: torch.Tensor, signs: torch.Tensor,
    ) -> torch.Tensor:
        result = torch.zeros(
            *values.shape[:-1], self.dimension,
            dtype=torch.float32, device=values.device,
        )
        indices = hashes.reshape((1,) * (values.ndim - 1) + (len(hashes),))
        indices = indices.expand_as(values)
        result.scatter_add_(
            -1, indices, values.float() * signs.reshape((1,) * (values.ndim - 1) + (len(signs),)),
        )
        return result

    def project(self, logits_gradient: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """Return one projected row per batch item."""
        if logits_gradient.ndim != 3 or hidden.ndim != 3:
            raise ValueError("TensorSketch inputs must be 3-D")
        if logits_gradient.shape[:2] != hidden.shape[:2]:
            raise ValueError("TensorSketch batch/token dimensions do not match")
        if logits_gradient.shape[-1] != self.vocabulary_size or hidden.shape[-1] != self.hidden_size:
            raise ValueError("TensorSketch feature dimensions do not match projector")
        batch_size, token_count = logits_gradient.shape[:2]
        result = torch.zeros(batch_size, self.dimension, dtype=torch.float32, device=logits_gradient.device)
        for weight_hash, weight_sign, hidden_hash, hidden_sign in zip(
            self._weight_hashes, self._weight_signs, self._hidden_hashes, self._hidden_signs,
        ):
            repeat_result = torch.zeros_like(result)
            for start in range(0, token_count, self.token_chunk_size):
                stop = min(token_count, start + self.token_chunk_size)
                left = self._sketch_last_dimension(
                    logits_gradient[:, start:stop], weight_hash, weight_sign,
                )
                right = self._sketch_last_dimension(
                    hidden[:, start:stop], hidden_hash, hidden_sign,
                )
                left_frequency = torch.fft.rfft(left, n=self.dimension, dim=-1)
                right_frequency = torch.fft.rfft(right, n=self.dimension, dim=-1)
                repeat_result.add_(torch.fft.irfft(
                    left_frequency * right_frequency,
                    n=self.dimension, dim=-1,
                ).sum(dim=1))
                del left, right, left_frequency, right_frequency
            result.add_(repeat_result)
            del repeat_result
        return result / math.sqrt(self.repeats)


class ParameterHashProjector:
    """Hash one flattened parameter gradient directly into a fixed-width row."""

    def __init__(
        self,
        parameter_size: int,
        projection_dimension: int = 4096,
        seed: int = 42,
        device: torch.device | str = "cpu",
        value_chunk_size: int = 1 << 20,
        hashes=None,
        signs=None,
    ) -> None:
        self.parameter_size = _positive_int(parameter_size, "parameter_size")
        self.dimension = _positive_int(projection_dimension, "projection_dimension")
        self.seed = int(seed)
        self.device = torch.device(device)
        self.value_chunk_size = _positive_int(value_chunk_size, "value_chunk_size")
        rng = np.random.default_rng(seed)
        self.hashes = torch.as_tensor(
            rng.integers(0, self.dimension, size=self.parameter_size) if hashes is None else hashes,
            dtype=torch.int32, device=self.device,
        )
        self.signs = torch.as_tensor(
            rng.choice((-1.0, 1.0), size=self.parameter_size) if signs is None else signs,
            dtype=torch.int8, device=self.device,
        )

    def project(self, gradient: torch.Tensor) -> torch.Tensor:
        flattened = gradient.detach().reshape(-1).float()
        if flattened.numel() != self.parameter_size:
            raise ValueError("parameter gradient size does not match projector")
        result = torch.zeros(self.dimension, dtype=torch.float32, device=flattened.device)
        for start in range(0, self.parameter_size, self.value_chunk_size):
            stop = min(self.parameter_size, start + self.value_chunk_size)
            result.index_add_(
                0,
                self.hashes[start:stop],
                flattened[start:stop] * self.signs[start:stop],
            )
        return result


__all__ = ["TensorSketchProjector", "ParameterHashProjector"]
