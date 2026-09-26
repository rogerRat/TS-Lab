"""Model-free, NumPy-only reader for indexed gradient datasets."""
import json
from pathlib import Path
import numpy as np


def decode_bf16(bits):
    """Expand stored BF16 bits to NumPy's FP32 interchange representation."""
    return (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32)


class GradientDataset:
    """Join preprocessed windows to cached gradients by dataset/fragment/start.

    The cache directory is self-contained. Reading never imports a model,
    requires Torch, opens a GPU or computes a missing gradient.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.manifest = json.loads((self.path / "cache_manifest.json").read_text())
        if self.manifest.get("format") != "window-gradients-bf16-v1":
            raise ValueError("unsupported gradient dataset format")
        with np.load(self.path / "window_index.npz", allow_pickle=False) as index:
            self.datasets = index["datasets"]
            self.dataset_ids = index["dataset_ids"]
            self.window_ids = index["window_ids"]
            self.shards = index["shards"]
            self.shard_ids = index["shard_ids"]
            self.shard_rows = index["shard_rows"]
        self._positions = {(str(self.datasets[dataset]), int(fid), int(start)): row
            for row, (dataset, (fid, start)) in enumerate(zip(self.dataset_ids, self.window_ids))}

    def __len__(self):
        return len(self.window_ids)

    def rows_at(self, positions):
        """Return aligned IDs, losses and gradients in the requested row order."""
        positions = np.asarray(positions, dtype=np.int64).reshape(-1)
        if not len(positions):
            raise ValueError("request at least one gradient row")
        values = losses = None
        for shard_id in np.unique(self.shard_ids[positions]):
            targets = np.flatnonzero(self.shard_ids[positions] == shard_id)
            rows = self.shard_rows[positions[targets]]
            with np.load(self.path / str(self.shards[shard_id]), allow_pickle=False) as shard:
                gradients = decode_bf16(shard["gradients_bf16"])
                shard_losses = shard["losses"]
            if values is None:
                values = np.empty((*gradients.shape[:-2], len(positions), gradients.shape[-1]), dtype=np.float32)
                losses = np.empty((*shard_losses.shape[:-1], len(positions)), dtype=shard_losses.dtype)
            values[..., targets, :] = gradients[..., rows, :]
            losses[..., targets] = shard_losses[..., rows]
        return dict(window_ids=self.window_ids[positions],
                    datasets=self.datasets[self.dataset_ids[positions]], gradients=values, losses=losses)

    def lookup(self, dataset, window_ids):
        positions = [self._positions[(dataset, int(fid), int(start))] for fid, start in window_ids]
        return self.rows_at(positions)["gradients"]

    def for_index(self, index, positions, *, dataset="source"):
        """Use the same WindowIndex positions as index.windows_at(positions)."""
        ids = np.stack(index.fragment_arrays(np.asarray(positions, dtype=np.int64)), axis=1)
        return self.lookup(dataset, ids)

    def __getitem__(self, positions):
        if isinstance(positions, slice):
            positions = np.arange(*positions.indices(len(self)))
        return self.rows_at(positions)
