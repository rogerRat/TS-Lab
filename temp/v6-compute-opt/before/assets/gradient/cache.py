"""Run-local, field-validated, append-only BF16 window-gradient resources."""
from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
from pathlib import Path
import re
import tempfile
import warnings
import time

import numpy as np
from .dataset import decode_bf16


def _shard_cache_budget_bytes():
    """Byte bound for decoded shards, sized to the container cgroup, not the host."""
    try:
        limit = int(Path("/sys/fs/cgroup/memory.max").read_text())
    except (OSError, ValueError):
        limit = 110 * 2**30
    if limit <= 0:
        limit = 110 * 2**30
    return min(4 * 2**30, max(256 * 2**20, limit // 16))


def atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        temporary = Path(stream.name)
    temporary.replace(path)


def readable(value):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(value)).strip(".") or "unspecified"


def _materialize(entries):
    """Gather each request once per batch, retaining union order."""
    groups = {}
    for target, (_, (request, row)) in enumerate(entries):
        groups.setdefault(id(request), (request, [], []))[1].append(row)
        groups[id(request)][2].append(target)
    windows = None
    for request, rows, targets in groups.values():
        values = np.asarray(request.windows(np.asarray(rows, dtype=np.int64)))
        if windows is None:
            windows = np.empty((len(entries), *values.shape[1:]), dtype=values.dtype)
        windows[targets] = values
    return windows


def _prefetched_batches(entries, batch):
    """One CPU gather overlaps the current GPU batch; bounded to two batches."""
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="gradient-windows") as executor:
        future = executor.submit(_materialize, entries[:batch])
        for begin in range(0, len(entries), batch):
            values = future.result()
            if begin + batch < len(entries):
                future = executor.submit(_materialize, entries[begin + batch:begin + 2 * batch])
            yield values


@dataclass
class WindowRequest:
    """One method's ordered (fragment_id, physical_start) gradient requirement."""
    dataset: str
    ids: np.ndarray
    windows: object  # callable(row_indices) -> [N, context+horizon]

    def __post_init__(self):
        ids = np.asarray(self.ids)
        if ids.ndim != 2 or ids.shape[1] != 2 or not np.isfinite(ids).all():
            raise ValueError("window IDs must be a finite [N,2] array")
        if not np.equal(ids, ids.astype(np.int64)).all() or np.any(ids[:, 1] < 1):
            raise ValueError("window IDs must be integers with 1-based physical starts")
        self.ids = ids.astype(np.int64, copy=False)


class GradientCache:
    """Union requests, compute missing rows only, then serve immutable resources.

    NPZ shards retain window identity, losses and BF16 gradient bits. Only row
    locations are held in memory; values are loaded a shard at a time.
    """

    def __init__(self, run_dir, engine, *, model_identity, data_fields,
                 context_length, prediction_length):
        self.engine = engine
        self.context_length, self.prediction_length = context_length, prediction_length
        model = dict(model_identity)
        if not model.get("model_id"):
            raise ValueError("gradient assets require model_id")
        model.setdefault("ckpt", model.get("loaded_path") or model["model_id"])
        model.setdefault("ckpt_revision", model.get("revision"))
        config = getattr(engine, "config", None)
        probe = {key: getattr(config, key, None) for key in
                 ("layer", "projection_dimension", "projection_seed", "projection_repeats")}
        self.manifest = json.loads(json.dumps(dict(
            format="window-gradients-bf16-v1", model=model, data=data_fields,
            gradient_protocol="independent-native-loss-bf16-v2",
            context_length=context_length, prediction_length=prediction_length,
            probe=probe, value_kind="projected_per_window_gradient",
            storage_dtype="bfloat16", storage_encoding="uint16-bits",
            index_columns=["fragment_id", "physical_start_1based"])))
        root = Path(run_dir) / "grad_cache"
        root.mkdir(parents=True, exist_ok=True)
        name = readable(model["model_id"]) + "--" + readable(model.get("ckpt_revision"))
        self._lock = (root / (name + ".lock")).open("a+")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock.close()
            raise RuntimeError("another process is preparing this model's gradient cache")
        try:
            version = 0
            while True:
                self.path = root / (name + (f"--fields-{version}" if version else ""))
                marker = self.path / "cache_manifest.json"
                if not self.path.exists():
                    self.path.mkdir()
                    atomic_json(marker, self.manifest)
                    break
                if marker.exists() and json.loads(marker.read_text()) == self.manifest:
                    break
                warnings.warn(f"Gradient cache fields differ or are missing: {self.path}; using a new directory")
                version += 1
            self.rows = {}
            self.shards = sorted(self.path.glob("windows-*.npz"))
            indexed = set()
            index_path = self.path / "window_index.npz"
            if index_path.exists():
                with np.load(index_path, allow_pickle=False) as index:
                    datasets, shard_names = index["datasets"], index["shards"]
                    indexed = set(shard_names.tolist())
                    paths = [self.path / str(name) for name in shard_names]
                    if any(not path.is_file() for path in paths):
                        raise FileNotFoundError("published gradient index refers to a missing shard")
                    for dataset, (fid, start), shard, row in zip(index["dataset_ids"],
                            index["window_ids"], index["shard_ids"], index["shard_rows"]):
                        self.rows[(str(datasets[dataset]), int(fid), int(start))] = (paths[shard], int(row))
            for path in self.shards:
                if path.name in indexed:
                    continue
                with np.load(path, allow_pickle=False) as payload:
                    dataset = str(payload["dataset"])
                    ids = payload["window_ids"]
                    for row, (fid, start) in enumerate(ids):
                        key = (dataset, int(fid), int(start))
                        if key in self.rows:
                            raise ValueError(f"duplicate cached window: {key}")
                        self.rows[key] = (path, row)
            from .timing import timing_identity
            self._timing_identity = timing_identity(self.manifest, engine)
            self._timing_path = self.path / 'gradient_timing.json'
            self._timing_records = []
            if self._timing_path.exists():
                payload = json.loads(self._timing_path.read_text())
                if payload.get('schema') == 'gradient-timing-v1':
                    self._timing_records = payload['records']
            self._timing_record = next((r for r in self._timing_records
                if r['identity'] == self._timing_identity), None)
            if self._timing_record is None:
                self._timing_record = dict(identity=self._timing_identity,
                    gradient_windows=0, gradient_seconds=0., batches=0, batch_sizes={})
                self._timing_records.append(self._timing_record)
            self.computed_gradient_seconds = 0.
            self.computed_windows = 0
            self.frozen = False
            self._ready = True
            self._decoded_shards = OrderedDict()
            self._decoded_shard_bytes = 0
            self._shard_cache_budget = _shard_cache_budget_bytes()
        except BaseException:
            self.close()
            raise

    def gradient_benchmark(self):
        """Cumulative same-model/execution observation, or unavailable mean=None."""
        from .timing import benchmark_summary
        # JSON copy: callers may retain the snapshot while preparation continues.
        return benchmark_summary(json.loads(json.dumps(self._timing_record)))

    def _publish_timing(self):
        if hasattr(self, '_timing_records'):
            atomic_json(self._timing_path, dict(schema='gradient-timing-v1', records=self._timing_records))

    def close(self):
        if not self._lock.closed:
            try:
                if getattr(self, "_ready", False):
                    self._publish_index()
                    self._publish_timing()
            finally:
                fcntl.flock(self._lock, fcntl.LOCK_UN)
                self._lock.close()
                release = getattr(self.engine, "release_gradient_workers", None)
                if release is not None:
                    release()

    def _publish_index(self):
        datasets = list(dict.fromkeys(key[0] for key in self.rows))
        dataset_ids = {name: i for i, name in enumerate(datasets)}
        shard_ids = {path: i for i, path in enumerate(self.shards)}
        with tempfile.NamedTemporaryFile(dir=self.path, delete=False) as stream:
            np.savez(stream, datasets=np.asarray(datasets, dtype=str),
                dataset_ids=np.array([dataset_ids[key[0]] for key in self.rows], dtype=np.int32),
                window_ids=np.array([key[1:] for key in self.rows], dtype=np.int64).reshape(-1, 2),
                shards=np.array([path.name for path in self.shards], dtype=str),
                shard_ids=np.array([shard_ids[path] for path, row in self.rows.values()], dtype=np.int32),
                shard_rows=np.array([row for path, row in self.rows.values()], dtype=np.int32))
            temporary = Path(stream.name)
        temporary.replace(self.path / "window_index.npz")

    def freeze(self):
        self.frozen = True
        self.close()
        self.engine = None
        return self

    def prepare(self, requests):
        """Merge overlapping method requirements before any model computation."""
        pending = {}
        for request in requests:
            ids = np.asarray(request.ids, dtype=np.int64).reshape(-1, 2)
            for row, (fid, start) in enumerate(ids):
                key = (request.dataset, int(fid), int(start))
                if key not in self.rows:
                    pending.setdefault(key, (request, row))
        if not pending:
            return
        if self.frozen or self._lock.closed:
            raise RuntimeError("selection requested an unprepared gradient; add it to assets requirements")
        from rage.execution import execution_policy
        batch = execution_policy(self.engine).batch_size(
            self.engine, "gradient", fallback=32, count=len(pending))
        # Dataset shards keep source and reference identities separate.
        datasets = dict.fromkeys(key[0] for key in pending)
        for dataset in datasets:
            entries = [(key, value) for key, value in pending.items() if key[0] == dataset]
            for begin in range(0, len(entries), 1024):
                chunk = entries[begin:begin + 1024]
                loss_parts, gradient_parts = [], []
                for windows in _prefetched_batches(chunk, batch):
                    gradient_started = time.perf_counter()
                    losses, gradients = self.engine.batch_gradient_sketch(windows)
                    # Engines return host NumPy arrays, so device work is complete.
                    losses, gradients = np.asarray(losses), np.asarray(gradients)
                    gradient_seconds = time.perf_counter() - gradient_started
                    self.computed_gradient_seconds += gradient_seconds
                    record = self._timing_record
                    record['gradient_windows'] += len(windows)
                    record['gradient_seconds'] += gradient_seconds
                    record['batches'] += 1
                    size = str(len(windows))
                    record['batch_sizes'][size] = record['batch_sizes'].get(size, 0) + 1
                    loss_parts.append(np.asarray(losses))
                    gradient_parts.append(np.asarray(gradients))
                gradients = np.concatenate(gradient_parts, axis=-2)
                losses = np.concatenate(loss_parts, axis=-1)
                if gradients.ndim not in (2, 3) or gradients.shape[-2] != len(chunk):
                    raise ValueError("gradient result must be [N,D] or [models,N,D]")
                if not np.isfinite(gradients).all() or not np.isfinite(losses).all():
                    raise ValueError("non-finite gradient resource")
                raw = np.asarray(gradients, dtype=np.float32).view(np.uint32)
                bits = ((raw + np.uint32(0x7fff) + ((raw >> 16) & 1)) >> 16).astype(np.uint16)
                if not np.isfinite(decode_bf16(bits)).all():
                    raise ValueError("gradient exceeds finite BF16 range")
                ids = np.asarray([key[1:] for key, _ in chunk], dtype=np.int64)
                next_shard = max((int(path.stem.split("-")[-1]) for path in self.shards), default=-1) + 1
                path = self.path / f"windows-{next_shard:08d}.npz"
                with tempfile.NamedTemporaryFile(dir=self.path, delete=False) as stream:
                    np.savez(stream, dataset=dataset, window_ids=ids,
                             gradients_bf16=bits, losses=losses)
                    temporary = Path(stream.name)
                temporary.replace(path)
                self.shards.append(path)
                for row, (key, _) in enumerate(chunk):
                    self.rows[key] = (path, row)
                self.computed_windows += len(chunk)
                self._publish_timing()

    def _decoded(self, path):
        """Byte-bounded LRU of decoded shards; one load per shard per round."""
        values = self._decoded_shards.get(path)
        if values is None:
            with np.load(path, allow_pickle=False) as payload:
                values = decode_bf16(payload["gradients_bf16"])
            self._decoded_shards[path] = values
            self._decoded_shard_bytes += values.nbytes
            while (self._decoded_shard_bytes > self._shard_cache_budget
                   and len(self._decoded_shards) > 1):
                _, evicted = self._decoded_shards.popitem(last=False)
                self._decoded_shard_bytes -= evicted.nbytes
        else:
            self._decoded_shards.move_to_end(path)
        return values

    def get_many(self, requests):
        """Serve several requests together: each shard is read at most once."""
        if not requests:
            return []
        self.prepare(requests)
        grouped = {}
        for index, request in enumerate(requests):
            for target, (fid, start) in enumerate(request.ids):
                path, row = self.rows[(request.dataset, int(fid), int(start))]
                grouped.setdefault(path, []).append((index, target, row))
        results = [None] * len(requests)
        for path, locations in grouped.items():
            values = self._decoded(path)
            for index, target, row in locations:
                if results[index] is None:
                    results[index] = np.empty((*values.shape[:-2], len(requests[index].ids),
                                               values.shape[-1]), dtype=np.float32)
                results[index][..., target, :] = values[..., row, :]
        if any(result is None for result in results):
            raise ValueError("gradient request must contain windows")
        return results

    def get(self, request):
        return self.get_many([request])[0]


def index_request(dataset, index, positions):
    positions = np.asarray(positions, dtype=np.int64)
    return WindowRequest(dataset, np.stack(index.fragment_arrays(positions), axis=1),
                         lambda rows: index.windows_at(positions[rows]))


class FragmentGradients:
    """Preserve physical identities when CRB merges requests across fragments."""
    def __init__(self, cache, dataset):
        self.cache, self.dataset = cache, dataset

    def __call__(self, fragment, starts):
        return self.compute_many([(fragment, starts)])

    def compute_many(self, entries):
        requests = []
        for fragment, starts in entries:
            starts = np.asarray(starts, dtype=np.int64)
            def windows(rows, fragment=fragment, starts=starts):
                return np.stack([fragment.extract_window(int(starts[row]),
                    self.cache.context_length, self.cache.prediction_length) for row in rows])
            requests.append(WindowRequest(self.dataset,
                np.column_stack([np.full(len(starts), fragment.fragment_id), starts]), windows))
        return np.concatenate(self.cache.get_many(requests))
