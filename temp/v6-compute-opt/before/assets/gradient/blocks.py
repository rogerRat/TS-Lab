"""CRB resource construction and legacy block serialization."""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
from rags.crb import CRBConfig, CRBEngineMulti, CRBResult, GradientBlock

def save_pool_blocks(path: Path, blocks: list[GradientBlock], counters: dict) -> None:
    directions = np.stack([b.direction for b in blocks]).astype(np.float32)
    frag_ids = np.asarray([b.fragment_id for b in blocks], dtype=np.int64)
    offsets = np.zeros(len(blocks) + 1, dtype=np.int64)
    for i, block in enumerate(blocks):
        offsets[i + 1] = offsets[i] + len(block.start_positions)
    starts = (np.concatenate([b.start_positions for b in blocks]).astype(np.int64)
              if blocks else np.zeros(0, dtype=np.int64))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, directions=directions, frag_ids=frag_ids, offsets=offsets,
                 starts=starts, meta=json.dumps(dict(format="gradient-blocks-npz-v1", **counters)))
    temporary.replace(path)


def load_pool_blocks(path: Path) -> tuple[list[GradientBlock], dict]:
    with np.load(path) as payload:
        directions = payload["directions"]
        frag_ids = payload["frag_ids"]
        offsets = payload["offsets"]
        starts = payload["starts"]
        meta = json.loads(str(payload["meta"]))
    if meta.get("format") not in (None, "gradient-blocks-npz-v1"):
        raise ValueError("Unsupported gradient-block cache format")
    blocks: list[GradientBlock] = []
    for i in range(len(frag_ids)):
        blocks.append(GradientBlock(
            fragment_id=int(frag_ids[i]), left_index=0, right_index=0,
            start_positions=starts[offsets[i] : offsets[i + 1]].astype(np.int64),
            mean_gradient=directions[i], direction=directions[i]))
    return blocks, meta


def compute_pool_blocks(pool, index: WindowIndex, *, config: CRBConfig,
                        gradient_computer, log_every: int = 250,
                        trajectory_batch_size: int = 256,
                        trajectory_assets=None,
                        ) -> tuple[list[GradientBlock], dict]:
    """CRB over every pool trajectory; blocks store window positions.

    Trajectories are driven together (CRBEngineMulti) so the sequential
    error-stage probes of different fragments share GPU batches; per
    trajectory the result is identical to the sequential engine.
    """
    engine = CRBEngineMulti(config, gradient_computer)
    blocks: list[GradientBlock] = []
    counters = {"gradients": 0, "discarded_no_signal": 0,
                "discarded_antagonistic": 0, "banned_budget": 0,
                "trajectories": 0, "split_audits": []}
    started = time.time()
    usable = [f for f in pool
              if len(f.valid_starts(index.context_length,
                                    index.prediction_length, index.stride)) > 0]
    def results_in_batches():
        for begin in range(0, len(usable), trajectory_batch_size):
            group = usable[begin:begin + trajectory_batch_size]
            yield from zip(group, engine.process_fragments(group))

    for number, (fragment, result) in enumerate(results_in_batches()):
        result: CRBResult = result
        lo = index.bounds(fragment.fragment_id)[0]
        converted = [GradientBlock(
            fragment_id=block.fragment_id,
            left_index=block.left_index, right_index=block.right_index,
            start_positions=(lo + (block.start_positions - 1) // index.stride
                             ).astype(np.int64),
            mean_gradient=block.mean_gradient, direction=block.direction)
            for block in result.blocks]
        blocks.extend(converted)
        counters["gradients"] += result.computed_gradients
        counters["discarded_no_signal"] += result.discarded_no_signal
        counters["discarded_antagonistic"] += result.discarded_antagonistic
        counters["banned_budget"] += result.banned_budget
        counters["trajectories"] += 1
        if result.split_audit is not None:
            counters["split_audits"].append(result.split_audit)
        if trajectory_assets is not None:
            from .variants import aggregate_anchors, trajectory_counts
            aggregate = aggregate_anchors(result, expected_count=sum(config.budgets(len(result.valid_starts))))
            if aggregate is not None:
                aggregate.start_positions = (lo + (aggregate.start_positions - 1) // index.stride).astype(np.int64)
                trajectory_assets['trajectories'].append(aggregate)
            trajectory_assets['counts'].append(trajectory_counts(result))
        result.anchors.clear()  # Block directions no longer need the probe vectors.
        if log_every and number % log_every == 0:
            print("  [pool CRB] %d/%d trajectories, %d blocks (%.0fs)" % (
                number + 1, len(pool), len(blocks), time.time() - started),
                flush=True)
    return blocks, counters
