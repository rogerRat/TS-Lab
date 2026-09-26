"""Static full-pool distributions; native scoring units, lazy window draws.

Only the source marginal is stored. Reference rows are normalized and audited
while constructing it, never read or rescored by the training loop.
"""
from __future__ import annotations

import json
import numpy as np

REVISION = "full-pool-rank-volume-v5"
SAMPLER_REVISION = "positive-support-cdf-alias-v1"


def alias_table(probabilities):
    """Alternative discrete sampling table without subtracting adjacent CDFs.

    Used when cumulative rounding would erase positive support. This changes
    only the representation of the same probabilities, never the scoring rule.
    """
    size = len(probabilities)
    scaled = np.asarray(probabilities, dtype=np.float64) * size
    acceptance, aliases = np.ones(size), np.arange(size, dtype=np.int64)
    small, large = np.flatnonzero(scaled < 1).tolist(), np.flatnonzero(scaled >= 1).tolist()
    while small and large:
        low, high = small.pop(), large.pop()
        acceptance[low], aliases[low] = scaled[low], high
        scaled[high] -= 1 - scaled[low]
        (small if scaled[high] < 1 else large).append(high)
    return acceptance, aliases


def trajectory_distribution(index, trajectory_ids, scores, *, pool_manifest, metadata,
                            native_mass=False, temperature=0.1, mapping="rank", gating=True,
                            source_weighting="legal_window_counts"):
    if source_weighting not in ("legal_window_counts", "equal_trajectory"):
        raise ValueError("Unknown source trajectory weighting")
    units = [np.arange(*index.bounds(int(fid))) for fid in trajectory_ids]
    counts = [len(p) for p in units] if source_weighting == "legal_window_counts" else None
    if native_mass:
        probabilities = np.asarray(scores, dtype=np.float64)
        mapping_name = "native_OT_mass"
    elif mapping == "value":
        probabilities = value_mass(scores, counts,
                                    temperature=temperature, gating=gating)
        mapping_name = "finite_alignment_value" if not gating else "positive_alignment_value"
    else:
        probabilities = ranked_mass(scores, counts,
                                    gradient_alignment=gating, temperature=temperature)
        mapping_name = ("positive_alignment_rank" if gating else "finite_alignment_rank")
    return ResampledWindows.from_index(index, units, probabilities,
        pool_manifest=pool_manifest, metadata=dict(metadata, scoring_unit="trajectory",
            trajectory_ids=list(map(int, trajectory_ids)), native_scores=np.asarray(scores).tolist(),
            probability_mapping=mapping_name, source_weighting=source_weighting,
            temperature=None if native_mass else temperature))


def window_distribution(index, scores, *, pool_manifest, metadata, temperature=0.1,
                        mapping="rank"):
    # Avoid millions of singleton arrays/Python objects for window-level arms.
    selected = ResampledWindows.from_index(index, [np.arange(len(index))], [1.],
        pool_manifest=pool_manifest, metadata=dict(metadata, scoring_unit="window",
            probability_mapping="value" if mapping == "value" else "rank",
            temperature=temperature))
    selected.unit_offsets = np.arange(len(index) + 1, dtype=np.int64)
    selected.probabilities = (value_mass(scores, temperature=temperature, gating=False)
                               if mapping == "value"
                               else ranked_mass(scores, temperature=temperature))
    selected.native_scores = np.asarray(scores)
    selected._prepare_sampling()
    return selected


def random_distribution(index, *, pool_manifest):
    return ResampledWindows.from_index(index, [np.arange(len(index))], [1.],
        pool_manifest=pool_manifest, metadata=dict(scoring_unit="none", probability_mapping="uniform_window"))


class InfeasibleReference(ValueError):
    def __init__(self, reference_id, share):
        self.reference_id, self.share = reference_id, float(share)
        super().__init__(f"Reference {reference_id!r} has no positive alignment; pi={share:.17g}")


class _DirectionRows:
    """Keep block vectors in their native storage; convert only working batches."""
    def __init__(self, rows):
        self.rows = rows
        if isinstance(rows, np.ndarray):
            if rows.ndim != 2:
                raise ValueError("CRB directions must be matrices")
            self.shape = rows.shape
        else:
            if not len(rows) or np.asarray(rows[0]).ndim != 1:
                raise ValueError("CRB directions must be nonempty row vectors")
            self.shape = (len(rows), len(rows[0]))

    def batch(self, begin, end):
        values = np.asarray(self.rows[begin:end], dtype=np.float64)
        if values.shape != (end-begin, self.shape[1]) or not np.isfinite(values).all():
            raise ValueError("CRB directions must be matching finite row vectors")
        return values


def _gpu_reference_marginal(refs, gpu_sources, pi, m, reference_ids, temperature,
                            row_batch, gating):
    """Keep score-sized softmax work on device; copy only row audits and beta."""
    import torch
    target = gpu_sources.device
    counts = torch.as_tensor(m, device=target)
    log_counts = counts.log()
    beta = torch.zeros(len(m), dtype=torch.float64, device=target)
    audits = []
    for begin in range(0, refs.shape[0], row_batch):
        end = min(begin + row_batch, refs.shape[0])
        scores = torch.as_tensor(refs.batch(begin, end), device=target) @ gpu_sources.T
        finite = torch.isfinite(scores).all(dim=1)
        positive = scores > 0
        positive_counts = positive.sum(dim=1)
        support = positive if gating else torch.ones_like(positive)
        supported = scores.masked_fill(~support, -torch.inf)
        logits = (supported - supported.max(dim=1, keepdim=True).values) / temperature
        logits.add_(log_counts)
        logits.sub_(logits.max(dim=1, keepdim=True).values)
        weights = logits.exp()
        conditional = weights / weights.sum(dim=1, keepdim=True)
        joint = conditional * torch.as_tensor(pi[begin:end], device=target)[:, None]
        underflow = ((joint == 0) & support).any(dim=1)
        nonpositive_mass = joint.masked_fill(positive, 0).sum(dim=1)
        # One small host transfer validates the batch before modifying beta.
        stats = torch.stack((finite, positive_counts, underflow, nonpositive_mass,
                             conditional.sum(dim=1), joint.sum(dim=1)), dim=1).cpu().numpy()
        for offset, stat in enumerate(stats):
            j = begin + offset
            identity = j if reference_ids is None else reference_ids[j]
            if not stat[0]:
                raise ValueError(f"Nonfinite CRB alignment in reference row {j}")
            if gating and stat[1] == 0:
                raise InfeasibleReference(identity, pi[j])
            if stat[2]:
                raise ValueError("Temperature underflows alignment mass; increase temperature")
            audits.append(dict(reference_id=identity, pi=float(pi[j]),
                positive_pairs=int(stat[1]), nonpositive_pairs=len(m)-int(stat[1]),
                nonpositive_mass=0.0 if gating else float(stat[3]),
                conditional_mass=float(stat[4]), joint_mass=float(stat[5])))
        beta.add_(joint.sum(dim=0))
    return beta.cpu().numpy(), audits


def rags_marginal(reference_directions, source_directions, reference_counts,
                  source_counts, *, reference_ids=None, temperature=0.1, device="auto", row_batch=128,
                  gating=True, source_batch=4096):
    """Batch cosine products, preserving each reference's independent joint mass.

    Inputs are the original CRB unit directions, not recomputed window scores.
    Score workspace is bounded by row_batch * source blocks. Large products
    default to GPU; CPU-only readers retain a batched NumPy path.
    With ``gating=False`` (no-hard-gating ablation) every finite cosine,
    including negative alignment, participates in each reference row's softmax.
    """
    refs = _DirectionRows(reference_directions)
    sources = _DirectionRows(source_directions)
    n = np.asarray(reference_counts, dtype=np.float64)
    m = np.asarray(source_counts, dtype=np.float64)
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    if row_batch < 1 or source_batch < 1:
        raise ValueError("row_batch and source_batch must be positive")
    if (refs.shape[1] != sources.shape[1]
            or len(n) != refs.shape[0] or len(m) != sources.shape[0]
            or not len(n) or not len(m) or np.any(n <= 0) or np.any(m <= 0)):
        raise ValueError("CRB directions require matching nonempty legal-start counts")
    pi = n / n.sum()
    beta = np.zeros(len(m), dtype=np.float64)
    audits = []
    gpu_sources = None
    source_bytes = sources.shape[0] * sources.shape[1] * 8
    if device != "cpu" and (device != "auto" or refs.shape[0]*refs.shape[1]*sources.shape[0] >= 1_000_000):
        try:
            import torch
        except ImportError:
            torch = None
        if torch is not None and torch.cuda.is_available():
            target = "cuda" if device == "auto" else device
            # Gradient preparation can leave reclaimable allocator reservations.
            # Do not mistake those idle pages for a permanently CPU-only task.
            torch.cuda.empty_cache()
            free, _ = torch.cuda.mem_get_info(target)
            # Bound all on-device softmax workspaces as well as score storage.
            if source_bytes + 8 * row_batch * sources.shape[0] * 8 < free * .3:
                try:
                    gpu_sources = torch.empty(sources.shape, dtype=torch.float64, device=target)
                    for begin in range(0, sources.shape[0], source_batch):
                        end = min(begin + source_batch, sources.shape[0])
                        gpu_sources[begin:end].copy_(torch.as_tensor(sources.batch(begin, end), device=target))
                except torch.cuda.OutOfMemoryError:
                    gpu_sources = None
                    torch.cuda.empty_cache()

    if gpu_sources is not None:
        print(f"RAGS_MATCHING_BACKEND cuda refs={refs.shape[0]} sources={sources.shape[0]} "
              f"dimension={sources.shape[1]} row_batch={row_batch}", flush=True)
        try:
            beta, audits = _gpu_reference_marginal(refs, gpu_sources, pi, m,
                reference_ids, temperature, row_batch, gating)
            return beta, dict(pi=pi.tolist(), rows=audits, temperature=temperature,
                mapping=("exp(cosine/temperature), pairwise cosine>0" if gating else
                         "exp(cosine/temperature), all finite cosines (no hard gating)"),
                marginal_mass=float(beta.sum()))
        except torch.cuda.OutOfMemoryError:
            # Retry the complete marginal on CPU; no partial beta is published.
            gpu_sources = None
            torch.cuda.empty_cache()

    print(f"RAGS_MATCHING_BACKEND cpu refs={refs.shape[0]} sources={sources.shape[0]} "
          f"dimension={sources.shape[1]} row_batch={row_batch}", flush=True)

    def cpu_products(rows):
        products = np.empty((len(rows), sources.shape[0]), dtype=np.float64)
        for begin in range(0, sources.shape[0], source_batch):
            end = min(begin + source_batch, sources.shape[0])
            products[:, begin:end] = rows @ sources.batch(begin, end).T
        return products

    for j in range(refs.shape[0]):
        if j % row_batch == 0:
            rows = refs.batch(j, min(j + row_batch, refs.shape[0]))
            if gpu_sources is None:
                products = cpu_products(rows)
            else:
                try:
                    products = (torch.as_tensor(rows, device=gpu_sources.device) @ gpu_sources.T).cpu().numpy()
                except torch.cuda.OutOfMemoryError:
                    gpu_sources = None
                    torch.cuda.empty_cache()
                    products = cpu_products(rows)
        scores = products[j % row_batch]
        if not np.all(np.isfinite(scores)):
            raise ValueError(f"Nonfinite CRB alignment in reference row {j}")
        positive = scores > 0
        identity = j if reference_ids is None else reference_ids[j]
        if gating:
            if not positive.any():
                raise InfeasibleReference(identity, pi[j])
            support = positive
        else:
            support = np.ones(len(scores), dtype=bool)
        # Center before division: prevents overflow even for a tiny temperature.
        logits = (scores[support] - scores[support].max()) / temperature
        logits += np.log(m[support])
        weights = np.exp(logits - logits.max())
        conditional = weights / weights.sum()
        joint = pi[j] * conditional
        if np.any(joint == 0):
            raise ValueError("Temperature underflows alignment mass; increase temperature")
        beta[support] += joint
        row_nonpositive_mass = float(joint[scores[support] <= 0].sum()) if not gating else 0.0
        audits.append(dict(reference_id=identity, pi=float(pi[j]),
                           positive_pairs=int(positive.sum()),
                           nonpositive_pairs=int((~positive).sum()),
                           nonpositive_mass=row_nonpositive_mass,
                           conditional_mass=float(conditional.sum()),
                           joint_mass=float(joint.sum())))
    # Do not renormalize across reference rows: their separate budgets are final.
    return beta, dict(pi=pi.tolist(), rows=audits, temperature=temperature,
                      mapping=("exp(cosine/temperature), pairwise cosine>0" if gating else
                               "exp(cosine/temperature), all finite cosines (no hard gating)"),
                      marginal_mass=float(beta.sum()))


def rags_global_marginal(reference_directions, source_directions, reference_counts,
                         source_counts, *, reference_ids=None, temperature=0.1,
                         gating=True, direction_batch=4096):
    """Aggregate all reference alignment into one global source competition."""
    refs = _DirectionRows(reference_directions)
    sources = _DirectionRows(source_directions)
    n = np.asarray(reference_counts, dtype=np.float64)
    m = np.asarray(source_counts, dtype=np.float64)
    if (refs.shape[1] != sources.shape[1]
            or len(n) != refs.shape[0] or len(m) != sources.shape[0]
            or not len(n) or not len(m) or np.any(n <= 0) or np.any(m <= 0)):
        raise ValueError("CRB directions require matching nonempty legal-start counts")
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    if direction_batch < 1:
        raise ValueError("direction_batch must be positive")
    weights = n / n.sum()
    # Linearity avoids materializing the reference-by-source product matrix.
    # Keep the aggregate's norm: normalizing it would change the cosine sum.
    aggregate = np.zeros(refs.shape[1], dtype=np.float64)
    for begin in range(0, refs.shape[0], direction_batch):
        end = min(begin + direction_batch, refs.shape[0])
        aggregate += weights[begin:end] @ refs.batch(begin, end)
    scores = np.empty(sources.shape[0], dtype=np.float64)
    for begin in range(0, sources.shape[0], direction_batch):
        end = min(begin + direction_batch, sources.shape[0])
        scores[begin:end] = sources.batch(begin, end) @ aggregate
    beta = value_mass(scores, m, temperature=temperature, gating=gating)
    audit = dict(reference_ids=None if reference_ids is None else list(reference_ids),
        reference_weights=weights.tolist(), source_scores=scores.tolist(),
        aggregation='reference_window_weighted_mean_cosine',
        global_nonpositive_scores=int((scores <= 0).sum()),
        mapping=("exp(weighted_mean_cosine/temperature), global positive gating" if gating
                 else "exp(weighted_mean_cosine/temperature), all finite cosines"),
        marginal_mass=float(beta.sum()))
    return beta, audit


def rags_benchmark_marginal(reference_directions, source_directions, reference_weights,
                            *, reference_ids=None, temperature=0.1, direction_batch=4096,
                            source_weighting="equal_block"):
    """Average reference trajectories with optional internal block fractions.

    Each reference trajectory has total weight 1/N; all weights sum to ONE.
    Never multiply source-unit mass by its legal-window count.
    """
    refs, sources = _DirectionRows(reference_directions), _DirectionRows(source_directions)
    weights = np.asarray(reference_weights, dtype=np.float64)
    if (refs.shape[1] != sources.shape[1] or weights.shape != (refs.shape[0],)
            or not len(weights) or not sources.shape[0] or not np.isfinite(weights).all()
            or np.any(weights <= 0) or direction_batch < 1):
        raise ValueError("Invalid benchmark reference block weights/directions")
    if not np.isclose(weights.sum(), 1., rtol=0., atol=1e-12):
        raise ValueError("Benchmark reference weights must sum to one")
    if source_weighting not in ("equal_block", "equal_trajectory"):
        raise ValueError("Unknown benchmark source-unit weighting")
    aggregate = np.zeros(refs.shape[1], dtype=np.float64)
    for begin in range(0, refs.shape[0], direction_batch):
        end = min(begin + direction_batch, refs.shape[0])
        aggregate += weights[begin:end] @ refs.batch(begin, end)
    scores = np.empty(sources.shape[0], dtype=np.float64)
    for begin in range(0, sources.shape[0], direction_batch):
        end = min(begin + direction_batch, sources.shape[0])
        scores[begin:end] = sources.batch(begin, end) @ aggregate
    beta = value_mass(scores, temperature=temperature, gating=False)
    return beta, dict(reference_ids=reference_ids, reference_weights=weights.tolist(),
        reference_weight_sum=float(weights.sum()), source_scores=scores.tolist(),
        aggregation="mean_reference_trajectory_cosines",
        source_weighting=source_weighting, temperature=temperature,
        mapping="exp(mean_reference_trajectory_cosines/temperature), all finite scores",
        marginal_mass=float(beta.sum()))


def value_mass(scores, counts=None, *, temperature=0.1, gating=True):
    """Raw-score softmax mass: exp(s/T) with legal-window counts.

    RAGS-style value mapping applied to any native score vector: the numeric
    gaps of the scores drive concentration. With ``gating=True`` only strictly
    positive scores receive mass; ``gating=False`` keeps every finite score.
    Mirrors ranked_mass's calling convention.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if not np.all(np.isfinite(scores)) or not len(scores):
        raise ValueError("Native scores must be nonempty and finite")
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    keep = scores > 0 if gating else np.ones(len(scores), dtype=bool)
    if not keep.any():
        raise ValueError("No positive alignment for value mass")
    logits = (scores[keep] - scores[keep].max()) / temperature
    if counts is not None:
        counts = np.asarray(counts, dtype=np.float64)
        if counts.shape != scores.shape or np.any(counts <= 0):
            raise ValueError("Native units require positive legal-window counts")
        logits += np.log(counts[keep])
    weights = np.exp(logits - logits.max())
    mass = np.zeros(len(scores), dtype=np.float64)
    mass[keep] = weights / weights.sum()
    # Units far below the max legitimately receive exactly zero mass (the
    # mathematical limit of the softmax); sampling is unaffected.  Only a
    # fully-degenerate distribution would be an error, which max-subtraction
    # makes impossible.  (2026-09-17: relaxed after chronos/azure/LESS crash.)
    return mass


def ranked_mass(scores, counts=None, *, gradient_alignment=False, temperature=0.1):
    """Tie-preserving ascending empirical ranks -> exp(rank/temperature).

    All native scores participate in ranking; only gradient alignment methods
    apply sign gating. Counts convert per-window weight to trajectory mass.
    Native OT trajectory mass must NOT pass through this function.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if not np.all(np.isfinite(scores)) or not len(scores):
        raise ValueError("Native scores must be nonempty and finite")
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    _, inverse, frequency = np.unique(scores, return_inverse=True, return_counts=True)
    midrank = (np.cumsum(frequency) - (frequency + 1) / 2)[inverse]
    rank = midrank / max(1, len(scores) - 1)
    keep = scores > 0 if gradient_alignment else np.ones(len(scores), dtype=bool)
    if not keep.any():
        raise ValueError("No positive trajectory alignment")
    logits = (rank[keep] - rank[keep].max()) / temperature
    if counts is not None:
        counts = np.asarray(counts, dtype=np.float64)
        if counts.shape != scores.shape or np.any(counts <= 0):
            raise ValueError("Native units require positive legal-window counts")
        logits += np.log(counts[keep])
    weights = np.exp(logits - logits.max())
    mass = np.zeros(len(scores), dtype=np.float64)
    mass[keep] = weights / weights.sum()
    if np.any(mass[keep] == 0):
        raise ValueError("Temperature underflows positive rank mass")
    return mass


class ResampledWindows:
    """One packed pool, an index of legal starts, and disjoint native units.

    __getitem__ uses global legal-window row IDs. sample_indices draws units
    with their final mass, then uniformly chooses an owned row. CRB envelope
    overlaps do not duplicate start identities. No dense overlapping windows.
    """
    def __init__(self, values, window_locations, window_ids, unit_offsets,
                 unit_positions, probabilities, *, window_length, pool_n, metadata):
        self.values = np.asarray(values, dtype=np.float32)
        self.window_locations = np.asarray(window_locations, dtype=np.int64)
        self.window_ids = np.asarray(window_ids, dtype=np.int64)
        self.unit_offsets = np.asarray(unit_offsets, dtype=np.int64)
        self.unit_positions = np.asarray(unit_positions, dtype=np.int64)
        self.probabilities = np.asarray(probabilities, dtype=np.float64)
        self.window_length = int(window_length)
        self.pool_n = self.budget_windows = int(pool_n)
        self.metadata = dict(metadata, resampling_revision=REVISION)
        self.native_scores = None
        counts = np.diff(self.unit_offsets)
        if (len(counts) != len(self.probabilities) or np.any(counts <= 0)
                or not np.all(np.isfinite(self.probabilities))
                or np.any(self.probabilities < 0)
                or not np.isclose(self.probabilities.sum(), 1., atol=1e-12, rtol=0)):
            raise ValueError("Invalid native-unit probability distribution")
        self._prepare_sampling()

    def _prepare_sampling(self):
        self.unit_counts = np.diff(self.unit_offsets)
        self.cumulative_mass = np.cumsum(self.probabilities)
        last_positive = len(self.probabilities) - 1 - np.argmax(self.probabilities[::-1] > 0)
        self.cumulative_mass[last_positive:] = 1.
        increments = np.diff(np.r_[0., self.cumulative_mass])
        self.alias = (alias_table(self.probabilities) if np.any(
            (self.probabilities > 0) & (increments <= 0)) else None)
        self.metadata["sampler_revision"] = SAMPLER_REVISION
        self.metadata["sampler_table"] = "alias" if self.alias is not None else "cdf"

    def sampler_probabilities(self):
        """Deterministic distribution represented by the actual sampling table."""
        if self.alias is None:
            return np.diff(np.r_[0., self.cumulative_mass])
        acceptance, aliases = self.alias
        result = acceptance.copy()
        np.add.at(result, aliases, 1 - acceptance)
        return result / len(result)

    @classmethod
    def from_index(cls, index, positions_by_unit, probabilities, *, pool_manifest, metadata):
        width = index.context_length + index.prediction_length
        actual_n = sum(f.length // width for f in index.by_fragment.values())
        if actual_n != pool_manifest["total_equivalents"]:
            raise ValueError("Pool manifest N differs from complete trajectory equivalents")
        if index.stride != 1:
            raise ValueError("Full-pool resampling requires all legal starts (stride=1)")
        positions = np.concatenate(positions_by_unit).astype(np.int64)
        if len(np.unique(positions)) != len(positions):
            raise ValueError("Native units must own unique legal starts, not duplicate envelopes")
        locations = np.empty(len(index), dtype=np.int64)
        for fid, (lo, hi) in index._bounds.items():
            locations[lo:hi] = index._value_offsets[fid] + index.starts[lo:hi] - 1
        return cls(index._values, locations, np.column_stack((index.fragment_ids, index.starts)),
                   np.cumsum([0] + [len(p) for p in positions_by_unit]), positions, probabilities,
                   window_length=width, pool_n=actual_n,
                   metadata=dict(metadata, pool_manifest=pool_manifest))

    def __len__(self):
        return len(self.window_locations)

    def __getitem__(self, rows):
        return self.values[self.window_locations[rows][..., None] + np.arange(self.window_length)]

    def sample_indices(self, rng, batch_size):
        # Build CDF once, not an O(all units) rng.choice normalization each step.
        if self.alias is None:
            units = np.searchsorted(self.cumulative_mass, rng.random(batch_size), side="right")
        else:
            acceptance, aliases = self.alias
            columns = rng.integers(0, len(acceptance), size=batch_size)
            units = np.where(rng.random(batch_size) < acceptance[columns], columns, aliases[columns])
        local = rng.integers(0, self.unit_counts[units])
        return self.unit_positions[self.unit_offsets[units] + local]

    def window_probabilities(self):
        """Audit only: expand probability scalars, never the actual windows."""
        q = np.zeros(len(self), dtype=np.float64)
        counts = np.diff(self.unit_offsets)
        q[self.unit_positions] = np.repeat(self.probabilities / counts, counts)
        return q

    def coverage(self):
        counts = np.diff(self.unit_offsets)
        positive = self.probabilities > 0
        return dict(pool_equivalents=self.pool_n, pool_legal_windows=len(self),
                    native_units=len(counts), positive_units=int(positive.sum()),
                    supported_legal_windows=int(counts[positive].sum()),
                    filtered_legal_windows=int(len(self) - counts.sum()),
                    zero_mass_legal_windows=int(counts[~positive].sum()))

    def payload(self):
        result = dict(values=self.values, window_locations=self.window_locations,
                    window_ids=self.window_ids, unit_offsets=self.unit_offsets,
                    unit_positions=self.unit_positions, probabilities=self.probabilities,
                    window_length=self.window_length, pool_n=self.pool_n,
                    resampling_metadata=json.dumps(self.metadata))
        if self.native_scores is not None:
            result["native_scores"] = self.native_scores
        return result

    @classmethod
    def from_payload(cls, payload):
        metadata = json.loads(str(payload["resampling_metadata"]))
        if metadata["resampling_revision"] != REVISION:
            raise ValueError("Resampling protocol changed; create a new run")
        selected = cls(**{key: payload[key] for key in
                      ("values", "window_locations", "window_ids", "unit_offsets",
                       "unit_positions", "probabilities", "window_length", "pool_n")},
                   metadata=metadata)
        if "native_scores" in payload:
            selected.native_scores = payload["native_scores"]
        return selected
