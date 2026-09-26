"""Compact V7 selection files: shared pool, contiguous units, atomic NPZs."""
import json
import os
import tempfile
from pathlib import Path
import numpy as np
from .resampling import ResampledWindows

NATIVE_ZERO_SUPPORT_ARMS = frozenset(('tsds', 'lp-mirror'))

def validate_support(selected, arm):
    """Preserve authorized native zeros without altering probability mass."""
    p = selected.probabilities
    if (not np.isfinite(p).all() or np.any(p < 0) or
            not np.isclose(p.sum(), 1., atol=1e-12, rtol=0)):
        raise ValueError(f'{arm}: invalid selection probability mass')
    coverage = selected.coverage()
    if coverage['filtered_legal_windows']:
        raise ValueError(f'{arm}: selection must own the whole actual pool: {coverage}')
    if arm not in NATIVE_ZERO_SUPPORT_ARMS and coverage['zero_mass_legal_windows']:
        raise ValueError(f'{arm}: selection requires positive support: {coverage}')
    return dict(coverage, support_policy=('native_zero_allowed_no_smoothing'
        if arm in NATIVE_ZERO_SUPPORT_ARMS else 'strictly_positive'))

def atomic_torch(path, value):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as stream:
        temporary=Path(stream.name)
        try:
            torch.save(value,stream)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary,path)

def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        try:
            np.savez_compressed(f, **arrays)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)

def save(path, output, identity):
    s = output.selected
    validate_support(s, output.name)
    is_window = len(s.probabilities) == len(s) and np.all(s.unit_counts == 1)
    starts = s.unit_positions[s.unit_offsets[:-1]]
    if is_window:
        if not np.array_equal(starts, np.arange(len(s))):
            raise ValueError('Window units must follow canonical index order')
        starts = np.empty(0, dtype=np.int64)
        counts = starts
    else:
        counts = s.unit_counts
        # Bounded scan: avoid duplicating a whole multi-million-window index.
        for i, (start, count) in enumerate(zip(starts, counts)):
            if not np.array_equal(s.unit_positions[s.unit_offsets[i]:s.unit_offsets[i+1]],
                                  np.arange(start, start+count)):
                raise ValueError('V7 compact artifact requires contiguous native units')
    metadata = dict(s.metadata)
    metadata.pop('native_scores', None)
    audit = metadata.get('reference_audit')
    if audit:
        metadata['reference_audit'] = {k:v for k,v in audit.items()
            if k not in ('source_scores', 'source_scores_for_softmax')}
    atomic_npz(path, format='v7-compact-selection-v1', identity=json.dumps(identity),
        metadata=json.dumps(metadata), name=output.name, cost=json.dumps(vars(output.cost)),
        diagnostics=json.dumps({k:v for k,v in output.diagnostics.items() if k != 'reference_audit'}),
        unit_mode='window' if is_window else 'contiguous', starts=starts, counts=counts,
        probabilities=s.probabilities, scores=(s.native_scores if s.native_scores is not None
            else np.asarray(s.metadata.get('native_scores', []), dtype=np.float64)),
        descending_order=(np.argsort(-s.native_scores,kind='stable')
            if output.name in ('grad-dot','grad-cos','grand-static') else np.empty(0,dtype=np.int64)))

def load(path, index, manifest, identity):
    from .arms import ArmOutput, SelectionCost
    with np.load(path, allow_pickle=False) as data:
        if str(data['format']) != 'v7-compact-selection-v1' or json.loads(str(data['identity'])) != identity:
            raise ValueError('V7 selection identity differs')
        if str(data['unit_mode']) == 'window':
            offsets = np.arange(len(index)+1)
            positions = np.arange(len(index))
        else:
            counts, starts = data['counts'], data['starts']
            offsets = np.r_[0, np.cumsum(counts)]
            positions = np.empty(int(offsets[-1]), dtype=np.int64)
            for i, (start, count) in enumerate(zip(starts, counts)):
                positions[offsets[i]:offsets[i+1]] = np.arange(start, start+count)
        if len(positions) != len(index) or not np.array_equal(np.sort(positions), np.arange(len(index))):
            raise ValueError('Native units must partition every legal window')
        locations = np.empty(len(index), dtype=np.int64)
        for fid, (lo, hi) in index._bounds.items():
            locations[lo:hi] = index._value_offsets[fid] + index.starts[lo:hi] - 1
        metadata = json.loads(str(data['metadata']))
        if metadata.get('pool_manifest') != manifest:
            raise ValueError('V7 selection pool identity differs')
        s = ResampledWindows(index._values, locations, np.column_stack((index.fragment_ids,index.starts)),
            offsets, positions, data['probabilities'], window_length=index.context_length+index.prediction_length,
            pool_n=manifest['total_equivalents'], metadata=metadata)
        validate_support(s, str(data['name']))
        s.native_scores = data['scores']
        return ArmOutput(str(data['name']), s, [], SelectionCost(**json.loads(str(data['cost']))),
                         json.loads(str(data['diagnostics'])))
