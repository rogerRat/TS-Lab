"""Budget-matched window scores, bounded memory; no gradient computation path."""
import numpy as np
from .ranking import rank_mass, frozen_rank_protocol
from .resampling import ResampledWindows
from .v7_protocol import gradient_method_protocol, NEW_ARMS

def uniform_anchors(size, count):
    if not 1 <= count <= size:
        raise ValueError('Uniform budget must be in [1, legal_windows]')
    anchors = np.unique(np.rint(np.linspace(0, size - 1, count)).astype(np.int64))
    if len(anchors) != count:
        raise ValueError('Rounded anchors must exactly match actual RAGE BP')
    return anchors

def interpolated_chunks(size, positions, gradients, chunk=1024):
    positions = np.asarray(positions, dtype=np.int64)
    gradients = np.asarray(gradients, dtype=np.float64)
    if (chunk < 1 or len(positions) < 1 or gradients.ndim != 2 or
            len(gradients) != len(positions) or not np.isfinite(gradients).all() or
            positions[0] != 0 or np.any(np.diff(positions) <= 0) or
            (len(positions) > 1 and positions[-1] != size-1)):
        raise ValueError('Invalid finite endpoint anchor matrix')
    for begin in range(0, size, chunk):
        x = np.arange(begin, min(begin+chunk, size))
        right = np.clip(np.searchsorted(positions, x, side='right'), 0, len(positions)-1)
        left = np.maximum(right-1, 0)
        width = positions[right] - positions[left]
        w = np.divide(x-positions[left], width, out=np.zeros(len(x)), where=width != 0)
        yield begin, gradients[left] * (1-w[:,None]) + gradients[right] * w[:,None]

def reference_mean(trajectories):
    """Each tuple is (legal count, uniform positions, raw anchor gradients)."""
    total = None
    count = 0
    for size, positions, gradients in trajectories:
        positions = np.asarray(positions, dtype=np.int64)
        gradients = np.asarray(gradients, dtype=np.float64)
        # Validate with one bounded chunk, not a dense W x D expansion.
        next(interpolated_chunks(size, positions, gradients, chunk=1))
        weights = np.zeros(len(positions), dtype=np.float64)
        if len(positions)==1:
            weights[0]=size
        else:
            widths=np.diff(positions)
            # Each interval owns integers [left,right); last endpoint is separate.
            weights[:-1]+=(widths+1)/2
            weights[1:]+=(widths-1)/2
            weights[-1]+=1
        mean = (weights / size) @ gradients
        total = mean if total is None else total + mean
        count += 1
    if count == 0 or not np.isfinite(total).all():
        raise ValueError('Reference requires finite trajectories')
    return total / count

def score_trajectory(size, positions, gradients, target, chunk=65536):
    # target=None is the standalone GraNd path: no reference resource is read.
    target = (np.zeros(np.asarray(gradients).shape[-1],dtype=np.float64) if target is None
              else np.asarray(target, dtype=np.float64))
    if target.ndim != 1 or not np.isfinite(target).all():
        raise ValueError('Nonfinite reference gradient')
    out = {a: np.empty(size, dtype=np.float64) for a in NEW_ARMS}
    target_norm = np.linalg.norm(target)
    positions=np.asarray(positions,dtype=np.int64)
    gradients=np.asarray(gradients,dtype=np.float64)
    next(interpolated_chunks(size,positions,gradients,chunk=1))
    anchor_dot=gradients @ target
    anchor_norm2=np.einsum('ij,ij->i',gradients,gradients)
    adjacent=np.einsum('ij,ij->i',gradients[:-1],gradients[1:])
    for begin in range(0,size,chunk):
        x=np.arange(begin,min(begin+chunk,size))
        right=np.clip(np.searchsorted(positions,x,side='right'),0,len(positions)-1)
        left=np.maximum(right-1,0)
        width=positions[right]-positions[left]
        w=np.divide(x-positions[left],width,out=np.zeros(len(x)),where=width!=0)
        dot=anchor_dot[left]+w*(anchor_dot[right]-anchor_dot[left])
        cross=adjacent[left] if len(positions)>1 else anchor_norm2[left]
        scale=(1-w)**2*anchor_norm2[left]+w**2*anchor_norm2[right]
        norm2=anchor_norm2[left]+2*w*(cross-anchor_norm2[left])+w*w*(
            anchor_norm2[left]+anchor_norm2[right]-2*cross)
        # Near antipodal cancellation, the quadratic loses relative accuracy.
        # Recompute only those rare scalar positions from actual linear vectors.
        unstable=np.flatnonzero((norm2 <= 64*np.finfo(np.float64).eps*scale) & (scale > 0))
        for start in range(0,len(unstable),1024):
            rows=unstable[start:start+1024]
            values=(1-w[rows,None])*gradients[left[rows]]+w[rows,None]*gradients[right[rows]]
            norm2[rows]=np.einsum('ij,ij->i',values,values)
            dot[rows]=values@target
        for edge,which in ((w==0,left),(w==1,right)):
            norm2[edge]=anchor_norm2[which[edge]]
            dot[edge]=anchor_dot[which[edge]]
        norm=np.sqrt(np.maximum(norm2,0))
        cosine = np.divide(dot, norm * target_norm, out=np.zeros(len(x)),
                           where=(norm * target_norm) != 0)
        if not all(np.isfinite(v).all() for v in (dot, norm, cosine)):
            raise ValueError('Nonfinite window score')
        end = begin + len(x)
        out['grad-dot'][begin:end] = dot
        out['grad-cos'][begin:end] = cosine
        out['grand-static'][begin:end] = norm
    return out

def deploy(index, scores, *, arm, manifest, identity):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(index),):
        raise ValueError('One score per legal window required')
    _, probabilities, audit = rank_mass(scores, np.ones(len(scores), dtype=np.int64))
    if np.any(probabilities <= 0):
        raise ValueError('Window rank must have positive support')
    locations = np.empty(len(index), dtype=np.int64)
    for fid, (lo, hi) in index._bounds.items():
        locations[lo:hi] = index._value_offsets[fid] + index.starts[lo:hi] - 1
    selected = ResampledWindows(index._values, locations,
        np.column_stack((index.fragment_ids, index.starts)),
        np.arange(len(index)+1, dtype=np.int64), np.arange(len(index), dtype=np.int64),
        probabilities, window_length=index.context_length+index.prediction_length,
        pool_n=manifest['total_equivalents'], metadata=dict(identity,
            pool_manifest=manifest, method=gradient_method_protocol(arm), scoring_unit='window',
            rank_probability=audit, frozen_rank_protocol=frozen_rank_protocol(),
            source_length_applied_once=True, selection_new_bp=0))
    selected.native_scores = scores
    return selected
