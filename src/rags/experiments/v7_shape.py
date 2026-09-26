"""Zero-BP CRB block-shape matching from cached endpoint gradients.

Each native block is a line on u in [-1/2, 1/2].  The midpoint and slope
are normalized by the line's integrated squared energy before matching.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from assets.gradient.cache import atomic_json, index_request
from .reference_opinions import AGGREGATIONS, reference_scores_many

VERSION = 'v7-rags-block-shape-integral-v1'
PRIORITY = ('expSum', 'linearSum', 'zeroClipSum', 'maxSum')
ARMS = tuple(f'rags-shape-{mode}' for mode in PRIORITY)
INV_SQRT12 = 1.0 / np.sqrt(12.0)


def protocol():
    return dict(version=VERSION, object='retained_CRB_block',
        curve='g(u)=midpoint+u*endpoint_difference; u_in_[-0.5,0.5]',
        block_length='each_block_rescaled_to_unit_interval_for_shape_only',
        endpoint_values='original_4096D_projected_gradients_from_frozen_V7_cache',
        energy='||midpoint||^2+||endpoint_difference||^2/12',
        shape='[midpoint,endpoint_difference/sqrt(12)]/sqrt(energy); zero_energy_is_zero',
        integrated_squared_distance='||shape_source-shape_reference||^2',
        similarity='normalized_shape_inner_product; equals_1-distance_squared/2_for_two_nonzero_shapes',
        aggregations=list(PRIORITY),
        transforms=dict(linearSum='alpha*similarity',
            zeroClipSum='alpha*max(similarity,0)',
            expSum='alpha*exp(similarity); inner_temperature=1',
            maxSum='max(alpha*similarity)'),
        reference_weight='block_legal_windows/(reference_trajectory_legal_windows*trajectory_count)',
        source_weight='block_legal_windows_after_equal_block_rank_mapping_once',
        ranking='frozen_V7_rank_curve_no_gating',
        bp='existing_CRB_endpoint_rows_only; zero_new_BP',
        methods=list(ARMS))


def shape_rows(left, right):
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2 or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('Expected matching finite endpoint gradient matrices')
    midpoint = 0.5 * (left + right)
    slope = (right - left) * INV_SQRT12
    energy = np.einsum('ij,ij->i', midpoint, midpoint) + np.einsum('ij,ij->i', slope, slope)
    result = np.concatenate((midpoint, slope), axis=1)
    positive = energy > 0
    result[positive] /= np.sqrt(energy[positive])[:, None]
    result[~positive] = 0
    return result.astype(np.float32)


def integrated_squared_distance(left_shape, right_shape):
    """Distance of normalized unit-interval linear reconstructions."""
    left_shape, right_shape = np.asarray(left_shape, np.float64), np.asarray(right_shape, np.float64)
    if left_shape.shape != right_shape.shape or left_shape.ndim != 1:
        raise ValueError('Shape vectors must have the same dimension')
    return float(np.dot(left_shape-right_shape, left_shape-right_shape))


def endpoint_positions(blocks, index):
    """Physical cached anchor rows, never inferred from a half-open owned block end."""
    endpoints = np.empty((len(blocks), 2), dtype=np.int64)
    for row, block in enumerate(blocks):
        lo, hi = index.bounds(int(block.fragment_id))
        left, right = lo + int(block.left_index), lo + int(block.right_index)
        if not lo <= left <= right < hi:
            raise ValueError('CRB block endpoint outside owning trajectory')
        endpoints[row] = (left, right)
    return endpoints


def reference_weights(blocks, results):
    legal = {int(r.fragment_id): len(r.valid_starts) for r in results}
    if not legal or any(n <= 0 for n in legal.values()):
        raise ValueError('Missing legal reference trajectory')
    owned = dict.fromkeys(legal, 0)
    weights = []
    for block in blocks:
        fid = int(block.fragment_id)
        if fid not in legal or block.size < 1:
            raise ValueError('Invalid reference block owner')
        owned[fid] += block.size
        weights.append(block.size / (len(legal) * legal[fid]))
    if owned != legal or not np.isclose(sum(weights), 1.0, rtol=0, atol=1e-12):
        raise ValueError('Reference blocks must own every legal window once')
    return np.asarray(weights, dtype=np.float64)


def feature_cache(path, *, cache, dataset, index, blocks, fields, batch=512):
    """Atomic field-validated NPY cache; source shapes may be shared by tasks."""
    path = Path(path)
    ends = endpoint_positions(blocks, index)
    marker, payload, positions = path/'cache_manifest.json', path/'features.npy', path/'endpoints.npy'
    projection = cache.manifest['model']['projection_dimension']
    if projection != 4096:
        raise ValueError('Shape supplement requires the frozen 4096-D V7 projection')
    expected = dict(fields, version=VERSION, dataset=dataset, blocks=len(blocks),
                    dimension=2*projection)
    if marker.exists():
        if json.loads(marker.read_text()) != expected or not payload.is_file() or not positions.is_file():
            raise ValueError('Shape feature cache manifest or payload differs')
        with positions.open('rb') as stream:
            old = np.load(stream, allow_pickle=False)
        if not np.array_equal(old, ends):
            raise ValueError('Shape feature endpoint identities differ')
        features = np.load(payload, mmap_mode='r', allow_pickle=False)
        if features.shape != (len(blocks), expected['dimension']) or features.dtype != np.float32:
            raise ValueError('Shape feature geometry differs')
        return features, dict(cached=True, blocks=len(blocks), new_bp=0, path=str(path))
    if payload.exists() or positions.exists():
        raise RuntimeError('Unmarked shape feature payload exists; recovery audit required')
    path.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    temp_feature = temp_ends = None
    try:
        with tempfile.NamedTemporaryFile(dir=path, delete=False) as stream:
            temp_feature = Path(stream.name)
        feature = np.lib.format.open_memmap(temp_feature, mode='w+', dtype=np.float32,
                                            shape=(len(blocks), expected['dimension']))
        for begin in range(0, len(blocks), batch):
            rows = ends[begin:begin+batch]
            left = cache.get(index_request(dataset, index, rows[:, 0]))
            right = cache.get(index_request(dataset, index, rows[:, 1]))
            feature[begin:begin+len(rows)] = shape_rows(left, right)
        feature.flush()
        del feature
        with tempfile.NamedTemporaryFile(dir=path, delete=False) as stream:
            temp_ends = Path(stream.name)
            np.save(stream, ends)
        os.replace(temp_feature, payload)
        os.replace(temp_ends, positions)
        atomic_json(marker, expected)
    finally:
        for temporary in (temp_feature, temp_ends):
            if temporary is not None and temporary.exists():
                temporary.unlink()
    return np.load(payload, mmap_mode='r', allow_pickle=False), dict(
        cached=False, blocks=len(blocks), new_bp=0, elapsed_seconds=time.perf_counter()-started,
        path=str(path))


def scores_many(reference_shapes, source_shapes, weights, *, aggregations=AGGREGATIONS):
    """Selected V7 transforms share one dot tile; full Shape keeps all four."""
    return reference_scores_many(reference_shapes, source_shapes, weights,
                                 aggregations=aggregations)
