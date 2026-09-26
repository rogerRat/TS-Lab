"""Explicit, read-only donor migration; never alias a different pool manifest."""
import json
from pathlib import Path
import numpy as np
from assets.gradient.cache import GradientCache
from assets.gradient.dataset import decode_bf16

TIMER_REUSE_APPROVAL = 'user-approved-timer-donor-20260923-v1'

def approved_timer_model_pair(old, new):
    """One explicitly authorized migration, not general version normalization."""
    required = dict(model_id='thuml/timer-base-84m', student='timerxl-84m',
        revision='70077a71acce1b4c00d98332fcaabc694255d8e5',
        ckpt_revision='70077a71acce1b4c00d98332fcaabc694255d8e5',
        loaded_parameter_dtypes=['torch.bfloat16'],
        precision_policy='bf16-model-gradient-adamw-v2')
    if not isinstance(old, dict) or not isinstance(new, dict):
        return False
    if any(old.get(k) != v or new.get(k) != v for k, v in required.items()):
        return False
    a, b = old.get('config'), new.get('config')
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    if (a.get('torch_dtype') != 'bfloat16' or 'dtype' in a or
            b.get('dtype') != 'bfloat16' or 'torch_dtype' in b or
            a.get('transformers_version') != '4.40.1' or
            b.get('transformers_version') != '4.57.1'):
        return False
    a = {k: v for k, v in a.items() if k not in ('torch_dtype', 'transformers_version')}
    b = {k: v for k, v in b.items() if k not in ('dtype', 'transformers_version')}
    return a == b and {k: v for k, v in old.items() if k != 'config'} == {
        k: v for k, v in new.items() if k != 'config'}

class V7Cache(GradientCache):
    def __init__(self, *args, donors=(), readonly=False, shared_data=None, **kwargs):
        if shared_data is not None:
            kwargs['data_fields'] = shared_data
        super().__init__(*args, **kwargs)
        self.readonly = readonly
        self.imported_windows = 0
        self.donor_rows = {}
        self.reuse_audit = []
        for donor, mapping, references in donors:
            donor = Path(donor)
            fields = json.loads((donor/'cache_manifest.json').read_text())
            differences = [k for k in self.manifest if k != 'data' and fields.get(k) != self.manifest[k]]
            original_differences = list(differences)
            approved = differences == ['model'] and approved_timer_model_pair(
                fields.get('model'), self.manifest.get('model'))
            if approved:
                differences = []
            audit = dict(path=str(donor), differing_fields=differences,
                         matched=not differences, source_trajectories=len(mapping), references=references)
            if approved:
                audit.update(original_differing_fields=original_differences,
                    compatibility_approval=TIMER_REUSE_APPROVAL,
                    numerical_equivalence_claimed=False,
                    approved_model_fields=['config.torch_dtype->config.dtype',
                                           'config.transformers_version:4.40.1->4.57.1'])
            self.reuse_audit.append(audit)
            if differences:
                continue
            with np.load(donor/'window_index.npz', allow_pickle=False) as rows:
                datasets, shards = rows['datasets'], rows['shards']
                for di, (fid, start), si, row in zip(rows['dataset_ids'], rows['window_ids'],
                                                    rows['shard_ids'], rows['shard_rows']):
                    dataset = str(datasets[di])
                    new_fid = mapping.get(int(fid)) if dataset == 'source' else int(fid)
                    if new_fid is None or (dataset != 'source' and dataset not in references):
                        continue
                    self.donor_rows.setdefault((dataset, new_fid, int(start)),
                        (donor/str(shards[si]), int(row), int(fid)))

    def prepare(self, requests):
        if self.manifest.get('data', {}).get('extension_source_reuse'):
            missing_source = [(int(fid), int(start)) for r in requests if r.dataset == 'source'
                              for fid, start in r.ids
                              if ('source', int(fid), int(start)) not in self.rows]
            if missing_source:
                raise RuntimeError(f'Extension source reuse incomplete: {len(missing_source)} rows; no BP allowed')
        pending = { (r.dataset, int(fid), int(start)) for r in requests for fid, start in r.ids
                    if (r.dataset, int(fid), int(start)) not in self.rows }
        if pending and (self.readonly or self.frozen or self._lock.closed):
            raise RuntimeError(f'V7 selection zero-BP gate: {len(pending)} missing rows')
        grouped = {}
        for key in pending:
            if key in self.donor_rows:
                path, row, old_fid = self.donor_rows[key]
                grouped.setdefault((path, key[0]), []).append((key, row, old_fid))
        for (path, dataset), entries in grouped.items():
            with np.load(path, allow_pickle=False) as data:
                if str(data['dataset']) != dataset:
                    raise ValueError('Donor dataset differs from published index')
                rows = np.array([r for _, r, _ in entries], dtype=np.int64)
                expected = np.array([(fid, key[2]) for key, _, fid in entries])
                if not np.array_equal(data['window_ids'][rows], expected):
                    raise ValueError('Donor physical window keys differ')
                bits = data['gradients_bf16'][..., rows, :]
                losses = data['losses'][..., rows]
                if not np.isfinite(decode_bf16(bits)).all() or not np.isfinite(losses).all():
                    raise ValueError('Nonfinite donor gradient/loss')
                target, offset = self._write_shard(dataset,
                    np.array([key[1:] for key, _, _ in entries]), bits, losses)
            for i, (key, _, _) in enumerate(entries):
                self.rows[key] = (target, offset+i)
            self.imported_windows += len(entries)
        super().prepare(requests)

def verified_donors(config, student, pool, manifest, refs_by_task):
    """Match native row + complete values, not reassigned fragment numbers alone."""
    if 'task_extension' in config:
        # Extension setup already publishes field-verified same-inode source
        # shards; never fall back to an unrelated V6 reference namespace.
        return []
    from .development_inputs import load_fragments
    legacy = Path(config['legacy_inputs'][student])
    old_manifest = json.loads((legacy/'pool_manifest.json').read_text())
    old_pool = load_fragments(legacy/'pool.npz')
    mapping = {}
    if all(old_manifest.get(k) == manifest.get(k) for k in ('source_load', 'window_length', 'pool_protocol')):
        new_rows = {row: i for i, row in enumerate(manifest['rows'])}
        for i, row in enumerate(old_manifest['rows']):
            if row in new_rows:
                j = new_rows[row]
                old, new = old_pool[i], pool[j]
                if old.entity_id != new.entity_id or not np.array_equal(old.data, new.data):
                    raise ValueError('Same native row has changed source values')
                mapping[int(old.fragment_id)] = int(new.fragment_id)
    references = []
    for task, refs in refs_by_task.items():
        old_refs = load_fragments(legacy/f'{task}__reference.npz')
        if len(refs) != len(old_refs) or any(a.fragment_id != b.fragment_id or
                a.entity_id != b.entity_id or not np.array_equal(a.data, b.data)
                for a,b in zip(refs, old_refs)):
            raise ValueError('Reference physical identities/values differ from donor task')
        references.append('reference:'+task)
    donors = []
    for root in config['legacy_gradient_roots']:
        for marker in Path(root).glob('*/cache_manifest.json'):
            fields = json.loads(marker.read_text())
            if fields.get('data', {}).get('pool') == old_manifest and (marker.parent/'window_index.npz').exists():
                verified_refs=[]
                from assets.gradient.cache import readable
                for task, refs in refs_by_task.items():
                    ref_marker=marker.parent/('reference-'+readable(task)+'.json')
                    expected=dict(task=task,fragments=[dict(fragment_id=int(f.fragment_id),
                        entity_id=str(f.entity_id),length=int(f.length)) for f in refs])
                    if ref_marker.exists() and json.loads(ref_marker.read_text())==expected:
                        verified_refs.append('reference:'+task)
                donors.append((marker.parent, mapping, verified_refs))
    return donors
