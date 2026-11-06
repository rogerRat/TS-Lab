from pathlib import Path
p=Path('temp/v5-kappa-20260918')
s=(p/'crb.before.py').read_text()
s=s.replace('"closest_zero_v4", "closest_zero_v5"):', '"closest_zero_v4", "closest_zero_v5", "min_kappa_v5"):')
s=s.replace('closest_zero_v4, or closest_zero_v5")','closest_zero_v4, closest_zero_v5, or min_kappa_v5")')
s=s.replace('("fixed", "trajectory_p10"):', '("fixed", "trajectory_p10", "none"):')
s=s.replace('epsilon_policy must be fixed or trajectory_p10','epsilon_policy must be fixed, trajectory_p10, or none')
needle='        if self.context_length <= 0'
insert='''        if self.split_policy == "min_kappa_v5":
            if self.epsilon_policy != "none" or self.priority_tie_policy != "right":
                raise ValueError("min_kappa_v5 requires epsilon_policy=none and priority_tie_policy=right")
            if self.initial_placement != "longest":
                raise ValueError("min_kappa_v5 requires deterministic longest coverage")
        elif self.epsilon_policy == "none":
            raise ValueError("epsilon_policy=none requires min_kappa_v5")
'''
assert needle in s;s=s.replace(needle,insert+needle,1)
s=s.replace('    epsilon_audit: dict | None = None\n','    epsilon_audit: dict | None = None\n    split_audit: dict | None = None\n',1)
needle='    if total == 0:\n'
insert='''    kappa_mode = config.split_policy == "min_kappa_v5"

    def kappa_audit(used):
        if not kappa_mode:
            return None
        return dict(policy="min_kappa_v5", fragment_id=int(fragment.fragment_id),
                    priority="norm_endpoint_sum_over_sum_endpoint_norms",
                    zero_zero_kappa=1.0, priority_tie_policy="right",
                    recursive=True, tie_rng_enabled=False,
                    split_budget=split_budget, split_used=used,
                    split_unused=split_budget-used, split_stop_reason="budget_exhausted")
'''
assert needle in s;s=s.replace(needle,insert+needle,1)
s=s.replace('return CRBResult(fragment.fragment_id, starts, {}, [], 0, 0, 0, 0, 0, 0)', 'return CRBResult(fragment.fragment_id, starts, {}, [], 0, 0, 0, 0, 0, 0,\n                         split_audit=kappa_audit(0))',1)
s=s.replace('return CRBResult(fragment.fragment_id, starts, anchors, [block], 0, 0, 0, used, 0, 0)', 'return CRBResult(fragment.fragment_id, starts, anchors, [block], 0, 0, 0, used, 0, 0,\n                         split_audit=kappa_audit(0))',1)
needle='''        heapq.heappush(frontier_heap, (
            CRBEngine._interval_mean_norm(item, anchors), item.left, item))'''
replacement='''        if kappa_mode:
            # No endpoint gate or quantile, including two zero endpoints.
            heapq.heappush(frontier_heap, (
                CRBEngine._interval_kappa(item, anchors), -item.left, item))
        else:
            heapq.heappush(frontier_heap, (
                CRBEngine._interval_mean_norm(item, anchors), item.left, item))'''
assert needle in s;s=s.replace(needle,replacement,1)
s=s.replace('        limit = 1 if strict_recursive else split_budget - split_used\n        if config.priority_tie_policy != "left":', '        limit = 1 if (strict_recursive or kappa_mode) else split_budget - split_used\n        if kappa_mode:\n            frontier = [heapq.heappop(frontier_heap)[2]]\n        elif config.priority_tie_policy != "left":',1)
needle='    if strict_recursive:\n        epsilon_audit.update('
replacement='''    if kappa_mode and split_used != split_budget:
        raise RuntimeError("min_kappa_v5 did not consume its capped split budget")
    if strict_recursive:
        epsilon_audit.update('''
assert needle in s;s=s.replace(needle,replacement,1)
s=s.replace('banned, cover_used, error_used, split_used, epsilon_audit,\n','banned, cover_used, error_used, split_used, epsilon_audit, kappa_audit(split_used),\n',1)
needle='    @staticmethod\n    def _interval_mean('
insert='''    @staticmethod
    def _interval_kappa(interval: _Interval, anchors: dict[int, np.ndarray]) -> float:
        left = anchors[interval.left].astype(np.float64)
        right = anchors[interval.right].astype(np.float64)
        denominator = float(np.linalg.norm(left) + np.linalg.norm(right))
        numerator = float(np.linalg.norm(left + right))
        if not np.isfinite(denominator) or not np.isfinite(numerator):
            raise ValueError("Non-finite kappa inputs")
        if denominator == 0:
            return 1.0  # No evidence of cancellation; still competes for budget.
        return min(1.0, max(0.0, numerator / denominator))

'''
assert needle in s;s=s.replace(needle,insert+needle,1)
s=s.replace('    epsilon_policy: str = "fixed"  # Legacy explicit thresholds; production uses trajectory_p10.', '    epsilon_policy: str = "fixed"  # Legacy thresholds; min_kappa_v5 explicitly uses none.')
s=s.replace('    ``closest_zero_v5`` recurses with a strict > P10 gate and current-minimum selection.', '    ``closest_zero_v5`` recurses with a strict > P10 gate and current-minimum selection.\n    ``min_kappa_v5`` has no threshold, minimizes relative cancellation, breaks\n    exact ties to the right, and consumes the entire capped split budget.')
(p/'crb.py').write_text(s)

mapping={'full_pool_worker.py':'src/rage/experiments/full_pool_worker.py',
         'formal_main_entry.py':'src/rage/experiments/formal_main_entry.py',
         'full_pool_run.py':'src/rage/experiments/full_pool_run.py',
         'full_pool_assets.py':'src/rage/experiments/full_pool_assets.py',
         'blocks.py':'src/assets/gradient/blocks.py',
         'prepare.py':'src/assets/gradient/prepare.py'}
for name in mapping:
    path=p/name;s=path.read_text();(p/(name+'.before')).write_text(s)
    if name=='full_pool_worker.py':
        s=s.replace('    calibration = None if local_p10 else load_p10_calibration', '    no_threshold = config["crb"].get("epsilon_policy") == "none"\n    calibration = None if (local_p10 or no_threshold) else load_p10_calibration')
        s=s.replace('**({} if local_p10 else dict(', '**({} if (local_p10 or no_threshold) else dict(')
    elif name=='formal_main_entry.py':
        s=s.replace("if config['crb'].get('epsilon_policy') == 'trajectory_p10':", "if config['crb'].get('epsilon_policy') in ('trajectory_p10', 'none'):")
    elif name=='full_pool_run.py':
        s=s.replace('if config["crb"].get("epsilon_policy") != "trajectory_p10":', 'if config["crb"].get("epsilon_policy") not in ("trajectory_p10", "none"):')
    elif name=='full_pool_assets.py':
        s=s.replace("if protocol['crb'].get('epsilon_policy') != 'trajectory_p10':", "if protocol['crb'].get('epsilon_policy') not in ('trajectory_p10', 'none'):")
        s=s.replace('audit=json.dumps(dict(counts=trajectory_counts(result), epsilon=result.epsilon_audit)))', 'audit=json.dumps(dict(counts=trajectory_counts(result), epsilon=result.epsilon_audit,\n                             **({"split":result.split_audit} if result.split_audit is not None else {}))))')
    elif name=='blocks.py':
        needle='    started = time.time()\n'
        assert needle in s;s=s.replace(needle,'    if config.split_policy == "min_kappa_v5":\n        counters["split_audits"] = []\n'+needle,1)
        needle='        if result.epsilon_audit is not None:\n'
        assert needle in s;s=s.replace(needle,'        if result.split_audit is not None:\n            counters["split_audits"].append(result.split_audit)\n'+needle,1)
    elif name=='prepare.py':
        start=s.index('            atomic_json(audit_root / ("crb-thresholds-"')
        end=s.index('            from .variants import',start)
        old=s[start:end]
        new='''            if crb_config.split_policy == "min_kappa_v5":
                atomic_json(audit_root / ("crb-splits-" + readable(dataset_tag) + ".json"),
                    dict(schema="min-kappa-split-v5", model=model_identity, crb=asdict(crb_config),
                         source=prepared.source_counters['split_audits'],
                         reference=[r.split_audit for r in prepared.reference_results]))
            else:
'''+''.join('    '+line+'\n' for line in old.splitlines())
        s=s[:start]+new+s[end:]
    assert s != (p/(name+'.before')).read_text(), name
    path.write_text(s)
mapping['crb.py']='src/rage/crb.py'
(p/'mapping.json').write_text(__import__('json').dumps(mapping))
