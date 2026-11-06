# RAGE performance verification — 2026-09-08

Changes in `engines.py`, `experiments/train_eval.py`, and `experiments/arms.py`:

- Pre-tokenize repeatedly sampled training windows using the existing CPU
  Chronos tokenizer. Cache lifetime is one training call, bounded to 64 MiB;
  oversized sets and sets with no expected reuse use the original path.
  The NumPy sampling RNG, row order, optimizer, precision, and loss remain unchanged.
- Explicitly supply T5 decoder input IDs obtained from the model's own label
  shift routine. This avoids computing the unused internal cross-entropy while
  retaining the existing per-sample cross-entropy, teacher forcing, and masks.
  Applied to single-window loss/gradient, vmap gradients, and fixed-step training.
- Reduce repeated source-block recall scores before expanding windows. The
  final maximum reduction still handles overlapping endpoints. Ranking,
  tie-breaking and selected-window ordering remain unchanged.

## Evidence

Hardware: RTX 4090; official local Chronos-T5-small checkpoint; float32;
512 context points and 64 prediction points. CPU threads were set to 4 for
the real-model comparisons. No formal experiment batch was launched.

| Check | Result |
|---|---|
| Tokenization, repeated indices and constant/zero windows | Bitwise equal; bounded-cache fallback verified |
| Fixed-step training, 20 steps, batch 16 | All final model tensors and reported losses exactly equal |
| Vmap gradient batch of 16, two runs | Before 0.274/0.306 s; after 0.106/0.107 s |
| Gradient numerical differences | Loss exactly equal; max sketch absolute difference below 5e-8; tied weights preserved |
| Repeated-recall stress case, 10,000 queries | Before 0.826 s; after 0.00465 s; selected windows and diagnostics exactly equal |
| Selection → training → evaluation, seeds 0/1/2 | Selected window IDs, data, 30 gradient calls, all losses and MASE/SQL/MAE exactly equal |
| Same small pipeline timing | Before 2.648 s; after 2.551 s; selection 0.506 → 0.441 s |

The pipeline uses synthetic finite fragments (four source fragments of length
1152 and two reference fragments of length 800), four selected training
windows, 20 steps per seed, and four evaluation windows. These checks establish
equivalence on exercised inputs; they do not establish a formal-batch speedup
or long-training outcome distribution. Short training timings were noisy;
no standalone training speedup is claimed. Recall stress speedup is specific
to repeated hits, not a whole-experiment estimate.

39 relevant tests passed:

```sh
/root/autodl-tmp/rage/envs/rage/bin/python -m pytest -q --import-mode=importlib \
  tests tests/test_rage_run.py tests/test_crb.py
```

The broader check including `tests/test_rage_ablation.py` had 44 passes
and two failures. Both were reproduced with the pre-optimization arms module:
the legacy test expects indices instead of the current `(values, indices)`
return, and its Namespace lacks `n_windows`. Those pre-existing test-contract
issues were not changed as part of performance work.

## Verification artifacts on rage

Run from `/root/TS-Lab` with `PYTHONPATH=.` and the rage Python environment:

- `/tmp/rage-perf-verify-training.py`
- `/tmp/rage-verify-gradient.py`
- `/tmp/rage-perf-verify-recall.py`
- `/tmp/rage-verify-pipeline.py`

The scripts use baseline snapshots `/tmp/rage-engines-pre-forward.py`,
`/tmp/rage-perf-arms-before.py`, and
`/tmp/rage-perf-training-before.py`. These temporary artifacts can be
removed by host cleanup; retain them when reproducing the comparisons.

Verified source SHA-256 values:

```text
71542672ec0da2d4a8c4edd2706886f5ab818803d7d642b1686e632dca37a263  rage/engines.py
188aea4214be010c43f543a979e39e78364f11e49b039b1897114a15529f077a  rage/experiments/arms.py
20b8151f1241934a22efe5623258fe343914cbe960b31580d0cb979dc089c6ab  rage/experiments/train_eval.py
```
