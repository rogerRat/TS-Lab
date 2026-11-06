# External selection methods

This directory owns the external methods, not the project's RAGE algorithms:

- `less/`, `tsds/`, `lp_mirror/`, `ltsv/`: one independent Python package per method.
  Each owns `algorithms.py` (pure kernels), `runner.py` (model/data execution),
  `contract.py` (pinned source and fidelity), and a public `select` entry point.
- `runners.py`: unified `run_baseline` dispatcher only.
- `contracts.py`: unified registry; contracts are owned by each method package.
- `common.py`, `_contracts.py`: shared window helpers and provenance schema.
- Top-level `algorithms.py` contains compatibility re-exports, not implementations.

```python
from baselines.lp_mirror import select, CONTRACT
from baselines.ltsv.algorithms import ltsv_window_scores
```

LESS and TSDS are language-side adaptations. LP-Mirror preserves the official
linear-proxy Adam/SGD/loss-drop/top-k method. LTSV follows the paper's one-step
valuation and block/point/window equations; its official HEAD is not a complete
runnable checkout. Source anchors remain in `requ/baselines.lock`.

Candidate/ref provenance, C/H lengths and final evaluation are experimental
settings. Model-specific training/readout changes are disclosed in contracts.
LP-Mirror and LTSV retain overlapping top-k; do not equate N with unique coverage.

Install from the repository root with `python -m pip install --no-deps -e .`.
See `docs/dev/BaselineAudit.md` for method details and verification scope.
