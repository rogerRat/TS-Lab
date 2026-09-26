# External baseline audit for static selection

Audit date: 2026-09-12. Immutable source anchors are in `requ/baselines.lock`;
runtime result provenance is emitted from `baselines.contracts`.
The shared constraint is deliberate: every method sees one fixed feasible source
pool and target safe history, and may return only identities already in that pool.

## LESS

- Audited source: `princeton-nlp/LESS` at `8abf9628`.
- Upstream files: `collect_grad_reps.py`, `matching.py`, and the data-selection
  scripts/README.
- Exact upstream signal: for a source example, load the checkpoint Adam moments,
  compute `(0.9*m + 0.1*g) / sqrt(0.999*v + 0.001*g^2 + 1e-8)`, random-project,
  and L2-normalize. Target examples use normalized SGD gradients. Per-checkpoint
  source-target dot products are combined with normalized average-learning-rate
  checkpoint weights.
- Preserved here: the exact Adam-direction kernel, normalization, matrix-product
  influence, checkpoint weight normalization, multi-checkpoint axis, projected
  per-example features, and top-score selection under the shared deployment.
- Current adaptation (decision 11): one trajectory is one example. Uniformly
  sampled raw window gradients are averaged before cosine matching, on both
  source and reference sides. Counts match the configured three RAGS budgets;
  score is the sum of per-reference-trajectory cosines, followed by ranked
  whole-trajectory point-budget delivery. LoRA is replaced by full-parameter projection.
- Bilateral SGD gradients and one pretrained checkpoint are explicit decisions,
  not pending work. No Adam modulation is used by the selector; the historical
  Adam/multi-checkpoint kernels remain independently tested but unused here.
  The arm remains labelled `core-signal-only`, never “original LESS.”

## TSDS

- Audited source: `ZifanL/TSDS` at `3dfe1395`, especially `tsds.py` and
  `faiss_helper.py`.
- Preserved here: candidate/query kNN, compact-support KDE values, the heap-based
  optimal-transport probability assignment, multinomial sampling, and repeated
  indices for samples drawn more than once. A literal small-array copy of the
  upstream loop is used as an independent parity oracle in tests.
- Current adaptation (decision 12): one trajectory is one OT point, represented
  by the mean of uniformly sampled Chronos encoder window features. Both source
  and reference use the same configured RAGS budget counts. Candidate and KDE
  searches use upstream FAISS IVF, including default nlist/nprobe; exact Torch
  kNN is retired.
- Multinomial copies are retained: shuffle copies, then deliver whole trajectories
  until their cumulative length first fills N*(C+H) points (decision 15).
  Duplicate identities survive saving/loading and actual student training;
  no unselected trajectories are added. Uniform legal-start training sampling is a disclosed
  deployment adaptation, not an operation defined by the original text selector.
- User-frozen `max_k=64, kde_k=16, sigma=4, alpha=.1, c=5` are **experiment
  parameters**, not the upstream config defaults (`5000, 1000, .75, .5, 5`).
- Guard clarified: upstream implicitly requires at least two retrieved
  neighbours. The port now rejects smaller inputs explicitly instead of silently
  changing the `k+1` transport step.

## LP-Mirror

- Audited source: `LP-Mirror/LP_Mirror` at `43d01017`, especially
  `algorithm/forward_inf.py`.
- Preserved here: train a proxy on the complete source set, clone its state,
  adapt it to target data with SGD over a full shuffled target epoch, and score every candidate as
  `loss_original - loss_adapted` before selection.
- Static adaptations: a channel-independent linear proxy operates directly on
  the fixed univariate C/H windows. Scaling is fitted on each supplied safe
  trajectory (dataset/channel adapter), not separately on each window context.
  Source Adam training visits every window once per epoch in index order,
  including the final partial batch. Native top-k retains overlapping windows;
  fixed N chooses the cutoff, without an additional non-overlap filter.
- Remaining fidelity gap: the upstream dataset loader, backbone pipeline, and
  optional GMM cutoff are outside this fixed-budget experiment.

## LTSV

- Paper: arXiv `2511.11648`, Eqs. 8--11 and its experimental protocol.
- Official source: `b1302550313/LTSV` HEAD `38221dc7`; historical runnable
  valuation driver at `82d3f94b`.
- Preserved here: one optimizer step per overlapping block, context loss before
  minus after that step, block-to-point averaging, and point-to-window averaging.
  The paper's Adam settings (learning rate `1e-5`, betas `0.9/0.999`, epsilon
  `1e-8`, weight decay `0.1`, batch size 1, fp32, clipping 1.0) are the normative
  contract.
- Static adaptations: external target safe history replaces paper five-fold
  context partitions; this changes the experiment, not the valuation equation.
  Block training uses native Chronos CE. Reference scoring uses actual forecast
  MSE in original units, with deterministic autoregressive greedy decoding from
  context alone (no teacher-forced future labels). Longer blocks supply all C/H
  subwindows to one mean-loss update. These are explicit Chronos model adapters.
  Native top-k retains overlapping windows. Point means are computed BEFORE
  window means; the previous coverage-weighted aggregation was incorrect.
- Source caveat: official HEAD stores Time-MoE and Time-Series-Library as gitlinks
  but provides no `.gitmodules`, and removes the valuation driver. The historical
  driver launches one epoch per sample, which does NOT alone establish a conflict
  with one step: the actual number of batches matters. This implementation follows
  the paper's explicit one-step update and is labelled `paper-equation-port`,
  not official-code parity.

## Test evidence

`tests/test_baseline_algorithms.py` checks the LESS Adam equation and weighted
checkpoint matrix product, TSDS probability assignment against a literal locked
source loop, and LP-Mirror loss-drop sign. `tests/test_baseline_contracts.py`
checks that all four methods publish their immutable provenance and unresolved
gaps. `run_baseline` additionally verifies every returned identity using the
shared `WindowIndex.locate` before any downstream training begins.

`tests/test_time_series_baselines.py` adds independent LP-Mirror training/score
parity, complete-epoch coverage, literal LTSV aggregation parity with unequal
coverage, context-only forecast MSE, native top-k, longer-block execution, and
model restoration after both successful and failed reference evaluation.

## Budget interpretation

Language-side LESS/TSDS adaptations remain accepted. Time-series selectors retain
their method and native top-k. N is a selected-window count for LP-Mirror/LTSV,
not a guarantee of N disjoint fresh windows. Final student initialization and
training/evaluation schedules remain shared; selection-side training does not
make the final selection dynamic. Do not claim identical unique data coverage.
