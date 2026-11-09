# Gradient proxy experiment

Last modified: 2026-11-08
Experiment: v0-gradient-proxy-test
Status: Local-Risk and Cross-Rank complete; execution and results tracked in ORCHE.md, LOG.md, and the result report.

## Objective and fixed inputs

Assess whether inexpensive per-window quantities preserve full-gradient direction
and magnitude geometry, and whether they track within-block gradient dispersion.
Direction is the primary outcome. This is proxy assessment, not a CoVar training
or allocation experiment. Existing CRB jobs continue unless resource pressure
requires stopping them, as authorized by the user.

Reuse all 12,836 windows of the ten retained trajectories at source ordinals
973,980,981,984,985,987,988,996,998,999. The sample came from the unprocessed tail
of the original first-1,000-eligible M5 selection. It is not a representative
sample of all M5 series. Use the original 100-step warmed checkpoint, context
512, prediction 64, stride 1, evaluation mode, identical tokenizer and per-token
cross entropy averaged over nonignored labels. Valid EOS labels are included.
Teacher-forced decoder features and loss require the observed future labels.

Reference gradients are complete BF16 copies of original FP32 full BP gradients.
No projection or coordinate subsampling is used. Three windows (first, middle,
last) per trajectory are recomputed with FP32 BP to check cache alignment;
relative L2 error must be <=0.01 and cosine >=0.9999. These are validation costs,
not deployable proxy costs. Input identity, source order, file sizes and completion
markers are checked without hashes.

## Proxies

1. Output-head gradient: exact derivative of the original loss through the output
   linear map only, keeping its input detached. T5 shares lm_head.weight with
   input embeddings; this proxy excludes the encoder/decoder embedding paths.
   It is consequently distinct from the shared parameter's total BP gradient.
   Use the framework CE derivative with respect to detached logits, then multiply by the actual scaled head input. This is the exact output weight derivative with the backbone held fixed. Mask ignored labels and average over valid tokens. Retain every output weight coordinate in FP32; compare the result with detached-head autograd.
2. Embedding: arithmetic mean of final decoder hidden states over valid output
   token positions, including EOS. These are contextual representations after
   final decoder normalization, before the tied-head rescaling. Not vocabulary
   embedding-table rows. Fixed dimensionality: 256.
3. Forward loss: the same scalar CE as the full-gradient experiment. No scalar
   cosine is reported, since positive scalar losses have trivial cosine one.

## Measurements

- Per-window: Pearson/Spearman association between full-gradient norm and each
  proxy norm, or scalar loss. Full and proxy vectors inhabit different spaces;
  their direct coordinatewise cosine is not defined.
- Primary direction analysis: for the SAME within-trajectory pair, correlate
  full-gradient cosine with output-head and embedding cosine. For loss, correlate
  absolute loss difference with 1 minus full-gradient cosine.
- Report each trajectory and lag separately: 1,2,4,8,16,32,64,128,256,512,575.
  This prevents a shared lag trend from masquerading as local predictive power.
  Also report all within-trajectory pairs as a descriptive pooled-lag view.
- Magnitude: paired L2 distance and absolute norm difference, with matching
  proxy measures. Preserve loss-difference associations for these targets too.
- Nonoverlapping blocks of 32 starts: full vector population variance is
  mean(||g||^2)-||mean(g)||^2. Direction dispersion is
  1-||mean(g/||g||)||^2. Correlate each with corresponding head/embedding
  statistics and loss variance. Save terminal partial blocks but exclude them
  from equal-length block correlations. This uses block-local statistics, not
  the previously pooled variance across pairs at a fixed lag.
- Report all ten trajectory effects individually and summarize with equal
  trajectory weight. No pair-independent p-values: windows and pairs overlap.
  Correlations are descriptive and do not establish held-out allocation gains.

## Execution, artifacts and completion

Entry: scripts/gradient_proxy_experiment.py
Config: configs/rage/v0-gradient-proxy-test.json
Environment: /root/autodl-tmp/rage/envs/rage (confirmed by user).
Command: PYTHONPATH=src /root/autodl-tmp/rage/envs/rage/bin/python -u
  -m scripts.gradient_proxy_experiment --config configs/rage/v0-gradient-proxy-test.json

Use tmux v0-gradient-proxy. Two CPU threads, batch size 8, GPU FP32, TF32 disabled.
Gram matrices use all coordinates, FP32 products and FP64 tile accumulation;
coordinate tiles contain 32,768 dimensions. Existing CRB CPU replay continues.
The host currently has about 439 GiB available RAM and 301 GiB free data disk.
Retained FP32 head gradients need about 50.2 GiB; full gradients are not copied.
Forward/head timing is descriptive, not a matched throughput benchmark against
single-window full BP. Proxy extraction includes hidden-state output overhead.

All artifacts live under outputs/train/v0-gradient-proxy-test on the data disk:
config, parameter layout, per-trajectory source manifests, FP32 head vectors,
window loss/embeddings, Gram matrices, pair metrics, blocks, correlations and
cache alignment checks. Logs live under logs/v0-gradient-proxy-test. Each completed
trajectory gets .complete; final .complete requires all ten analyses. Resume skips
completed extraction; an interrupted trajectory is recomputed. No subprocesses
are created by the Python entry. Progress, heartbeat and ETA logs are enabled.


## Cost benchmark amendment

Last modified: 2026-11-08

The follow-up cost benchmark measures full BP, output-head gradient, final decoder token-mean embedding, and forward CE loss on 8 evenly spaced windows from each of the ten fixed tracks, repeated 3 times. Every call uses batch size 1 and includes CPU tokenization, GPU forward, requested method computation, result copy to CPU and device synchronization. Methods are warmed and interleaved in seeded random order. Proxy output checks compare ten sampled calls to saved proxy features. Per-call latency, per-track medians, overall medians/means/p90, speedups, and descriptive call-bootstrap intervals are written to outputs/train/v0-gradient-proxy-cost/. The benchmark uses the already confirmed rage Conda environment and does not access the active CRB queue because its STOP marker is present.


## Part codes and appended Cross-Rank protocol

Last modified: 2026-11-08

This proxy batch contains two parts with separate questions and outcomes.

### Local-Risk

Question: Within nonoverlapping blocks of 32 consecutive window starts on the ten retained tracks, do inexpensive proxy statistics preserve the ranking of full-gradient vector variance and full-gradient direction dispersion?

Observed result: Other (positive descriptive association, not a binary selection-policy result). Equal-trajectory mean Spearman correlation is 0.9476 between output-head variance and full vector variance, and 0.8161 between output-head direction dispersion and full direction dispersion. The analysis does not test a budgeted proxy-selected interval policy.

### Cross-Rank

Question: For each full-gradient reference window from the two shortest retained tracks, do proxy scores rank a fixed pool of pairwise nonoverlapping source windows from the other eight tracks in the same order as full-gradient cosine similarity? Also measure absolute cosine calibration for vector proxies.

Fixed inputs and selection:

- Reference tracks are 00 and 07, with 607 and 1,270 windows, respectively. Use every reference window.
- Source tracks are 01-06, 08, and 09. In each source track, let L be its series length (window count plus 575) and m=floor(L/576). Select starts floor(i*L/m), for i=0,...,m-1. Verify every selected 576-point span is within the series and source spans do not overlap. The current eight source tracks each contribute three candidates, for 24 candidates total.
- The source tracks are distinct from both reference tracks. All pairs are therefore cross-track and nonoverlapping. Reference windows may overlap one another; each reference window is a separate ranking query.
- Reuse all 8,394,496 full-gradient coordinates from the retained BF16 copies of the FP32 BP gradients. No model or gradient recomputation is required.

Metrics:

- Primary ranking proxy: cosine between output-head-only gradients. Secondary ranking proxies: decoder token-mean embedding cosine and negative absolute CE-loss difference (higher means more similar).
- For every reference window, compare each proxy's ranking across 24 source candidates with full-gradient cosine using Spearman, Kendall tau-b, and top-5 set overlap.
- For head and embedding cosines, report per-reference MAE, RMSE, and signed bias relative to full-gradient cosine.
- Save every reference-source score row, every per-reference ranking summary, the fixed input manifest, and aggregate results. Give query-weighted summaries and separate summaries for each reference track.

Interpretation boundary: This tests ranking on the fixed pool of 24 candidates from eight cached source tracks. It does not establish ranking quality on arbitrary M5 windows or on other datasets/models. Source candidates are independent in time within each source track by construction; query windows overlap and are dependent. No pair-independent significance tests are planned. Report the observed correlations and calibration; classify the result as Yes, No, or Other only with the measured scope and limitations stated.

Entry: python -m scripts.cross_rank_gradient_proxies
Config: configs/rage/v0-gradient-proxy-cross-rank.json
Output: outputs/train/v0-gradient-proxy-test/cross_rank/
Log: logs/v0-gradient-proxy-test/cross-rank.log
Environment: confirmed rage Conda environment; tmux session v0-gradient-proxy-cross-rank; GPU cosine GEMM only, no model load or BP.


## Cross-Rank observed result

Completed 2026-11-08. Result category: Other (qualified positive output-head ranking signal on the fixed pool; moderate per-reference agreement does not establish direct replacement).

The run processed 1,877 reference windows and 24 nonoverlapping source candidates, yielding 45,048 pairs. Output-head cosine had pair-weighted Pearson/Spearman 0.786/0.795; per-reference mean Spearman/Kendall tau-b 0.640/0.481; mean top-5 overlap 0.487 versus random expectation 0.208; cosine MAE/RMSE/bias 0.071/0.091/-0.016. Embedding cosine had mean per-reference Spearman -0.020; negative absolute loss difference had 0.222. Query dependence, only two reference tracks, and the fixed selected tail sample limit interpretation. See docs/results/v0-gradient-proxy-test.md for full reporting.


## Local-Risk sampling extension

Last modified: 2026-11-08

Question: If only k output-head gradients are computed in each 32-start block, with k in {32,24,16,8,4}, how strongly do sampled-head variance and direction-dispersion estimates correlate with the full-gradient variance and direction dispersion computed from all 32 points?

Use the ten existing retained trajectories and the complete nonoverlapping 32-start blocks from the original Local-Risk analysis. Full-gradient target values are reused from each track's blocks.csv. Proxy values are recomputed from each track's saved head Gram matrix; no model load, gradient recomputation, or GPU allocation is needed.

For each block and each of 20 sampling replicates, draw a uniform random permutation of the 32 positions without replacement. Use nested prefixes of sizes 4, 8, 16, and 24 so results at different budgets within one replicate share the same sample ordering. For k=32, use all positions once as the exact baseline. The proxy population variance is mean sampled squared norm minus squared norm of the sampled mean. Proxy direction dispersion is one minus the squared norm of the sampled mean of unit head gradients. Compare each sampled statistic with the corresponding full-gradient statistic across complete blocks within each track.

Report Pearson and Spearman correlations separately for full vector variance and full direction dispersion. Average correlations equally across tracks. For k below 32, report the median and 10th-90th percentile across the 20 sampling replicates; k=32 has one deterministic result. The sampling range is a design-sensitivity summary, not a confidence interval over trajectories. The ten tracks are the selected M5 tail sample, and blocks are not independent because their windows overlap in raw observations.

Entry: scripts/analyze_sparse_proxy_risk.py
Config: configs/rage/v0-gradient-proxy-local-risk-sparse.json
Output: outputs/train/v0-gradient-proxy-test/local_risk_sparse/
Log: logs/v0-gradient-proxy-test/local-risk-sparse/
Environment: rage Conda environment; tmux v0-gradient-proxy-local-risk; CPU-only analysis of saved Gram matrices.


## Local-Risk sparse-sampling observed result

Completed 2026-11-08 over 396 complete blocks and ten retained tracks. Result category: Other. At k={32,24,16,8,4}, equal-track median Spearman for sampled output-head variance versus full-gradient variance is {0.948,0.942,0.935,0.916,0.867}; for sampled head direction dispersion versus full direction dispersion it is {0.816,0.806,0.783,0.704,0.579}. At k below 32, the respective 10th-90th sampling-replicate ranges are {0.938-0.945,0.930-0.940,0.907-0.922,0.852-0.889} and {0.801-0.813,0.776-0.798,0.688-0.737,0.538-0.619}. Pearson results are in outputs/train/v0-gradient-proxy-test/local_risk_sparse/summary.csv.

The analysis uses saved head Gram matrices to retrospectively subsample exact proxy vectors. It quantifies statistical information loss but does not measure runtime savings. It does not test an interval-selection policy or downstream training outcome.


## Local-Risk 64-point sparse-sampling extension

Last modified: 2026-11-08

Question: In each consecutive 64-start block, if only k output-head gradients are computed for k in {64,48,32,16,8,4}, how strongly do sampled proxy variance and direction dispersion correlate with the true full-gradient variance and direction dispersion computed over all 64 points?

For each track, use complete consecutive blocks of 64 window starts and omit only the final partial block. Compute full-gradient variance and direction dispersion over all 64 full gradients using the saved full-coordinate Gram matrix. Compute output-head proxy statistics from the saved head Gram matrix using only selected positions. No model or gradient recomputation is performed.

For k below 64, generate 20 fixed-seed uniform samples without replacement per block. Within a replicate, the sets are nested by using one random permutation and prefixes of size 4, 8, 16, 32, and 48. The 64-point result uses all positions once. For each track, correlate block-level sampled proxy variance with full 64-point full-gradient variance, and sampled head direction dispersion with full 64-point full-gradient direction dispersion. Report Pearson and Spearman separately, equal-track averages, and the median and 10th-90th percentiles across sampling replicates. These sampling ranges are design sensitivity summaries, not confidence intervals over trajectories.

The observed windows within blocks overlap in raw observations; blocks are disjoint by starting index but not independent in underlying data. The analysis is a retrospective subsampling of saved Gram matrices: it tests statistical information loss, not online runtime savings or downstream interval-selection gains.

Entry: scripts/analyze_sparse_proxy_risk_64.py
Config: configs/rage/v0-gradient-proxy-local-risk-64.json
Output: outputs/train/v0-gradient-proxy-test/local_risk_sparse_64/
Log: logs/v0-gradient-proxy-test/local-risk-64/
Environment: rage Conda environment; tmux v0-gradient-proxy-local-risk-64; CPU-only saved-Gram analysis.


## Local-Risk 64-point sparse-sampling observed result

Completed 2026-11-08 over 194 complete blocks from ten retained tracks. Result category: Other. At k={64,48,32,16,8,4}, equal-track median Spearman for sampled output-head variance versus full-gradient variance is {0.935,0.931,0.924,0.914,0.886,0.837}; for sampled head direction dispersion versus full direction dispersion it is {0.808,0.808,0.805,0.771,0.693,0.557}. At k below 64, the respective sampling ranges are {0.926-0.935,0.920-0.929,0.898-0.924,0.868-0.899,0.792-0.859} and {0.800-0.816,0.786-0.812,0.741-0.789,0.651-0.741,0.512-0.650}. Pearson values are in outputs/train/v0-gradient-proxy-test/local_risk_sparse_64/summary.csv.

At eight of 64 points, variance ranking remains strong and direction ranking is moderate. At four points, the variance ranking remains useful but direction ranking is weaker. This is retrospective Gram-matrix subsampling and does not demonstrate runtime savings or interval-selection gains. The block windows overlap in their raw observations, and the track sample is unchanged.


## Local-Risk 128-, 256-, and 576-start sparse-sampling extension

Last modified: 2026-11-08

Question: In 128-, 256-, and 576-start blocks, how does correlation between sparse output-head variance/direction dispersion and full-gradient variance/direction dispersion change as proxy count decreases?

Preserve the sampling fractions from the 64-start experiment: 1.0, 0.75, 0.5, 0.25, 0.125, and 0.0625. Corresponding counts are 128/96/64/32/16/8, 256/192/128/64/32/16, and 576/432/288/144/72/36. For counts below the full block size, perform 20 fixed-seed nested uniform samples without replacement per block. Use all-point full and head Gram statistics as baselines.

Compute Pearson and Spearman correlations against full-gradient vector variance and direction dispersion. Report pooled-block correlations and track-level correlations separately. Use equal-track summaries only for tracks with at least three complete blocks. A 576-start block yields only 19 complete blocks overall and at most two per track; its pooled correlations are exploratory and cannot support an equal-track summary or strong inference.

A block is a run of consecutive window-start indices; separate blocks do not overlap in start index, but the underlying 576-point input windows can overlap in raw observations. This is retrospective Gram-matrix subsampling, not actual runtime measurement or interval-selection evaluation.

Entry: scripts/analyze_sparse_proxy_risk_long.py
Config: configs/rage/v0-gradient-proxy-local-risk-long.json
Output: outputs/train/v0-gradient-proxy-test/local_risk_sparse_long/
Log: logs/v0-gradient-proxy-test/local-risk-long/
Environment: rage Conda environment; tmux v0-gradient-proxy-local-risk-long; CPU-only saved-Gram analysis.


## Local-Risk long-block observed results

Completed 2026-11-08. Result category: Other. At 128-start blocks, k={128,96,64,32,16,8} gives equal-track median Spearman {0.872,0.869,0.860,0.845,0.840,0.809} for variance and {0.827,0.830,0.823,0.803,0.770,0.702} for direction. At 256-start blocks, k={256,192,128,64,32,16} gives nine-track median Spearman {0.900,0.900,0.889,0.883,0.867,0.867} for variance and {0.822,0.839,0.839,0.800,0.811,0.772} for direction. Nine tracks contribute because track 00 has only two complete 256-start blocks.

At 576-start blocks, k={576,432,288,144,72,36} gives pooled-block Spearman {0.949,0.941,0.946,0.941,0.921,0.892} for variance and {0.840,0.846,0.839,0.825,0.767,0.704} for direction. There are only 19 complete blocks total and at most two per track, so no within-track correlation is estimable; pooled results are exploratory and potentially confounded by track-level differences.

Pearson and 10th-90th sampling ranges are in outputs/train/v0-gradient-proxy-test/local_risk_sparse_long/summary.csv. These are retrospective Gram-matrix subsampling results, not runtime or selection-policy evaluations.


## Local-Risk target variance spread check

Completed 2026-11-08 by deduplicating full-gradient target statistics from the saved block outputs. After normalizing each block's full variance by its own track median, pooled p90/p10 is 7.44, 5.25, 4.42, 3.88, and 2.06 for block lengths 32, 64, 128, 256, and 576. Complete block counts are 396, 194, 93, 46, and 19. Median within-track CV is 0.810, 0.697, 0.505, 0.435, and 0.218; the final value uses tracks with two blocks and is unstable.

The observed targets are not concentrated at one common value for 32-256 starts. At 576 starts, only 19 pooled blocks are available, so the apparent lower spread is exploratory. This checks target discriminability, not proxy top-tail recall. Full quantiles and direction-dispersion ranges are in docs/results/v0-gradient-proxy-test.md.
