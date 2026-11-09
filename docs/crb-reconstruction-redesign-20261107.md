# CRB reconstruction redesign: evidence, derivation, and testable hypotheses

Last modified: 2026-11-07

Status: Research proposal, not an executed replacement experiment. The running v0-crb-reconst-test protocol and queue are unchanged.

## Scope and current evidence

Question: Can fixed-budget queries reconstruct all window gradients better than uniform placement, and what information should drive query selection?
Sources: src/rage/crb.py; scripts/exp/v0_crb_reconst_test.py; scripts/exp/v0_grad_mech_test.py; current evaluation JSONL files; selected FP32 trajectory summaries.
Snapshot near 22:00 CST: trajectory 00 has all 16 candidates at q=4,16,64; L2 win counts against uniform are 1,10,0, and cosine win counts 0,5,2. Trajectory 01 has 15/16 candidates at q=4, with 7 L2 wins and zero cosine wins. These are configuration counts, not independent statistical replicates.
All ten BF16 caches exist. No new full-gradient replay or diagnostic computation was executed for this report.
Selected source ordinals: 973,980,981,984,985,987,988,996,998,999.
Across these ten trajectories the FP32 mean cosine ranges are 0.642-0.810 at lag 1, 0.453-0.631 at lag 4, 0.283-0.442 at lag 16, and 0.034-0.189 at lag 64.
Probe code calls model.eval(). Fast variation is not evidence of dropout noise, independent noise, high effective rank, or fundamental unpredictability.
The current comparison is equispaced uniform, not random. Cover-only CRB uses recursive longest-gap bisection with rightmost ties; it is not the same grid as the uniform baseline.

## Derivation: what the current risk assumes

For twice differentiable vector g(t) and approximately constant second derivative c over width h, integrated squared linear-interpolation error is approximately h^5 ||c||^2 / 120.
At an exact midpoint let d = g(mid) - (g(left)+g(right))/2. The implementation estimates ||c|| = 8||d||/h^2, hence risk = (8/15) h ||d||^2.
Crucially, child intervals inherit the parent curvature. Halving h predicts each child's risk to be 1/32 of the parent's and their total risk 1/16, only under the constant-curvature approximation. This scale transfer must be tested, not assumed.
For a contrasting model g(t)=m(t)+e(t), independent zero-mean residuals with E||e||^2=V give non-anchor interpolation residual MSE V[1+(1-a)^2+a^2], or 1.5V at the midpoint. Predicting m alone has MSE V. This is an illustrative conditional derivation, not a fitted description of the cached gradients.
If a residual d is multiplied by shrinkage lambda around a known mean, the population MSE-optimal lambda is E[(g-m)^T d]/E||d||^2, clipped to [0,1] if constrained. Mean estimation and coefficient estimation add error in practice.
One must measure residual dependence and predictive error before treating e as unlearnable.

## Candidate redesign

H1 (keep, first): Separate anchor placement from the reconstruction rule.
Fix exactly the same anchors and compare nearest-neighbor, linear interpolation, anchor mean, and shrinkage toward the anchor mean. Return the exact observed gradient at queried anchors. Fit shrinkage with paid anchor validation or training trajectories, never the full target trajectory truth.
Failure: no held-out trajectory benefit over linear interpolation. Improvement in L2 without cosine improvement is a partial result.

H2 (keep after H1): Replace forced interpolation with regularized multi-anchor regression.
Use g_hat(t)=m + k(t,A)[K(A,A)+lambda I]^-1(G_A-1m^T), initially with a rough Matern/exponential time kernel, then a time-plus-input-feature kernel if justified. The same scalar weights combine full 8,394,496-dimensional anchor vectors; dimensionality reduction is unnecessary.
For nonqueried windows regularization models unresolved variation; queried windows retain exact values. Kernel ridge regression and GP regression are existing methods, not a novelty claim.
Input statistics and token summaries may be cheap covariates, but feature extraction and any forward passes must be costed. Chronos tokenization/normalization is a candidate cause to inspect, not an established cause here.
Failure: no benefit over an anchor-mean or time-only baseline on held-out trajectories, or excessive wall-clock cost.

H3 (revise, conditional): Select by expected reduction in total prediction error.
For a scalar covariance shared across outputs, a GP-style acquisition is Delta(i)=sum_j C_A(j,i)^2/[C_A(i,i)+tau_i^2] for reducing latent prediction variance. Distinguish latent variance from exact finite-pool gradient error: querying i also reveals its own residual, and off-diagonal covariance determines transfer to other windows.
Fixed kernels yield location/feature-driven selection; output adaptation requires paid hyperparameter/calibration updates. A stationary time kernel on a near-uniform domain may simply prefer space filling, which is acceptable.
Retain explicit coverage, compare pure greedy and a prespecified coverage mixture. Stop escalating complexity if predicted error reduction is uncalibrated.
Failure: predicted gains do not rank actual gains better than width-only coverage.

H4 (park for pointwise task; keep for a separate aggregate task): Surrogate plus residual correction.
For fixed surrogate m_i and U sampled uniformly from all N indices independently of surrogate fitting,
g_bar_hat = (1/N)sum_i m_i + (1/r)sum_{i in U}(g_i-m_i).
Its expectation is the true mean gradient. Repeated draws can reuse cached queries but must be counted transparently. If sampling only unqueried indices or without replacement, use the corresponding finite-population correction; do not silently use the all-N formula.
A residual importance distribution can reduce variance; under independent with-replacement draws the trace-variance optimum is p_i proportional to ||g_i-m_i||, but true residuals are unavailable before querying and cannot be used by the deployable method.
This estimates the aggregate, not every window. It cannot automatically solve cosine ranking or top-k retrieval.

## Diagnostic and replay plan (proposed, not started)

D0: Freeze metadata and result snapshot. Verify exact query accounting, anchor sets, source precision, and all methods' information access.
D1: Use full-coordinate cached vectors to measure finite differences, midpoint residual scaling, parent-to-child error transfer, and kappa versus actual gain. Report per trajectory and gap width; compare ranking to width-only control. A greedy full-truth comparator is an oracle heuristic, not a proven global optimum or deployable algorithm.
D2: Same-anchor reconstruction comparison (H1 then H2) with equispaced, random-without-replacement, stratified-random, recursive-cover, and frozen-CRB placement. Keep mandatory endpoints and exact unique budget K matched.
D3: Only if H2 predicts better, compare adaptive acquisition H3 using the same estimator and budget.
D4: Aggregate residual correction H4 only under separately declared aggregate metrics.
Use q=4,16,64 for a small development stage; retain original eight-q results as the original experiment, not overwritten. Proposal seed set for random baselines: 0..19, fixed before execution. For K=2 only endpoints exist; omit impossible tuning stages.
Use outer leave-one-trajectory-out, with all hyperparameter selection confined to the other nine and any in-trajectory tuning charged to K. Product-group holdout sensitivity is necessary because several selected trajectories share FOODS_1_101/102 items. These ten already inform hypothesis design, so even outer validation is exploratory; fresh trajectories/checkpoints are required for confirmation.
Metrics: original mean L2 and cosine on all windows; non-anchor diagnostics; common never-queried evaluation mask as sensitivity, recognizing mask changes across methods; global relative RMSE sqrt(sum||err||^2/sum||g||^2); norm error; lower cosine and upper L2 tails; wall time and unique backward-equivalent queries.
Do not treat overlapping windows or sixteen configurations as independent experiments. Report trajectory-level paired effects and random-seed variability; small-n uncertainty remains substantial. Do not choose the best configuration independently on each test trajectory.
A useful success gate is positive paired effects across trajectories without systematic cosine/tail degradation, plus a prespecified practical improvement threshold chosen before implementation. No numeric threshold has been authorized or validated yet.

## Prior work and transfer limits

1. Rasmussen and Williams (2006), Gaussian Processes for Machine Learning, chapter 2.
https://gaussianprocess.org/gpml/chapters/RW2.pdf
Inspected chapter text. Supports regularized prediction and predictive uncertainty; does not establish a correct kernel for these gradients.

2. Yu, Bi, and Tresp (2006), Active Learning via Transductive Experimental Design.
https://doi.org/10.1145/1143844.1143980
Author full text: https://www.dbs.ifi.lmu.de/~tresp/papers/icml2006_actv.pdf
Inspected method framing. Select points useful for the whole unobserved pool, not merely outliers. Transfer to vector gradients is our proposal.

3. Oates and Girolami (2016), Control Functionals for Quasi-Monte Carlo Integration.
https://proceedings.mlr.press/v51/oates16.html
Inspected full text, including approximation-plus-residual construction. Smoothness-based convergence guarantees do not automatically apply here.

4. Katharopoulos and Fleuret (2018), Not All Samples Are Created Equal: Deep Learning with Importance Sampling.
https://proceedings.mlr.press/v80/katharopoulos18a.html
Inspected full text excerpts. Supports cost-aware gradient sampling and variance-based activation. Large gradient norm is not synonymous with interpolation error.

5. Mirzasoleiman, Bilmes, and Leskovec (2020), Coresets for Data-efficient Training of Machine Learning Models.
https://proceedings.mlr.press/v119/mirzasoleiman20a.html
Abstract-level assessment. Weighted subset approximates aggregate gradient, not all individual gradients. Oracle full-gradient access must not be free in our comparison.

6. Killamsetty et al. (2021), GRAD-MATCH: Gradient Matching based Data Subset Selection for Efficient Deep Model Training.
https://proceedings.mlr.press/v139/killamsetty21a.html
Inspected abstract/introduction and objective framing. Supports matching training/validation aggregate gradients. Does not validate pointwise interpolation.

7. Farquhar, Gal, and Rainforth (2021), On Statistical Bias In Active Learning: How and When To Fix It.
https://arxiv.org/abs/2101.11665
Abstract-level assessment. Motivates explicit sampling distributions and bias accounting; does not imply any inverse-probability correction is valid after arbitrary adaptive sampling.

8. Ankenman, Nelson, and Staum (2010), Stochastic Kriging for Simulation Metamodeling.
https://doi.org/10.1287/opre.1090.0754
Publisher abstract assessed. Separates surface uncertainty and intrinsic simulation randomness; our deterministic evaluation-mode gradients do not establish the latter.

9. Bach (2013), Sharp analysis of low-rank kernel matrix approximations.
https://proceedings.mlr.press/v30/Bach13
Abstract-level assessment. Computational option only if effective dimension supports it; no evidence yet that our gradient matrix is low rank.

## Search coverage and limitations

Search date: 2026-11-07. Targeted narrative search via web search over primary PMLR, NeurIPS, arXiv, publisher, and author-hosted sources; not direct exhaustive database exports, no PRISMA completeness claim. No date exclusion. Include regression, finite-pool sampling, gradient subsets, and variance reduction; exclude unrelated gradient inversion/privacy and parameter finite-difference optimization.
Query groups: "gradient estimation active sampling Gaussian process gradients data selection"; "CRAIG gradient matching coresets GradMatch"; "control functionals Monte Carlo integration Oates Girolami Chopin"; "Active Learning with Statistical Models"; "Gaussian processes machine learning regression noise covariance"; "Katharopoulos Fleuret importance sampling"; "stochastic kriging simulation metamodeling"; "active learning importance weighted unbiased Farquhar Gal Rainforth"; "gradient prediction data selection gradient estimation kernel 2024 2025"; "active learning regression transductive experimental design"; "gradient estimation surrogate control variates stochastic optimization".
Search hits include duplicate reports. Full text was not assessed for every source; coverage and novelty claims remain limited. Recent broad queries did not establish a newer direct solution to this exact problem.

## Prioritization rubric

Scores 1-5; higher risk means greater risk. Judgments are provisional, not measured outcomes. Robotics relevance is not applicable.
| Candidate | Novelty | Feasibility | Evaluability | Publishability alone | Risk | Decision |
| H1 shrinkage | 1 | 5 | 5 | 1 | 2 | Keep as diagnostic baseline |
| H2 regularized regression | 2 | 4 | 5 | 2 | 3 | Keep as first substantive candidate |
| H3 expected error reduction | 2 | 3 | 4 | 2 | 4 | Revise after calibration evidence |
| H4 residual correction | 1 | 4 | 5 | 2 | 3 | Separate aggregate branch |
A publication claim requires demonstrated benefit, faithful costs, multiple datasets/checkpoints, and a contribution beyond combining known methods.

## Workflow attribution

Literature-review and research-idea-rubric skills structured this assessment. Scientific Agent Skills workflow reference: Kassis, Agarwal, He, Patel, and Brueckner (2026), Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents, https://arxiv.org/abs/2609.00065 (current record verified). This is workflow attribution, not evidence for the CRB proposals.

