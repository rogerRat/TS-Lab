# FEV 9M Static Sampling Experiment: Development and Handoff Plan

> 历史归档：保留当时的设计、操作或实验证据，不作为当前执行指令。当前入口见 `docs/README.md`。

**Status:** Implementation plan; no training run is authorized by this document

**Date:** 2026-08-14

**Benchmark:** FEV
**Purpose:** Determine whether task-dependent OT behavior is caused by task-conditioned relevance, excessive concentration, or an incorrect interpretation of the OT candidate set.

## 1. Executive Summary

The experiment uses **one model architecture**: a Chronos T5-Efficient-Tiny model with approximately 9M trainable parameters after resizing the vocabulary to 4096 tokens. The architecture is held fixed across all methods and tasks. Every method is trained independently from scratch for 20,000 optimizer steps.

The experiment contains three task-specific reference conditions:

| Task index | Task name | Why it is included |
|---:|---|---|
| 17 | uk_covid_nation_1D | Historical OT was strongly negative relative to random sampling. |
| 18 | us_consumption_1Y | Historical OT was clearly positive. |
| 2 | boomlet_619 | A third task with a relatively large active reference population and a measurable positive historical OT effect. |

The three tasks must be run in **three independent run namespaces**. A single run with task_indices [2, 17, 18] is invalid because the current runner constructs one joint reference directory and the transfer-cache builder would pool all three reference distributions.

The core comparison has six method variants:

1. rand
2. static OT with neighbor_count=64
3. static OT with neighbor_count=256
4. static OT with neighbor_count=1024
5. candidate-uniform-k256, using the exact k=256 candidate union but assigning equal mass to every candidate
6. static OT with neighbor_count=256 mixed with the original source distribution using lambda=0.5

This gives **18 independent training instances** (3 tasks × 6 methods) using the same 9M architecture. Each instance produces six declared checkpoints at 1K, 2K, 4K, 8K, 16K, and 20K steps. The first four methods can be used as an implementation-gated minimum if candidate-uniform is not yet available; the full causal comparison should use all six.

The central comparison is:

~~~
random -> static OT                         Does task-conditioned mass help?
OT k256 -> candidate-uniform k256           Does OT mass allocation help beyond candidate selection?
OT k64 -> OT k256 -> OT k1024               How sensitive is the result to candidate support size?
OT k256 -> lambda=0.5 mixture                Is the OT effect a relevance/concentration trade-off?
~~~

## 2. Research Questions and Hypotheses

When OT improves one FEV task substantially but hurts another, test whether the difference is caused by:

- a useful task-conditioned source distribution;
- an overly narrow candidate support;
- an incorrect mass allocation inside a candidate support;
- excessive concentration and loss of source diversity;
- or an upstream reference/feature mismatch.

| ID | Hypothesis | Expected evidence |
|---|---|---|
| H1 | OT gains come primarily from selecting a relevant candidate support. | OT and candidate-uniform perform similarly at the same k; both differ from random. |
| H2 | OT gains come from assigning non-uniform mass within the support. | s-ot-k256 materially differs from candidate-uniform-k256. |
| H3 | OT is harmed by excessive concentration. | Smaller lambda, larger k, or higher ESS reduces negative transfer. |
| H4 | Candidate support size is the dominant control variable. | Performance changes systematically across k=64,256,1024. |
| H5 | The reference task is not consistently represented by selected source windows. | OT diagnostics improve while delivered local-window similarity does not. |
| H6 | Historical task selection explains the apparent heterogeneity. | Positive and negative cases reproduce their historical directions under the same protocol. |

This is an exploratory diagnostic. Task 2 was selected using historical results and is not an unbiased held-out confirmation task.

## 3. Fixed Model and Training Protocol

### 3.1 Architecture

The from-scratch architecture source is:

~~~
/root/TS3/cache/pretrained_models/google/t5-efficient-tiny/3441d7e8bf3f89841f366d39452b95200416e4a9/config.json
~~~

The intended model configuration is:

~~~yaml
initialization_mode: from_scratch
architecture_source: /root/TS3/cache/pretrained_models/google/t5-efficient-tiny/3441d7e8bf3f89841f366d39452b95200416e4a9
pretrained_weights: null
tie_embeddings: false
architecture_overrides: {}
~~~

Expected architecture:

| Parameter | Value |
|---|---:|
| d_model | 256 |
| d_ff | 1024 |
| encoder layers | 4 |
| decoder layers | 4 |
| attention heads | 4 |
| vocabulary after resize | 4096 |
| embedding tying | disabled |
| expected parameter count | approximately 9,443,072 |

pretrained_weights must remain null. from_scratch must be visible in the generated manifest and model-construction log. Do not substitute the existing 46M model or a pretrained checkpoint.

### 3.2 Shared settings

Every task-method instance must use identical settings:

| Setting | Required value |
|---|---:|
| context length | 512 |
| prediction length | 64 |
| training steps | 20,000 optimizer updates |
| seed | 0 for the primary comparison |
| initialization | independent from scratch for every task-method pair |
| optimizer, learning rate, scheduler | identical across methods |
| precision, compilation, data-loader settings | identical across methods |
| upload | disabled |

Use the validated 9M batch size and record the exact value in each manifest. Gradient accumulation should remain 1 unless the validated configuration requires another value. If the runner needs separate deterministic random streams for initialization and sampling, record the rule and apply it symmetrically.

### 3.3 Checkpoints and evaluation

Use exactly:

~~~yaml
training_steps: 20000
checkpoint_steps: [1000, 2000, 4000, 8000, 16000, 20000]
evaluation.checkpoint_steps: [1000, 2000, 4000, 8000, 16000, 20000]
upload_trained_models: false
~~~

Checkpoint and evaluation lists must match. Completion requires all six checkpoints, corresponding evaluation outputs, and a readable manifest; scheduler exit alone is not completion evidence.

Primary metric: FEV SQL, lower is better. Retain WQL and other available metrics as secondary diagnostics. Compare methods only at matched task and checkpoint:

~~~
relative improvement = (random SQL - method SQL) / random SQL
~~~

## 4. Task and Reference Isolation

### 4.1 Selected tasks

Use task 17, task 18, and task 2, but in separate runs:

~~~yaml
task_indices: [17]
task_indices: [18]
task_indices: [2]
~~~

The historical rationale is:

| Task | Historical observation |
|---|---|
| 17 | Static OT was substantially worse than random. |
| 18 | Static OT was materially better than random. |
| 2 | Static OT had a positive effect and a larger active reference population. |

These observations motivate the cases; they are not evidence from the new run.

### 4.2 Reference construction

For each task-specific run:

1. Use the common source catalog configs/catalog/src/1b-fev-3-segl512s512.yaml.
2. Use only the selected task directory under configs/catalog/ref/1b-fev-3-segl512s512/fev.
3. Generate a run-specific reference view containing exactly one task.
4. Record the task index in both train and evaluation configuration.
5. Confirm the transfer-cache manifest reports the same single reference task.

Never pass [2, 17, 18] to one task-conditioned run. That tests a pooled reference distribution rather than three task-conditioned distributions.

### 4.3 Run namespaces

Use:

~~~
run-9m-task-02
run-9m-task-17
run-9m-task-18
~~~

Recommended roots:

~~~text
outputs/run-9m-task-02/
outputs/run-9m-task-17/
outputs/run-9m-task-18/
configs/experiments/run-9m-task-02/
configs/experiments/run-9m-task-17/
configs/experiments/run-9m-task-18/
~~~

Source catalog identity must remain shared and immutable; output and generated-config roots must remain task-specific.

## 5. Method Matrix

| Method | Policy | Parameters | Purpose |
|---|---|---|---|
| rand | random | none | Baseline without task-conditioned transfer mass. |
| s-ot-k64 | ot | neighbor_count=64, lambda=1.0 | Narrow static OT support. |
| s-ot-k256 | ot | neighbor_count=256, lambda=1.0 | Main static OT condition. |
| s-ot-k1024 | ot | neighbor_count=1024, lambda=1.0 | Broad static OT support. |
| candidate-uniform-k256 | new explicit policy | exact OT k=256 candidate union | Isolate candidate selection from OT mass allocation. |
| s-ot-mix-k256-lambda05 | ot | neighbor_count=256, lambda=0.5 | Test whether partial OT preserves diversity. |

The first four methods are supported by the current code path subject to validation. candidate-uniform-k256 requires implementation before the formal six-method run.

The current lambda semantics are:

~~~
mixed_mass = (1 - lambda) * source_mass + lambda * ot_target_mass
~~~

Preserve normalization and zero-mass accounting.

### 5.1 Candidate-uniform contract

Candidate-uniform must reuse the existing OT candidate construction. For each reference task with neighbor_count=256:

1. Construct the same active-reference set as static OT.
2. Construct the same nearest-source candidate union as Reference2SourceSegmentOTPlanner.
3. Keep candidate IDs unchanged.
4. Assign equal positive mass to every candidate.
5. Assign zero mass to every non-candidate source segment.
6. Normalize mass to one.

Report candidate count, positive-mass source count, zero-mass source count, entropy, and ESS. The required ablation is:

~~~
same source catalog
same reference task
same candidate union
different mass rule

OT:                 mass determined by Sinkhorn coupling
candidate-uniform:  mass = 1 / number_of_candidates
~~~

If exact candidate-union reuse cannot be guaranteed, do not include this method in the formal causal table; label it exploratory.

### 5.2 Optional lambda sweep

After the core run passes, use the following only if concentration remains unresolved:

~~~text
s-ot-k256-lambda025
s-ot-k256-lambda050
s-ot-k256-lambda075
s-ot-k256-lambda100
~~~

lambda=1.0 is already represented by s-ot-k256. The optional sweep adds nine task-method instances across the three tasks, for 27 total including the six-method core.

## 6. Model and Instance Count

There is one architecture and multiple independently trained instances:

| Level | Count | Meaning |
|---|---:|---|
| Architecture specifications | 1 | Fixed approximately 9M T5-Efficient-Tiny configuration. |
| Task namespaces | 3 | Tasks 2, 17, and 18. |
| Core methods | 6 | Random, three OT supports, candidate-uniform, lambda mixture. |
| Independent training instances | 18 | 3 × 6; each starts from scratch. |
| Checkpoints per instance | 6 | 1K, 2K, 4K, 8K, 16K, 20K. |
| Core checkpoint evaluations | 108 | 18 × 6, excluding repeated seeds. |

Reports must distinguish architecture count from trained-instance count. The full experiment trains 18 instances from one shared architecture.

## 7. Required Code Changes

### 7.1 Candidate-uniform implementation

Add the smallest isolated implementation that exposes candidate-uniform through the existing sampling configuration. It must:

- validate the new policy name;
- reuse existing source/reference feature spaces;
- reuse current OT candidate-union construction;
- avoid changing random, ot, kde, d-ot, or d-kde behavior;
- preserve normalized masses and zero-mass accounting;
- write candidate count, active-reference count, positive-mass count, entropy, and ESS;
- include policy and method names in transfer-cache manifests.

Preserve unrelated dirty worktree changes, especially in configs/logger.py, configs/scripts/run.template.yaml, configs/scripts/run.yaml, docs/Progresss.md, evaluate/fev.py, scripts/run.py, tests/test_logging_heartbeat.py, train/runner.py, visualize/loss_step_line.py, and visualize/perf_step_line.py. Use narrow patches and a separate experiment configuration.

### 7.2 Runner task isolation

The current runner creates a joint reference view for the requested task_indices. Use either three independent executable YAMLs with one task each, or a safe wrapper that invokes the runner three times with distinct run IDs and one positional task index each. Inspect generated train and evaluation YAMLs before training.

### 7.3 New 9M configuration

Add a separate executable configuration under configs/experiments or an equivalent documented location. Do not repurpose the current configs/scripts/run.yaml because it is a 46M, 100K-step configuration with uploads enabled.

The new configuration must explicitly set the model, schedule, evaluation, and upload values shown in Sections 3.1 and 3.3. Disable upload because the current upload path can remove local checkpoints after successful upload.

## 8. Implementation and Validation Sequence

### Phase A: Code and tests

1. Inspect the sampling schema, transfer-cache builder, planner, and related tests.
2. Add candidate-uniform with exact candidate-union reuse.
3. Add a fixture test verifying equal mass on a known candidate set.
4. Add diagnostics tests for normalization, candidate count, zero-mass count, entropy, and ESS.
5. Add the new 9M configuration without changing the existing 46M configuration.

### Phase B: Static validation

From the remote project root in the ts3 environment:

~~~text
compile or import-check every touched Python module
run focused transfer-policy tests
run runner dry-run for task 2
run runner dry-run for task 17
run runner dry-run for task 18
inspect every generated train/evaluation YAML
run git diff --check
~~~

Dry-run must prove:

- exactly one reference task per generated task config;
- all six methods and intended parameters are generated;
- checkpoint lists equal [1000, 2000, 4000, 8000, 16000, 20000];
- model is from scratch and uses the 9M source;
- uploads are disabled;
- train and evaluation task indices match;
- output roots are unique.

No GPU is required for this code-side phase. GPU matters only after config and cache logic pass.

### Phase C: Transfer-cache audit

Before training, record for each task-method pair:

| Diagnostic | Purpose |
|---|---|
| active reference segment count | Confirm task isolation and scale. |
| candidate segment count | Compare k=64, k=256, k=1024, and candidate-uniform. |
| positive-mass source count | Detect support collapse. |
| zero-mass source count | Measure source removal. |
| effective sample size | Quantify concentration. |
| entropy/perplexity | Complement ESS. |
| maximum mass and top-k mass | Detect repeated exposure. |
| OT convergence and marginal error | Reject invalid caches. |

The audit must prove s-ot-k256 and candidate-uniform-k256 use identical candidate IDs.

### Phase D: Training and evaluation

Launch training only after the previous phases pass. Run each task namespace independently, optionally in separate tmux sessions. Do not merge logs or output directories.

For every method, verify:

- intended run ID;
- manifest with expected Git revision and full YAML;
- six local checkpoints;
- one evaluation summary per checkpoint;
- expected task index in summary.csv;
- no upload or cleanup removed checkpoints;
- method and sampling parameters visible in audit output.

## 9. Analysis Plan

### 9.1 Tables

For each task, produce a matched checkpoint table with:

~~~text
SQL at 1K, 2K, 4K, 8K, 16K, 20K
WQL at the same checkpoints
relative SQL improvement versus rand
~~~

Produce a second table with candidate count, positive-mass count, zero-mass count, ESS, entropy, and top-10/top-100 mass.

### 9.2 Plots

For each task, plot:

1. SQL versus training step for all methods.
2. WQL versus training step if available.
3. Relative SQL improvement versus random.
4. ESS and entropy by method.
5. Candidate count and positive-mass count by method.

The six checkpoints are predeclared. Do not select a “best checkpoint” after inspecting the complete test curve without labeling the selection rule. Report validation-selected results separately if used.

### 9.3 Interpretation

| Observation | Supported interpretation |
|---|---|
| OT and candidate-uniform are close and both beat random | Candidate selection matters more than mass allocation. |
| OT beats candidate-uniform at the same k | Non-uniform OT mass is useful. |
| Candidate-uniform beats OT | OT may over-concentrate or assign harmful mass. |
| k=1024 recovers performance relative to k=64 | Narrow support is likely harmful. |
| lambda=0.5 improves over lambda=1.0 | Relevance/diversity trade-off is likely. |
| All task-specific methods lose | Reference/feature direction may be wrong or transfer is not useful. |
| Cache diagnostics differ but curves do not | Sampling differences may not reach delivered windows or gradients. |

No single task proves a mechanism. Interpret the pattern jointly across all three tasks and all six checkpoints.

## 10. Artifact and Reproducibility Contract

Each task namespace must preserve:

~~~text
manifest.json
exact input experiment YAML
generated train YAMLs
generated evaluation YAMLs
reference-task metadata
transfer-cache manifests and diagnostics
training logs
losses.csv
all six local checkpoints per method
evaluation summary.csv files
plots and analysis tables
~~~

The manifest must record Git commit ID, run ID, catalog run ID, task index and name, model source and initialization mode, method and sampling parameters, seed, checkpoint lists, source/reference paths, upload setting, and exact generated configuration text.

catalog_run_id is immutable asset identity. A new run may have a new run_id, but must not silently replace source/reference assets.

## 11. Handoff Checklist

### Before code changes

- [ ] Remote filesystem is mounted; project commands run on the remote checkout.
- [ ] Existing dirty worktree changes are recorded and preserved.
- [ ] Current 46M configuration is not modified.
- [ ] Task 2, 17, and 18 reference directories exist.
- [ ] The 9M architecture source is readable on the remote host.

### Before training

- [ ] Candidate-uniform is complete, or the run is explicitly labeled the five-method minimum.
- [ ] Focused policy tests pass.
- [ ] Three independent dry-runs pass.
- [ ] Generated configs contain one reference task each.
- [ ] training_steps is 20000.
- [ ] Checkpoint/evaluation steps are [1000, 2000, 4000, 8000, 16000, 20000].
- [ ] initialization_mode is from_scratch.
- [ ] pretrained_weights is null.
- [ ] tie_embeddings is false.
- [ ] upload_trained_models is false.
- [ ] Output/config roots are unique per task.
- [ ] Candidate-uniform and OT k=256 candidate IDs are identical.

### After training

- [ ] All 18 core instances have six checkpoints or a documented failure.
- [ ] All 108 expected checkpoint evaluations are present for a complete run.
- [ ] SQL and secondary metrics are matched by task and checkpoint.
- [ ] Concentration diagnostics are included.
- [ ] No local checkpoint was removed by upload.
- [ ] Results disclose exploratory scope and task-selection bias.

## 12. Expected Deliverable

The handoff agent should deliver:

1. A narrow candidate-uniform implementation and focused tests.
2. Three independent executable 9M experiment configurations or an equivalent safe launcher.
3. Passing dry-run and schema-validation evidence.
4. Reproducible training commands for each task namespace, without starting training unless explicitly requested.
5. Tables and plots covering all six checkpoints and all core methods.
6. A short conclusion separating candidate-support, mass-allocation, and concentration effects.

The final scientific claim must be limited to what this controlled three-task experiment supports. Its purpose is to identify the mechanism behind task-dependent OT behavior, not to establish a universal OT ranking across all FEV tasks.
