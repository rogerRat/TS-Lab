# Foundation-model and external-baseline layering

## Non-negotiable experiment boundary

The RAGS experiment is a static data-selection experiment. Every method receives the
same immutable feasible source pool and target safe-history reference, and returns only
window identities already present in the shared `WindowIndex`. Selection never changes the
candidate pool, training schedule, configured students, or FEV 512/64 examination
protocol. `run_baseline` checks returned identities and writes a machine-readable fidelity
contract into every external-baseline result.

## Three layers

1. Bottom components contain immutable specifications and pure operations. Examples are
   `adapters.foundation.spec`, `adapters.foundation.components`, and
   `baselines.contracts`. They perform no orchestration and no downloads.
2. Middle adapters own optional upstream imports and model-specific behavior. Examples are
   `TimesFM3Adapter`, `TimesFM25Adapter`, `TimeMoEAdapter`, and `SundialAdapter`. Loading is lazy and uses
   the checkpoint revision from `requ/model_integrations.lock`.
3. Top entry points parse CLI arguments and orchestrate middle-layer calls. Model inspection
   and smoke prediction live in `scripts/exp/model_probe.py`; experiment CLIs remain under
   `scripts/exp/`. `scripts/exp/model_train.py` consumes a shared task manifest and delegates
   the model-specific native loss and shared epoch/checkpoint loop to `rags.experiments.students`.

No top-level CLI logic may be imported into a component or adapter. No model adapter may
read experiment datasets directly. It receives already-authorized, already-materialized
arrays from the static framework.

Capability and weight smoke checks are explicit CLI actions. `--smoke-forecast` may run for
all registered models; `--smoke-loss` and `--smoke-gradient` are accepted only when the
selected model declares the respective native capabilities. Unsupported training paths fail
before any weight download.

## Current experiment status (2026-09-14)

The current students are Chronos T5 Base, TimesFM 2.5 and TimeMoE-200M.
Model parameters, gradients and AdamW moments follow the BF16 policy documented in
[gradient assets](../gradient_assets.md). Installation is maintained in
[requ/INSTALL.md](../../requ/INSTALL.md), with the root `_install.sh` entry point.
The current parameter/task probe and its execution checkout are described in
[ParameterTaskProbing](ParameterTaskProbing.md) and the
[execution record](../experiments/probe_execution_20260914.md).
The [0911 three-student handoff](../archive/dev/MultiStudent0911.md) preserves the
older Small/FP32 milestone; it is not the current deployment configuration.

## Historical TimesFM and Sundial status (2026-09-10)

- TimesFM 3.0 is the latest release and is exposed for forecasting. Its official PyTorch
  model calls itself inference-only, so TS3 does not claim gradient or fine-tuning support.
  Its weights have a non-commercial, non-production license.
- TimesFM 2.5 is retained as the trainable TimesFM experiment model. The official
  Transformers forward path accepts `past_values` and `future_values` and returns a native
  loss. It supports the long-context W2048/S2048 valuation design without changing the
  downstream student or examination protocol.
- Sundial Base 128M is exposed for probabilistic forecasting up to its documented stable
  context length of 2,880. Although checkpoint remote
  code contains a TimeFlow loss implementation, the official repository has not released a
  fine-tuning recipe. TS3 keeps training disabled until that contract is published or an
  explicitly reviewed paper-faithful training protocol is approved.

TimesFM and Sundial use `requ/timesfm-requirements.txt` and `requ/sundial-requirements.txt`
respectively. They must remain isolated from the Chronos environment: the pinned TimesFM
2.5 checkpoint declares Transformers 5.3 development code (the environment uses the matching
5.3.0 release), while Sundial explicitly requires
Transformers 4.40.1. The adapter uses FP32 on the installed RTX 2080 Ti because that GPU does
not provide native BF16 execution.

Install or refresh all five environments with `bash _install.sh`.
Use `bash _install.sh --verify-only` to check installed dependencies, imports and CUDA.
Single-environment installation and storage options are documented in
[`requ/INSTALL.md`](../../requ/INSTALL.md). Override `PYPI_INDEX_URL` or
`PYTORCH_INDEX_URL` when another mirror is required.

Set `HF_ENDPOINT=https://hf-mirror.com` if direct access to
`huggingface.co` times out from the development container. On 2026-09-10 the pinned models
passed real GPU smoke checks: TimesFM 3.0 and Sundial each produced finite `[1, 16]`
forecasts, and TimesFM 2.5 produced a finite native training loss for a 128/16 window.

The long-context valuation probe uses `FoundationGradientEngine` with TimesFM 2.5's
`output_projection_point.output_layer.weight` and the existing seeded
`ParameterHashProjector`. Using this fixed 1.64M-parameter layer avoids allocating hash and
sign tables for all 231M parameters. `shifted_windows` materializes only caller-specified
prediction events with their preceding context; `grouped_mean_sketches` derives S2048 from
the exact W2048 event sketches without additional backward passes. The selected layer,
projection seed, model revision, context, and aggregation mode must be recorded per arm.
The exact 2048/64 path was exercised on the development RTX 2080 Ti with a finite native
loss and a finite non-zero 128-dimensional gradient sketch.

`scripts/exp/model_select.py` is the deployment bridge for the appendix 2x2 experiment. It
accepts fixed source long-context windows, immutable downstream `[fragment_id, start]`
identities, segment labels, reference windows, and reference segment labels. `window` mode
is W2048; `segment` mode is S2048 (mean gradient per source and reference segment, with each
source segment score broadcast to its events). Both modes pass the
same identities through the same non-overlapping 576-point deployment and record selected
IDs plus protocol metadata. The valuation context changes without changing the frozen
Chronos-small student or its 512/64 exam.

The training entry point now consumes an explicit task manifest with the shared 512/64
selected windows and four-epoch schedule. TimesFM 3.0 and Sundial are not experiment students. Example:

```bash
conda run -n ts3-timesfm python -m scripts.exp.model_train \
  --task-manifest outputs/<run-id>/main/tasks/ETT_1H.json \
  --output outputs/<run-id>/main/students/timesfm-2.5
```

## Baseline audit result

The four baselines are not equally faithful:

- LESS is a trajectory-level core-signal adaptation (decision 11): uniformly sampled
  window gradients are averaged before trajectory cosine matching. Bilateral SGD,
  a single pretrained checkpoint and full-parameter hash projection are deliberate
  decisions, not unfinished Adam/warm-up/LoRA work.
- TSDS and LP-Mirror are algorithm ports with explicit window/model adaptations.
  Their probability-assignment and loss-drop kernels have locked-source parity tests.
  TSDS now uses trajectory feature means and upstream FAISS IVF, preserving multinomial
  copies through actual training. See `PointBudgetDelivery0912.md` for whole-copy
  point-budget delivery and uniform legal-start training sampling.
- LTSV is a paper-equation port. On 2026-09-11 its coverage-weighted aggregation was
  corrected to Eqs. 10--11 (point means, then window means), its optimizer changed to
  the paper's Adam, and reference scoring changed to context-only forecast MSE.
  Native Chronos CE remains the block-training model adapter. The official repository HEAD
  stores runnable trees as gitlinks without `.gitmodules` and deletes the original valuation
  driver. Its historical one-epoch setting alone cannot establish a conflict with one step.
  LP-Mirror now uses full source/target epochs; both time-series baselines preserve native
  top-k, including overlap, rather than the shared non-overlap filter. Dataset/ref changes
  are experimental settings, not reasons to alter the valuation algorithms.

These facts are encoded in `baseline_contracts.py` and must be reported with experimental
results. Pure locked-source kernels live in `baseline_algorithms.py`; model/data extraction
and static deployment remain separate middle-layer responsibilities.
