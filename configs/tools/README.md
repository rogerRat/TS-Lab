# Tool configuration

Tool commands read YAML files below `configs/`. Each file uses
`schema_version: 1`; paths may be project-relative or absolute. Repeated
`--set key=value` options override YAML values. An experiment manifest
references stage YAML files and can be run with
`python scripts/tools/run.py --config configs/experiments/<name>/run.yaml`.

## Training and warm-up

```yaml
schema_version: 1
run_id: example
model: chronos-tiny
model_path: cache/pretrained_models/amazon/chronos-t5-tiny
device: cuda
data:
  id: source-v1
  windows: data/source/windows.npz
  windows_key: windows
  context_length: 512
  prediction_length: 64
training:
  steps: 10
  batch_size: 4
  learning_rate: 0.0001
  seed: 0
  weight_decay: 0.01
  log_every: 1
  lora: null
output_dir: outputs/train/example
```

Use this config with `scripts/tools/warmup.py` for a short training check
(`warmup_max_steps` defaults to 1000), or `scripts/tools/train.py` for a
training run. Registered model module targets are validated before LoRA starts.

## Gradient cache

```yaml
schema_version: 1
model: chronos-tiny
model_path: cache/pretrained_models/amazon/chronos-t5-tiny
device: cuda
data:
  id: source-v1
  windows: data/source/windows.npz
gradient:
  kind: full_model
  batch_size: 0
  workers: 2
output_dir: outputs/grad_cache/source-v1
```

The archive must contain `windows[N, C+H]` and `window_ids[N, 2]`.

## Selection

```yaml
schema_version: 1
run_id: example
method: crb
model: chronos-tiny
model_path: cache/pretrained_models/amazon/chronos-t5-tiny
source_id: source-v1
reference_id: reference-v1
source_archive: data/source/fragments.npz
reference_archive: data/reference/fragments.npz
context_length: 512
prediction_length: 64
window_stride: 1
selection:
  n_select: 4096
  seed: 0
  ads:
    eta: 1.0
    q: 16
    top_ratio: 0.2
  cover_density: 0.02
  error_density: 0.02
  split_density: 0.005
  block_budget: 8
output_dir: outputs/selection/example
```

Fragment archives use concatenated one-dimensional `values`, zero-based
`offsets`, and optional aligned `fragment_ids` and `entity_ids`. The
selector reuses the shared CRB/ADS APIs and writes its selection artifact,
gradient cache and score matrix or score summaries under standard output roots.
`uni` is uniform sampling and has no score matrix.

## Cached-gradient replay

```yaml
schema_version: 1
run_id: replay-example
context_length: 480
prediction_length: 96
window_stride: 1
replay:
  q_values: [4, 8, 16, 32, 64, 128]
  chunk_rows: 4096
  cosine_ge: 0.9
  save_window_metrics: true
  plot: true
  stores:
    - label: trajectory-000
      cache_dir: outputs/grad_cache/<run>/trajectory-000
output_dir: outputs/selection/replay-example
```

Each store must already contain `gradient_sketches.npy`, `manifest.json` and a
`.complete` marker. Replay reads anchors from the memory-mapped cache and
rebuilds every remaining window by linear interpolation, so it never computes
gradients. `replay.q_values` sets equivalent anchor spacing; each spacing
allocates `K = max(2, ceil(N / q))` anchors and builds both a longest-gap
bisection layout and a budget-matched uniform layout.

## Evaluation


```yaml
schema_version: 1
evaluation_config: configs/eval/fev/default.yaml
checkpoint: outputs/train/example/checkpoint
output_dir: outputs/eval/example
resume: false
```

Evaluation mode can be selected in the evaluation YAML or overridden with
`--evaluation-mode single_window|rolling`.
