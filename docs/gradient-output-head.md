# Output-head gradient cache

Last modified: 2026-11-09

## Scope

The cache runner accepts \`--gradient-kind output_head\` for the trainable models
registered in this project:

| Model key | Current checkpoint family | Output-head calculation |
|---|---|---|
| \`chronos-tiny\` | Chronos-T5 Tiny, 8M | Analytic CE derivative at the linear \`lm_head\` |
| \`chronos-small\` | Chronos-T5 Small, 46M | Analytic CE derivative at the linear \`lm_head\` |
| \`chronos-base\` | Chronos-T5 Base, 200M | Analytic CE derivative at the linear \`lm_head\` |
| \`timerxl-84m\` | Timer-XL / Timer Base, 84M | Analytic MSE derivative for every linear patch output head |
| \`timesfm-2.5\` | TimesFM 2.5, 200M | Head-only autograd for its nonlinear point and quantile projection blocks |

TimesFM 3.0 is inference-only in the project registry and has no training-loss
or gradient cache contract. No other Chronos or Timer-XL sizes are registered.

## Gradient semantics

Chronos-T5 uses the checkpoint's per-window mean cross-entropy over nonignored
labels. The probe captures the actual input to \`lm_head\`, including any
checkpoint scaling, then computes \`(P-Y)/N\` and its outer product with the
captured hidden state. It does not call autograd. For tied T5 embeddings, this
keeps only the output-path contribution to the shared table; the encoder and
decoder embedding-path contributions are intentionally omitted.

There are two existing ChronosTiny call paths in the project. The core
\`ChronosTinyGradientEngine\` defaults to \`lm_head.weight\` and already uses an
analytic \`torch.no_grad\` fast path. The \`v0-grad-mech-test\` script instead
loads the same warmed tiny checkpoint through \`ChronosT5Engine\` and its
\`gradient_sketch\` calls \`torch.autograd.grad\` over all trainable parameters.
That experiment's retained vectors are full-model gradients, including the
combined output and embedding paths of the tied shared weight. The generic
\`full_model\` cache mode also uses the full-model route.

Timer-XL uses the same shifted-patch MSE reduction as its native adapter. It
computes each patch head's weight and optional bias derivative from its
prediction residual and hidden state, without autograd.

TimesFM 2.5 uses two nonlinear residual projection blocks, one for point
predictions and one for quantile predictions, followed by a composite MSE and
quantile loss. It does not fit the single-linear-layer outer-product formula.
Before its forward pass, the probe freezes every backbone parameter and enables
gradients only for \`output_projection_point\` and
\`output_projection_quantiles\`. Autograd is requested only for those output
projection parameters. This computes the complete selected-head gradient
without requesting or accumulating Transformer parameter gradients. It is a
head-only reverse-mode calculation, not a forward-only analytic calculation.

All three outputs are deterministic sketches of their selected head parameter
gradients with the configured projection dimension and seed. The BF16 cache
manifest labels them \`projected_output_head_gradient\` and records the selected
parameter names, keeping them separate from existing full-model gradient
caches. These proxies must not be interpreted as full shared-parameter
gradients or complete model updates.

## Use

Example command shape:

\`\`\`sh
PYTHONPATH=src python -m assets.gradient.runner \\
  --model chronos-tiny \\
  --gradient-kind output_head \\
  --windows data/windows.npz \\
  --run-dir outputs/gradient-cache/chronos-tiny-head \\
  --dataset dataset-version
\`\`\`

The model-specific Conda environment, checkpoint path, window geometry, and
data identity must match the selected model and input file. Outputs are written
under the supplied run directory, which must be inside \`outputs/\`.

## Validation status

The cache path and model adapters are implemented. No model forward or
numerical comparison was run during this code change. Before using these caches
for selection, validate sampled windows against detached-head autograd for
Chronos and Timer-XL, and against a reference gradient restricted to the two
TimesFM projection blocks. Also compare cached loss values with each model's
existing native per-window loss. The GPU currently has an unrelated active
tmux task, so this change did not start a validation job.

