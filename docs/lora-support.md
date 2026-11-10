# LoRA Training Support

Last modified: 2026-11-10

## Scope

The optional LoRA path supports the registered trainable model batches below:

| Student | Model family | Default target modules | Notes |
|---|---|---|---|
| chronos-tiny | Chronos-T5 Tiny | q, v | Attention adapters only; tied input/output embeddings remain frozen |
| chronos-small | Chronos-T5 Small | q, v | Historical student alias; shares the Chronos-T5 training path |
| chronos-base | Chronos-T5 Base | q, v | Attention adapters only |
| timerxl-84m | TimerXL Base 84M | q_proj, v_proj | Uses the native TimerXL loss and forecast adapter |
| timesfm-2.5 | TimesFM 2.5 200M | q_proj, v_proj | Uses the native TimesFM training loss |

TimesFM 3.0 and Sundial remain inference-only or otherwise non-deployable in the student-training registry. They are not included in this LoRA training scope.

## Configuration

LoRA is opt-in through the model worker manifest's training object. Existing manifests without lora continue to use their existing full-parameter training path.

Example manifest fragment:

    {
      "training": {
        "lora": {
          "r": 8,
          "alpha": 16,
          "dropout": 0.0
        }
      }
    }

Defaults are rank 8, alpha 16, and dropout 0.0. target_modules may be supplied as a list of module suffixes for a controlled variant. The selected suffixes must exist in that student's model.

The implementation uses PEFT LoRA with bias="none" and task_type=None. Backbone parameters are frozen by PEFT; only adapter parameters are optimized and gradient-clipped. The original model loss functions stay in use. For adapter-backed models, the adapter model reference is updated to the PEFT wrapper so the native loss path uses the trainable modules.

Each student worker creates its baseline state after LoRA injection. PEFT's default zero-initialized B matrix keeps the initial forward function unchanged. Checkpoint state includes the wrapped model state in the existing format. The resume contract records the normalized LoRA settings, so a rank, target, or dropout change cannot silently reuse an incompatible optimizer checkpoint.

## Output-head Gradient Cache

LoRA training and output-head gradient caching are independent features. The gradient-cache runner remains on the pretrained checkpoint path and does not automatically load student LoRA checkpoints. No LoRA adapter is enabled for existing experiments unless a worker manifest explicitly includes training.lora.

## Dependencies and Validation Status

PEFT 0.18.1 is pinned in the Chronos, TimerXL, and TimesFM model requirement profiles. The profile installer checks the PEFT import in the corresponding runtime environments.

This change was statically inspected and syntax-checked. No model checkpoints were loaded, no gradient or optimizer step was run, and no LoRA behavior or quality comparison has been measured yet.
