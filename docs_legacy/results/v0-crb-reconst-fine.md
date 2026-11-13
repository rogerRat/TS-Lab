# v0-crb-reconst-fine results

Last modified: 2026-11-10

Status: completed. The fixed random seed 20261110 selected M5 manifest ordinal 771, source row 772, series ID FOODS_1_079_CA_3. Its 1,643 values give 1,068 legal 512-context plus 64-prediction windows at stride 1. The model is the pinned ChronosTiny 100-step warmed checkpoint with a freshly initialized, untrained PEFT LoRA adapter (r=8, alpha=16, dropout=0, q/v targets, model seed 0). All 98,304 trainable LoRA coordinates were included in each per-window gradient. BF16 model forward/backward produced FP32 stored gradient vectors. The model state was fixed across all 1,068 windows.

For each q, K=max(2,ceil(1068/q)) is the exact number of gradient anchors for both methods, with both endpoints counted. CRB divides the K-2 remaining calls as evenly as possible across coverage, error-risk, and recursive minimum-kappa. Uniform uses K evenly spaced anchors. Both reconstruct by linear interpolation of anchor vectors. The dense gradient sweep supplies ground truth and retrospective anchor queries; its cost is excluded from the simulated query budget.

| q | K | Uniform mean L2 | CRB mean L2 | CRB change | Uniform mean cosine | CRB mean cosine |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1068 | 0.000000 | 0.000000 | 0.0% | 1.0000 | 1.0000 |
| 2 | 534 | 0.001450 | 0.000497 | -65.7% | 0.9872 | 0.9988 |
| 4 | 267 | 0.002694 | 0.001804 | -33.0% | 0.9744 | 0.9900 |
| 8 | 134 | 0.004337 | 0.003893 | -10.2% | 0.9447 | 0.9630 |
| 26 | 42 | 0.008526 | 0.008667 | +1.7% | 0.8384 | 0.8174 |
| 32 | 34 | 0.008299 | 0.009292 | +12.0% | 0.8146 | 0.7676 |
| 64 | 17 | 0.011536 | 0.012020 | +4.2% | 0.6420 | 0.6573 |
| 128 | 9 | 0.016557 | 0.016180 | -2.3% | 0.4991 | 0.5131 |

Negative CRB change is lower L2 error. CRB wins primary mean L2 at q=2,4,8,128; Uniform wins at q=26,32,64; q=1 is an exact identity. At q=64, CRB has higher mean cosine despite worse mean L2. The raw results also include median L2, p95 L2, exact anchor indices, and CRB stage counts.

The result differs from the previous v0-crb-reconst-test in four material ways. That experiment reconstructed gradients of all 8,394,496 base-model parameters, used ten preselected late-manifest M5 trajectories, tested q={1,4,16,64,128,256,512,576}, and searched 16 CRB stage ratios per budget. It ended with three trajectories fully replayed, two partial, and five unstarted. The equal-stage prior configurations at q=64 on completed trajectories 00-02 all had worse mean L2 and mean cosine than matched Uniform. The new run uses one independently selected trajectory, LoRA parameter gradients, and a single fixed equal-stage policy. Absolute L2 magnitudes are not comparable across these gradient spaces or tracks. The v0-crb-var finding of lower within-block relative variance and direction dispersion at q about 64 does not imply lower pointwise interpolation error; this run directly measures the latter.

Validation: the dense cache has shape (1068,98304), finite values, and complete markers. All 16 result rows have unique in-range anchors, both endpoints, and exact matched K. Four gradients (rows 0, 128, 224, 1067) were recomputed from the fixed model and matched the cache exactly. A brief overlapping diagnostic process touched the cache while the tmux run was active; both used the identical fixed seed/model, and the sampled overlap rows matched exactly. At fresh standard LoRA initialization, 49,152 coordinates (the LoRA A matrices) have zero gradient because LoRA B starts at zero. This materially limits extrapolation to a trained LoRA state. One selected trajectory cannot establish an average effect across M5 trajectories or other model states.

Initial-track artifacts retained after cache release: `outputs/train/v0-crb-reconst-fine/results.json`, `selection.json`, `gradient_metadata.json`, `cache_released.json`, and `.complete`. Config: `configs/rage/v0-crb-reconst-fine.json`; entry: `scripts/v0_crb_reconst_fine.py`; log: `logs/v0-crb-reconst-fine/run.log_20261110170703.log`.

## Twelve-trajectory extension

Last modified: 2026-11-10

The first random trajectory was retained, then 11 additional distinct M5 manifest ordinals were drawn without replacement using seed 20261110. Frozen ordinals: 771, 835, 633, 220, 763, 706, 817, 619, 530, 522, 140, 64. Every trajectory used the same fixed ChronosTiny checkpoint, fresh model-seed-0 LoRA initialization, eight q values, equal-stage CRB policy, and matched anchor count. All 12 trajectories completed 16 method-budget rows each. The table weights trajectories equally; win counts compare mean L2 within each trajectory at the same q.

| q | Uniform mean L2 | CRB mean L2 | CRB change | CRB wins | Uniform wins | Uniform mean cosine | CRB mean cosine |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.000000 | 0.000000 | 0.0% | 0 | 0 | 1.0000 | 1.0000 |
| 2 | 0.026663 | 0.024898 | -6.6% | 10 | 2 | 0.9087 | 0.9072 |
| 4 | 0.047243 | 0.048444 | +2.5% | 2 | 10 | 0.8150 | 0.7928 |
| 8 | 0.064328 | 0.066970 | +4.1% | 2 | 10 | 0.7161 | 0.6852 |
| 26 | 0.086705 | 0.089827 | +3.6% | 0 | 12 | 0.5417 | 0.5053 |
| 32 | 0.090952 | 0.093020 | +2.3% | 3 | 9 | 0.5066 | 0.4730 |
| 64 | 0.102412 | 0.104674 | +2.2% | 2 | 10 | 0.3813 | 0.3490 |
| 128 | 0.112782 | 0.115050 | +2.0% | 4 | 8 | 0.2510 | 0.2389 |

Negative CRB change indicates lower L2. CRB has a mean-L2 advantage only at q=2; Uniform has a mean-L2 advantage at q=4 through 128. At q=26, Uniform wins on all 12 tracks. The first trajectory had unusual small gradient scale and CRB wins at q=4 and 8, but 10 of 12 tracks favor Uniform at each of those budgets. Equal-track mean absolute L2 and paired win counts are both shown to expose this difference. Mean cosine favors Uniform at every nontrivial q, including q=2 where CRB has lower mean L2.

All selected IDs are in the FOODS_1 M5 family because the frozen 1,000-eligible-row source manifest is ordered and concentrated in that family. This remains a within-family, one-model-state study, not a representative sample of all M5 products or a trained LoRA state. No significance test or cross-family generalization claim is made. The original full-parameter CRB reconstruction result remains a different parameter space and is not directly comparable by absolute L2.

Validation and storage: the queue verified every track selection, 16 finite method-budget rows, exact matched K, unique anchors, and both endpoints. Each newly generated dense cache was validated for shape and finiteness before release. All 12 dense gradient files, including the initial track, were deleted after their metric files were retained. Twelve `cache_released.json` markers and `twelve_tracks.complete` exist. The initial gradient spot checks described above preceded its cache release. There are no retained `lora_gradients.npy` files in this experiment output tree. Aggregate artifacts: `outputs/train/v0-crb-reconst-fine/twelve_track_metrics.csv` and `twelve_track_summary.json`; per-track selections, metadata, anchors, metrics, and completion markers remain under `trajectories/trajectory-NNNN/` (the initial track remains in the output root).

## Direction-cosine decision (2026-11-10)

Last modified: 2026-11-10. At the user request, the 12-track comparison is judged by mean per-window cosine between reconstructed and true LoRA gradient vectors, with higher values better. This reuses saved per-track metrics and does not recompute gradients or change anchors. L2 results above remain historical measurements but do not enter this decision.

| q | Uniform equal-track mean cosine | CRB equal-track mean cosine | CRB minus Uniform | CRB track wins | Uniform track wins |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.000000 | 1.000000 | 0.000000 | 0 | 0 |
| 2 | 0.908663 | 0.907173 | -0.001491 | 6 | 6 |
| 4 | 0.815012 | 0.792768 | -0.022244 | 1 | 11 |
| 8 | 0.716084 | 0.685169 | -0.030915 | 1 | 11 |
| 26 | 0.541725 | 0.505308 | -0.036417 | 0 | 12 |
| 32 | 0.506590 | 0.472957 | -0.033633 | 1 | 11 |
| 64 | 0.381343 | 0.349001 | -0.032342 | 1 | 11 |
| 128 | 0.251038 | 0.238948 | -0.012090 | 4 | 8 |

Uniform has higher equal-track mean cosine at every nontrivial budget. At q=2, its mean advantage is only 0.001491 and the track wins split 6/6; this is a weak directional distinction. At q=4,8,32,64, Uniform wins 11/12 tracks; at q=26 it wins 12/12. The single initial trajectory is the only CRB cosine win at q=4 and q=8, so its initial directional result is atypical among the selected tracks. The result is descriptive for one model state and the selected FOODS_1 tracks.
