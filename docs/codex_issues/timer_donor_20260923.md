# Timer donor rejection audit — 2026-09-23

Read-only diagnosis on cloned instance port 22145, code 820804f. No model
execution or new BP for this audit; no manifests or acceptance rules changed.

Observed rejection is the nested `model` field. Exact leaf differences:
old config.torch_dtype=bfloat16 versus new config.dtype=bfloat16;
old config.transformers_version=4.40.1 versus new=4.57.1. All remaining model
and non-data manifest fields match. Installed PretrainedConfig.to_dict explicitly
sets transformers_version to the serializer's installed version; this field alone
does not establish changed checkpoint weights or changed loss semantics.

Physical data identities validated using the existing verified_donors routine:
complete source trajectory values/native rows and task reference values. Published
indexes expose 35,428 source and 3,050 first-task reference overlapping windows.
Deterministic stratified sample: 32 source + 32 reference windows, selected across
the sorted overlap, using only already stored old/new gradient/loss arrays.

Results: 0/64 exact BF16 gradient equality; 0/64 exact loss equality;
minimum cosine 0.9970190770511257; maximum relative gradient L2 difference
0.07972383004928753; maximum relative loss difference 0.03851220297563741.
This is a sample, not a full-cache numerical equivalence result.
Machine-readable rows: outputs/storage-audit-20260923/timer-donor-audit.json.

Execution evidence: old GPU RTX PRO 6000 Blackwell Server Edition, new GPU A800;
both torch 2.7.1+cu128. Old recorded gradient batches 32/256 with VJP settings
changing over time, new batch8/VJP16. These changes are confounded: available
evidence cannot attribute the differences uniquely to hardware, batching,
Transformers runtime, or another execution detail. Revision/precision/loss
manifest fields do not show a scientific protocol change, but neither are the
stored values numerically identical. Do not claim model weights differ solely
from config serialization or claim safe compatibility solely from same revision.

Conservative disposition: do not silently ignore model differences or invent
numeric tolerances. Keep old/new Timer resources isolated pending further approved
equivalence validation. Continue the current manifest-valid V7 preparation and
reuse already completed V7 rows across tasks/seeds; this is not a requirement to
recompute the first task's source rows for all subsequent tasks. No worker restart
is necessary for this diagnosis. A future Python code change would require a
safe reentry, not automatically hot-load at the next task.
