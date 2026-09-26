# Window-Aware Trajectory-Balanced Sampling with Analysis Segments

> 历史归档：保留当时的设计、操作或实验证据，不作为当前执行指令。当前入口见 `docs/README.md`。

This document implements the mathematical problem statement in `FEVWindowAwareTrajectoryBalancedSamplingDesign.md`. Part I explains the design for people. Part II is the binding implementation contract for agents.

# Part I: Human Explanation

## Goal

The logical unit to match between Source and Reference is a training `Window`. A trajectory $t_i$ has $n_i$ legal windows. Each logical window has baseline mass $1/(|T|n_i)$, so every eligible trajectory has the same total baseline mass regardless of length.

Enumerating every window, computing its feature, and indexing it is infeasible. The solution does **not** change the mathematical target. It uses `AnalysisSegment` (AnaSeg) as a lightweight, computable proxy for windows.

## Objects

```text
Dataset
  -> Trajectory
       -> AnalysisSegment[]
       -> complete Window (created only after sampling)
```

An AnaSeg is a fixed interval of a trajectory. Offline, each AnaSeg receives a cached six-dimensional feature. `analysis_segment_stride == analysis_segment_length` preserves the original non-overlapping geometry; a smaller stride permits sliding overlapping AnaSegs. A window still belongs to its trajectory, not to one AnaSeg: a boundary-crossing window can be represented by multiple AnaSegs.

A `PseudoWin` is only the logical tuple `(trajectory_id, start, window_length)`. Selecting a PseudoWin means selecting a legal start. It has no values and is never materialized as a complete collection. The selected start is finally converted to one `WindowLocator` and one real `Window`.

## Segment Mass

For non-overlapping segments, define overlap ratio $\alpha_{i,j,k}=|w_{i,j}\cap a_{i,k}|/|w_{i,j}|$. For overlapping segments, divide each time point's contribution by its AnaSeg coverage count before summing, so $\sum_k\alpha_{i,j,k}=1$ for every legal window. The raw mass held by the segment is:

$b^a_{i,k}=\sum_{j=1}^{n_i}\frac{1}{|T|n_i}\alpha_{i,j,k}$.

This may be fractional. A window that is 40% in segment A and 60% in B contributes 0.4 and 0.6 of its mass respectively. Therefore $\sum_k b^a_{i,k}=1/|T|$ for every trajectory.

The feature-space point is the AnaSeg feature itself. Its mass is the raw segment mass above. There is no `PseudoWindowFeatureSpace`: the correct objects are `SourceAnalysisSegmentFeatureSpace` and `ReferenceAnalysisSegmentFeatureSpace`.

## End-to-End DAG

```text
Offline:
LoadBundle -> Trajectory -> AnaSeg[] -> cached AnaSeg features + Bundle metadata

Preflight:
Training Contract + Source/Reference Catalog
  -> SourceBundleReader / ReferenceBundleReader
  -> complete Bundle validation and Compute parts

Compute:
RuntimeDatasetComputationPart -> TrajComputationPart -> AnaSegComputationPart
  -> Source/ReferenceAnalysisSegmentFeatureSpace
  -> OT/KDE -> AnalysisSegmentMassResult

Sampling build:
MassResult -> RuntimeDatasetSamplingPart
  -> TrajectorySamplingPart -> AnalysisSegmentSamplingPart
  -> PseudoWindowsStartsRepresentation
  -> release compute features, indexes, references, and solver state

Online:
Trajectory -> AnaSeg -> legal start -> one complete Window -> WindowAccessor -> Adapter
```

The training side must not bypass Catalog. Catalog is the controlled training index for trajectory identity and Bundle locators; Bundle readers open only Bundles referenced by validated Source/Reference Catalogs.

Offline `ANALYSIS_SEGMENT_LENGTH` is immutable Feature metadata. At training startup, windows may be longer than one analysis segment; the runtime emits a warning and uses overlap-weighted segment mass for windows spanning multiple analysis segments. The full trajectory must still contain at least one legal window.

After OT/KDE assigns final mass $q_{i,k}$ to Source AnaSegs, trajectory mass is $P(t_i)=\sum_k q_{i,k}$. Sampling first chooses a trajectory, then an AnaSeg, then a legal start represented by that AnaSeg. A start may lie outside the selected segment provided the complete window overlaps it. A crossing window can be reached through either adjacent segment; the two path probabilities add.

# Part II: Agent Implementation Contract

## Invariants

- Do not change the anonymous Adapter protocol, `WindowAccessor`, `WindowLocator`, tickets, batches, or `run_core_loop`.
- Do not compute features at training time.
- Do not persist AnaSeg features in Catalog; Feature Bundle is their only persistent source.
- Do not instantiate all `Window`/`PseudoWin` objects or a per-start dictionary/array.
- `sampling.trajectory_quality` does not exist in this policy. Random, OT, and KDE schemas must reject it.
- `SamplingPart` must never access Feature vectors, Reference data, Bundle readers, OT/KDE state, or recompute mass.

## Assets and Readers

`ts3.features.config.ANALYSIS_SEGMENT_LENGTH` and `ANALYSIS_SEGMENT_STRIDE` are positive global feature constants and part of the feature definition. `feature/metadata.json` records `schema_version`, `analysis_segment_length`, `analysis_segment_stride`, and `feature_definition_fingerprint`; old metadata without stride is interpreted as non-overlapping. `feature/feature.arrow` has one row per Load trajectory with ordered AnaSeg records: `segment_index`, `start`, `length`, and six-dimensional `feature_values`.

`SourceBundleReader` and `ReferenceBundleReader` are the only readers. Construct each from a validated Catalog and its Bundle locator. They read metadata and Arrow data, validate ID/order/geometry/feature validity, build compute parts, and close after sampling parts are built. No `FeatureBundleReader` or metadata-only reader exists.

Invalid-value handling is intentionally role-specific at the offline Feature stage. A Source trajectory containing any `NaN` or infinity has no valid Feature row and is dropped as a whole, so it cannot reach training. A Reference trajectory preserves its immutable Load values, total length, and fixed AnaSeg starts/lengths. For Reference feature computation only, each AnaSeg masks non-finite values before feature calculation; the resulting feature remains attached to that AnaSeg's original geometry. Reference Load values never reach model training or loss computation.

## Class Boundaries

| Class | Owns | Allowed purpose | Forbidden | Lifetime |
|---|---|---|---|---|
| `RuntimeDatasetComputationPart` | metadata, `TrajectoryComputationPart[]` | build segment feature space | online sampling | solve/build only |
| `TrajectoryComputationPart` | locator, length, legal count, `AnalysisSegmentComputationPart[]` | trajectory normalization | window delivery | compute only |
| `AnalysisSegmentComputationPart` | geometry, `.feature`, raw mass | feature-space row and mass | start index, data reads | compute only |
| `SourceAnalysisSegmentFeatureSpace` | source feature matrix, raw mass vector, IDs | index and OT/KDE source input | PseudoWin rows | release after compile |
| `ReferenceAnalysisSegmentFeatureSpace` | reference matrix, mass, IDs | reference solve input | training residency | release after aggregation |
| `AnalysisSegmentFeatureIndex` | source-neighbor structure | OT/KDE support/filter | sampling | release after solve |
| `AnalysisSegmentMassResult` | AnaSeg ID -> reassigned mass, diagnostics | sole quality input to sampling build | feature storage | retain only serialized audit summary |
| `RuntimeDatasetSamplingPart` | `TrajectorySamplingPart[]`, fingerprint | `ScheduleCore` sampling input | feature/solver access | training lifetime |
| `TrajectorySamplingPart` | locator, length, trajectory mass, segment parts | choose a segment conditional on trajectory | feature access | training lifetime |
| `AnalysisSegmentSamplingPart` | geometry, reassigned mass, representation | choose a start through its representation | Feature/OT/KDE | training lifetime |
| `PseudoWindowsStartsRepresentation` | compressed start intervals, stride, boundary-linear/internal-constant weights, cumulative mass | `sample_start(rng)` | per-start object/dictionary/array | training lifetime |
| `SegmentWeightedWindowSampler` | read-only SamplingPart + window spec | trajectory -> segment -> start -> one Window | feature calculation or solver work | Core lifetime |

Conversion is one-way: `AnalysisSegmentComputationPart + reassigned_mass -> AnalysisSegmentSamplingPart`, then release `.feature` and compute mass. The Source computation dataset converts to `RuntimeDatasetSamplingPart`; all Reference compute datasets and all Feature/solver resources are released.

## Required Phases

1. **Preflight:** parse contract; read Catalog first; use Bundle readers; validate Bundle schema, metadata, segment geometry, legal trajectories, and $L_{analysis}\ge L_{window}$; fail early.
2. **Compute build:** create computation parts; calculate each AnaSeg raw mass from logical window baseline mass and overlap; build Source/Reference AnalysisSegment Feature Spaces. Compute overlap by compressed left-boundary, interior, and right-boundary rank ranges; never enumerate windows.
3. **Solve:** run segment-level OT/KDE, transfer filtering, aggregation, and lambda mixing. Output normalized `AnalysisSegmentMassResult`. Random bypasses this phase entirely: choose an eligible Source trajectory uniformly, then choose one legal window start uniformly within that trajectory.
4. **Sampling build:** sum AnaSeg mass into trajectory mass; build each `PseudoWindowsStartsRepresentation`; construct sampling parts; release all compute-only resources.
5. **Online sampling:** choose trajectory by trajectory mass; choose AnaSeg by its conditional mass; call `sample_start`; create exactly one `WindowLocator` and one Window; use `WindowAccessor` and Adapter.

`PseudoWindowsStartsRepresentation` represents the semantic mapping `start -> overlap contribution for this AnaSeg`. It must use compressed rank ranges and analytic cumulative weights, not a materialized mapping. Its legal starts include every full window overlapping the AnaSeg, including starts outside the AnaSeg.

## Correctness and Acceptance

For any window, hierarchical sampling must satisfy $P(w_{i,j})=\sum_k q_{i,k}\alpha_{i,j,k}/c_{i,k}$, equivalent to the approved window-level proxy distribution. Test metadata and geometry validation, early incompatibility failure, analytic versus explicit small-case mass, mass conservation, cross-segment windows, stride alignment, random baseline, one-Window-per-slot creation, resource release, audit/resume fingerprints, and existing Adapter/Core determinism.
