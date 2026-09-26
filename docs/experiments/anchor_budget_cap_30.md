# Anchor replay: 30% nominal budget ceiling

The user fixes `cover_density + error_density + split_density <= 0.30`.
The replay rejects negative, nonfinite, or over-cap input combinations before
reading gradient caches or writing outputs. Decimal comparison accepts the
mathematical 0.10 + 0.10 + 0.10 boundary without a floating-point false rejection.

The full grid retains all 125 original low-budget combinations and adds exactly
eight points: (0.12,0.02,0.01), (0.16,0.03,0.01), (0.20,0.04,0.01),
(0.30,0,0), (0.25,0.05,0), (0.25,0.04,0.01), (0.20,0.08,0.02),
(0.15,0.10,0.05). There is no automatic budget expansion. The small self-test
grid remains unchanged. This is a bounded search, not proof of the global
optimum over all possible allocations under 30%.

Final complete-sample replay selects the passing candidate with the fewest
actual unique anchors. Passing requires each of the five tasks' window-weighted
production block-constant mean cosine >=0.90. Diagnostic interpolation is not
used for this decision. Means used for ranking retain full numerical precision.

If no candidate passes, `recommendation` is null and `best_effort` reports the
candidate that maximizes the worst task mean, then the task macro mean, then
minimizes actual unique anchors. It explicitly retains `pass_block: false` and
all task means. Preliminary/incomplete runs emit neither recommendation.

Schema v2 summary records the finite grid, nominal ceiling, and distinct
`best_effort`; per-trajectory details now include the three densities and actual
unique-anchor/window ratio. Task rows and recommendations also report actual
cost. Required endpoints and integer rounding can make actual anchor proportions
differ from nominal densities. Candidate costs explicitly record
`actual_budget_usable`; recommendations additionally exclude candidates whose
total actual unique anchors / total legal windows exceeds 30%. Per-task and
per-trajectory rows also flag local overages for inspection. Production CRB
rounding and endpoint semantics are preserved; results are not truncated.

When no split event occurs, split density is not identifiable from these samples;
the output says so. A tied candidate's split setting is not evidence of an optimal
split budget. Outputs retain the scan's data-disk directory convention, atomic
JSON publication, and read-only gradient-cache reuse; no new hashes are used.

Already running preliminary replay processes retain their loaded old grid and
can finish normally. A subsequently launched final process loads this capped
grid. The gradient scanner is unchanged.
