# V7 result figures

From `/root/TS3`, run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:. \
 /root/miniconda3/envs/ts3/bin/python -m scripts.plot_v7_results
```

The existing twelve PNG filenames under `docs/results/figures` are overwritten
only after rendering all twelve successfully. No training/gradient computation is
triggered; diagnostics read selected existing shared-cache reference rows only.
Do not use the old `plot_all_results.py` / `plot_main_debug.py`: those target V6.

The active configuration supplies seven tasks and prescribed epochs: Chronos 3,
Timer/TimesFM 4. Every curve uses the arithmetic mean of available **completed**
cells' full prescribed half-epoch grids. Historical Chronos four-epoch curves are
truncated at three; incomplete curves are skipped and recorded. No best checkpoint
selection. As requested, figures show no seed counts. Per-cell contributors and
source paths remain in `v7_plot_snapshot.json` for reproducibility.

The main group includes grad-dot, grad-cos and grand-static. User's original styling
is preserved: plasma, best-present darkest (0), worst-present brightest (1), with
per-figure normalization. Main/aggregation/uni color ordering follows the original
pooled final-epoch MASE across all models/tasks/completed seeds; debug ordering uses
the original per-model pooled MASE. Debug trajectory colors use viridis. Curves are
solid with circle markers, orig black dashed; original sizes, grids and typography
are retained. Color ranking is the user's original visualization convention, not
a normalized cross-task benchmark claim. No seed counts appear in figures.

User's symlog trial: all MASE and gradient panels now use base 10 symlog with
linthresh=0.01 and linscale=1. No clipping, winsorization, deletion or change to curve
values/colors occurs. Narrow positive ranges use numeric tick labels for readability.
Use `--reuse-snapshot` to change presentation on the exact existing mean curves;
without that flag, the command reads the newest completed V7 cells.

Latest presentation revision: MASE returns to a linear axis. If Tukey outliers
expand the full range to at least twice the bulk span, the main panel focuses on
the bulk (8% padding) and an inset retains ALL original points and the full range,
with its maximum labeled. Up/down boundary triangles explicitly disclose points
outside the zoomed main panel. Uncompressed panels keep the original full view.
No values are removed or winsorized. The exact focus limits are recorded in the
snapshot. Gradient panels retain symlog but hide numeric y tick labels and offset
text; window-position x tick labels remain. All original colors/styles are retained.

Latest layout: no legends or overview axes inside any data panel. The two unused
bottom-right main-grid positions hold a method legend and a separate 3x3 grid of
full-range views, matching the seven task positions in the main grid. Tasks without
a zoom leave their corresponding overview slot blank, as do the final two slots.
Debug reference-trace legend also moves to this area. Source plots, colors, scales
and mean data are unchanged by this layout revision.

Placeholder revision: all nine overview slots retain a visible frame. Each task
slot always has its task name; if no overview is needed, it says `No zoom needed`.
The final two non-task slots are labeled `Unused slot`. This supersedes blank,
frameless placeholders so the spatial correspondence remains explicit.

Debug plots retain performance above reference-gradient profiles. The latter are
frozen pretrained projected-gradient L2 norms along cached window positions, NOT
gradients over training epochs. At most six evenly selected reference trajectory
IDs are displayed; sparse cache rows are connected for visualization. No data/model
hashes or cache writes occur. Seven-task selection was post-observation per the
approved V7 protocol; figures must not be described as the original nine-task benchmark.
