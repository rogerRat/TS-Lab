"""Pinned upstream source and fidelity disclosure for ltsv."""
from baselines._contracts import BaselineContract

CONTRACT = BaselineContract(
        name="ltsv",
        paper="LTSV (DASFAA 2026)",
        source_url="https://github.com/b1302550313/LTSV.git",
        source_revision="38221dc7505ccf5ae22615f2fcba40a39b59830b",
        fidelity="paper-equation-port",
        preserved_core=("one-step block adaptation", "context loss change", "point/window aggregation"),
        static_framework_adaptations=(
            "native Chronos CE block training; autoregressive greedy forecast MSE for reference",
            "blocks and outputs are constrained to the shared window protocol",
            "fixed feasible source pool and reference safe history",
            "native top-k with overlapping windows allowed",
            "target safe history supplies context instead of same-dataset folds",
        ),
        unresolved_gaps=(
            "official HEAD contains broken gitlinks and no runnable valuation driver",
            "historical epoch count alone does not establish optimizer step count",
            "Chronos forecast readout and longer-block subwindow training are model adapters",
        ),
    )
