"""Pinned upstream source and fidelity disclosure for lp_mirror."""
from baselines._contracts import BaselineContract

CONTRACT = BaselineContract(
        name="lp_mirror",
        paper="LP-Mirror (Scientific Reports 2026)",
        source_url="https://github.com/LP-Mirror/LP_Mirror.git",
        source_revision="43d010171ad9dfa713bf9e3b5d90ff744e1ead29",
        fidelity="algorithm-port",
        preserved_core=("source proxy pre-training", "target mirrored adaptation", "loss-drop scoring"),
        static_framework_adaptations=(
            "channel-independent linear proxy over materialized C/H windows",
            "fixed feasible source pool and reference safe history",
            "safe-trajectory scaling replaces dataset/channel scaler inputs",
            "official top-k mode with a fixed count; overlapping windows remain eligible",
        ),
        unresolved_gaps=("ported selector, not a reproduction of upstream datasets/backbones",),
    )
