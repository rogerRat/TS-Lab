"""Pinned upstream source and fidelity disclosure for less."""
from baselines._contracts import BaselineContract

CONTRACT = BaselineContract(
        name="less",
        paper="LESS (ICML 2024)",
        source_url="https://github.com/princeton-nlp/LESS.git",
        source_revision="8abf9628b9a814ac3045445eebc8ba3c908fdc78",
        fidelity="core-signal-only",
        preserved_core=(
            "projected per-example gradients",
            "cosine target influence",
        ),
        static_framework_adaptations=(
            "Chronos window loss under the shared C/H/D protocol",
            "fixed feasible source pool and reference safe history",
            "ranked whole-trajectory point-budget delivery; uniform legal-start training sampling",
            "one trajectory per example; uniform-window raw gradient mean on both sides",
            "sampling counts match the three configured RAGS budgets",
            "bilateral SGD gradients, single pretrained checkpoint (deliberate decisions 11-D2/D3)",
            "full-parameter shared hash projection instead of LoRA (decision 11-D5)",
        ),
        unresolved_gaps=(
            "not native LESS: Adam modulation, warm-up checkpoints and LoRA intentionally omitted",
        ),
    )
