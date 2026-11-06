"""Pinned upstream source and fidelity disclosure for tsds."""
from baselines._contracts import BaselineContract

CONTRACT = BaselineContract(
        name="tsds",
        paper="TSDS (NeurIPS 2024)",
        source_url="https://github.com/ZifanL/TSDS.git",
        source_revision="3dfe13953cffaefaeb9d401172e495aa7e0fd775",
        fidelity="algorithm-port",
        preserved_core=("FAISS IVF with upstream nlist/nprobe", "nearest-neighbour OT assignment",
                        "KDE diversity", "multinomial repeated trajectory copies"),
        static_framework_adaptations=(
            "one trajectory per OT point; uniform-sampled frozen Chronos window embedding mean",
            "sampling counts match the three configured RAGE budgets",
            "experiment-frozen max_k=64, kde_k=16, sigma=4, alpha=0.1, c=5 (not upstream defaults)",
            "shuffle multinomial trajectory copies, deliver whole copies until point budget",
            "duplicate trajectories retained in delivery and legal-start training sampling; repeated points reported",
            "fixed feasible source pool and reference safe history",
        ),
        unresolved_gaps=(),
    )
