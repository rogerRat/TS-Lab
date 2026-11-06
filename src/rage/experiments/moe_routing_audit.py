"""Read-only native MoE finite differences with actual expert-route observations."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .calibration import build_student
from .native_probe import NativeProbe
from .regions import load_selection
from .students import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-selection", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpu-memory-fraction", type=float, default=.18)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    source = load_selection(args.source_selection)
    # Same first source window as the initial 64-window audit.
    row = np.random.default_rng(20260913).choice(len(source.selected), 64, replace=False)[0]
    probe = NativeProbe(build_student("timemoe-200m"), model_key="timemoe-200m")
    routes, handles = {}, []
    k = probe.model.config.num_experts_per_tok

    def hook(name):
        def capture(module, inputs, output):
            # Upstream routes through softmax then top-k. Sorting ignores order
            # changes within an unchanged selected expert set.
            routes[name] = output.detach().float().softmax(-1).topk(k, dim=-1).indices.sort(-1).values.cpu()
        return capture

    for name, module in probe.model.named_modules():
        if name.endswith(".gate"):
            handles.append(module.register_forward_hook(hook(name)))
    baseline = {}

    def observe():
        if not baseline:
            baseline.update({name: value.clone() for name, value in routes.items()})
        changed = {name: int((value != baseline[name]).any(-1).sum()) for name, value in routes.items()}
        return {"changed_token_layer_routes": sum(changed.values()),
                "total_token_layer_routes": sum(len(value) for value in routes.values()),
                "changed_by_layer": changed}

    results = {}
    try:
        for label, prefix in [("full", None), ("prediction_heads", "lm_heads.")]:
            baseline.clear()
            results[label] = probe.finite_difference(source.selected[row],
                steps=(1e-3, 1e-4, 1e-5), parameter_prefix=prefix, forward_observer=observe)
            print(label, results[label], flush=True)
        # Diagnostic only: hold the discrete top-k branch at the original
        # model's assignments, but gather CURRENT routing probabilities. This
        # checks the derivative on that branch, not a modified training loss.
        original_topk = torch.topk
        assignments = []
        def record_topk(values, count, dim=-1, **kwargs):
            result = original_topk(values, count, dim=dim, **kwargs)
            assignments.append((dim, result.indices.detach().clone()))
            return result
        with patch("torch.topk", record_topk), torch.no_grad():
            probe.engine.training_loss(source.selected[row][None])
        position = 0
        def fixed_topk(values, count, dim=-1, **kwargs):
            nonlocal position
            recorded_dim, indices = assignments[position % len(assignments)]
            position += 1
            assert dim == recorded_dim and indices.shape[dim] == count
            return values.gather(dim, indices), indices
        baseline.clear()
        with patch("torch.topk", fixed_topk):
            results["full_fixed_discrete_branch_diagnostic_only"] = probe.finite_difference(
                source.selected[row], steps=(1e-3, 1e-4), forward_observer=observe)
        assert position % len(assignments) == 0
        print("fixed_branch", results["full_fixed_discrete_branch_diagnostic_only"], flush=True)
    finally:
        for handle in handles:
            handle.remove()
    atomic_json(Path(args.out), dict(source_id=source.window_ids[int(row)], audits=results,
        purpose="Diagnostic directions only; native training loss and routing are unchanged"))
    print("MOE_ROUTING_AUDIT_DONE", flush=True)


if __name__ == "__main__":
    main()
