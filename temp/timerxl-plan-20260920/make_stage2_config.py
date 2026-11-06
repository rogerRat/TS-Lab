"""Select stage-1 top-2 learning rates and emit the stage-2 (earlier-history) config.

Pre-registered rule (PLAN.md step 3/4): rank stage-1 lrs by the macro average over both
safe-history tasks of MASE/orig at the final common checkpoint (epoch 2). The best two
are re-confirmed on the independent earlier history blocks with seed 1.
"""
import argparse
import json
from pathlib import Path

STAGE2_PATH = Path("/root/TS3/configs/rage/development_timerxl_history_early_0920.json")


def macro_score(trial, datasets, original_by, epoch):
    for record in trial["curve"]:
        if abs(record["epoch"] - epoch) < 1e-9:
            metrics = record["metrics"]["by_dataset"]
            return sum(metrics[d]["mase"] / original_by[d] for d in datasets) / len(datasets)
    raise ValueError(f"epoch {epoch} missing for lr={trial['learning_rate']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", required=True, type=Path)
    args = parser.parse_args()
    results = json.loads((args.stage1 / "results.json").read_text())
    config = json.loads((args.stage1 / "protocol.json").read_text())
    datasets = config["datasets"]
    original_by = {d: results["original"]["by_dataset"][d]["mase"] for d in datasets}
    scores = {trial["learning_rate"]: macro_score(trial, datasets, original_by, epoch=2.0)
              for trial in results["trials"]}
    ranked = sorted(scores, key=scores.get)
    top2 = ranked[:2]
    stage2 = dict(config)
    stage2.update(
        inputs="/root/autodl-tmp/TS3/outputs/timerxl-calibration-20260920/inputs-early",
        seeds=[1], epochs=[0.25, 0.5, 1, 2, 4], learning_rates=top2,
        purpose="development_safe_history_confirmation_independent_earlier_blocks")
    STAGE2_PATH.write_text(json.dumps(stage2, indent=1) + "\n")
    trace = {"rule": "macro mean over tasks of MASE/orig at epoch 2, seed 0, devhistory split",
             "scores": {str(k): v for k, v in scores.items()}, "ranked": ranked, "top2": top2,
             "stage2_config": str(STAGE2_PATH)}
    (args.stage1 / "stage2_selection.json").write_text(json.dumps(trace, indent=1))
    print("STAGE2_TOP2", json.dumps(trace, indent=1))


if __name__ == "__main__":
    main()
