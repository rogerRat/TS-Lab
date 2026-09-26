"""Propose the frozen Timer-XL training recipe from stage-1 + stage-2 calibration.

Pre-registered rules (PLAN.md steps 3-4):
- lr: best stage-1 top-2 candidate by macro MASE/orig on the independent earlier blocks
  (seed 1), cross-checked to match the stage-1 ranking direction.
- epochs: argmin over observed checkpoints of the stage-2 macro score for the chosen lr.
  If the curve is still improving at the last observed checkpoint (epoch 4), freeze is
  NOT automatic — status 'needs_extension' requests 6/8-epoch confirmation instead.
Never opens formal exam arrays; safe-history only.
"""
import argparse
import json
from pathlib import Path


def macro(record, datasets, original_by):
    metrics = record["metrics"]["by_dataset"]
    return sum(metrics[d]["mase"] / original_by[d] for d in datasets) / len(datasets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", required=True, type=Path)
    parser.add_argument("--stage2", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    s1 = json.loads((args.stage1 / "results.json").read_text())
    s2 = json.loads((args.stage2 / "results.json").read_text())
    c2 = json.loads((args.stage2 / "protocol.json").read_text())
    datasets = c2["datasets"]
    original_by = {d: s2["original"]["by_dataset"][d]["mase"] for d in datasets}

    stage2_curves = {}
    for trial in s2["trials"]:
        curve = [(r["epoch"], macro(r, datasets, original_by)) for r in trial["curve"]]
        stage2_curves[trial["learning_rate"]] = sorted(curve)
    # Choose lr by the best observed checkpoint within the stage-1 horizon (<=2 epochs),
    # so a late-epoch fluke cannot pick a lr that was worse inside the compared horizon.
    best = {}
    for lr, curve in stage2_curves.items():
        within = [(e, s) for e, s in curve if e <= 2.0 + 1e-9]
        best[lr] = min(within, key=lambda t: t[1])
    chosen_lr = min(best, key=lambda lr: best[lr][1])
    curve = stage2_curves[chosen_lr]
    best_epoch, best_score = min(curve, key=lambda t: t[1])
    status = "frozen"
    note = None
    if best_epoch >= max(e for e, _ in curve) - 1e-9 and best_epoch >= 4.0 - 1e-9:
        earlier = [s for e, s in curve if abs(e - 2.0) < 1e-9]
        if earlier and best_score < earlier[0]:
            status = "needs_extension"
            note = ("stage-2 curve still improving at epoch 4; run 6/8-epoch confirmation "
                    "on the same earlier blocks before freezing epochs")
    stage1_rank = json.loads((args.stage1 / "stage2_selection.json").read_text())["ranked"]
    cross_check = {"stage1_ranking": stage1_rank,
                   "stage1_best": stage1_rank[0],
                   "stage2_chosen": chosen_lr,
                   "consistent": bool(stage1_rank and stage1_rank[0] == chosen_lr)}
    recipe = dict(student="timerxl-84m", learning_rate=chosen_lr,
                  epochs=best_epoch if status == "frozen" else None,
                  batch_size=c2["batch_size"], weight_decay=c2["weight_decay"],
                  optimizer=c2["optimizer"], schedule_epochs=c2["schedule_epochs"],
                  training_data=c2["training_data"], context_length=512, prediction_length=64,
                  status=status, note=note, cross_check=cross_check,
                  stage2_curves={str(k): v for k, v in stage2_curves.items()})
    args.out.write_text(json.dumps(recipe, indent=1) + "\n")
    print("RECIPE_PROPOSAL", json.dumps({k: v for k, v in recipe.items() if k != "stage2_curves"},
                                        indent=1))


if __name__ == "__main__":
    main()
