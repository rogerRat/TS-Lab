from pathlib import Path

import pytest

from visualize.loss_step_line import _run_directories, collect_rows


def _write_loss(run_dir: Path, loss: float = 1.0) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "losses.csv").write_text(
        "optimizer_step,loss\n1,%s\n" % loss,
        encoding="ascii",
    )


def test_searches_train_when_train_meta_is_missing(tmp_path: Path) -> None:
    train = tmp_path / "train"
    _write_loss(train / "rand")
    runs = _run_directories([tmp_path / "train_meta", train])
    assert runs == [("rand", (train / "rand").resolve())]
    assert collect_rows([tmp_path / "train_meta", train]) == [
        {"method": "rand", "step": 1, "loss": 1.0}
    ]


def test_archived_metadata_has_priority_over_live_train(tmp_path: Path) -> None:
    train_meta = tmp_path / "train_meta"
    train = tmp_path / "train"
    _write_loss(train_meta / "rand", 2.0)
    _write_loss(train / "rand", 3.0)
    _write_loss(train / "cu-r1", 4.0)
    runs = _run_directories([train_meta, train])
    assert [name for name, _ in runs] == ["rand", "cu-r1"]
    rows = collect_rows([train_meta, train])
    assert {(row["method"], row["loss"]) for row in rows} == {
        ("rand", 2.0), ("cu-r1", 4.0)
    }


def test_raises_only_when_all_search_paths_are_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no losses.csv"):
        _run_directories([tmp_path / "train_meta", tmp_path / "train"])
