import json

import numpy as np
import pytest

from rags.experiments import shared_formal_inputs as inputs
from rags.experiments.development_inputs import load_fragments


def test_full_reference_native_tail_and_full_causal_scales():
    # History under the old dev minimum1600 remains valid for formal use.
    values = np.arange(965, dtype=np.float64) ** 2
    refs, exam, series = inputs.build_task_arrays([("a", values, 768)], seasonality=600)
    np.testing.assert_array_equal(refs[0].data, values[:768].astype(np.float32))
    assert series[0]["exam_origins"] == [768, 832, 896]
    assert series[0]["unused_tail_points"] == 5
    for index, origin in enumerate(series[0]["exam_origins"]):
        np.testing.assert_array_equal(exam["windows"][index], values[origin-512:origin+64])
        expected = np.mean(np.abs(values[600:origin] - values[:origin-600]))
        assert exam["scales"][index] == pytest.approx(expected)
    # Future mutations cannot change pre-exam refs or the first MASE scale.
    changed = values.copy()
    changed[768:] += 1e5
    other_refs, other_exam, _ = inputs.build_task_arrays([("a", changed, 768)], seasonality=600)
    np.testing.assert_array_equal(other_refs[0].data, refs[0].data)
    assert other_exam["scales"][0] == exam["scales"][0]


def test_nonfinite_history_preserves_all_legal_contiguous_segments():
    values = np.arange(1600, dtype=np.float64)
    values[600] = np.nan
    refs, exam, series = inputs.build_task_arrays([("a", values, 1300)], seasonality=24)
    assert [f.length for f in refs] == [600, 699]
    assert [b["start"] for b in series[0]["reference_segments"]] == [0, 601]
    assert np.isfinite(exam["windows"]).all()
    assert series[0]["nonfinite_reference_points"] == 1


def test_exam_invalidity_does_not_remove_reference_and_constants_remain():
    values = np.ones(960)
    values[800] = np.nan
    refs, exam, series = inputs.build_task_arrays(
        [("bad_tail", values, 768), ("constant", np.ones(960), 768)], seasonality=24)
    assert len(refs) == 2
    assert [f.entity_id for f in refs] == ["bad_tail", "constant"]
    assert series[0]["invalid_exam_windows"] == 3
    assert exam["entities"].tolist() == ["constant"] * 3
    assert exam["scales"].tolist() == [0.0] * 3


def test_atomic_export_completion_identity_and_reuse(tmp_path, monkeypatch):
    declarations = tmp_path / "declarations"
    declarations.mkdir()
    task = dict(id="quito_min", benchmark="quito", subset="min", root="pinned-test-root")
    (declarations / "quito_min__manifest.json").write_text(json.dumps(dict(task=task, revision="pinned")))
    shared = dict(datasets=["quito_min"], input_roots={"quito_min": str(declarations)})
    calls = []
    def rows(_task):
        calls.append(_task)
        return iter([("a", np.arange(960), 768)]), dict(seasonality=24, revision="pinned")
    monkeypatch.setattr(inputs, "native_rows", rows)
    out = tmp_path / "inputs"
    result = inputs.prepare_shared_formal_inputs(shared, out)
    assert result[0]["protocol"] == inputs.PROTOCOL
    assert result[0]["exam_windows"] == 3
    assert load_fragments(out / "quito_min__reference.npz")[0].length == 768
    with np.load(out / "quito_min__exam.npz", allow_pickle=False) as exam:
        assert exam["windows"].shape == (3, 576)
    assert not list(out.glob("*.partial*"))
    inputs.prepare_shared_formal_inputs(shared, out)
    assert len(calls) == 1
    manifest_path = out / "quito_min__manifest.json"
    saved = json.loads(manifest_path.read_text())
    saved["exam_stride"] = 1
    manifest_path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="identity mismatch"):
        inputs.prepare_shared_formal_inputs(shared, out)


def test_reject_development_exam_mode():
    with pytest.raises(ValueError, match="rolling"):
        inputs.build_task_arrays([], seasonality=24, evaluation_mode="single_window")


def test_external_data_disk_path_is_supported(tmp_path):
    from evaluate.time import _project_relative
    assert _project_relative(str(tmp_path), "TIME dataset.root") == tmp_path.resolve()


def test_short_series_reported_without_excluding_usable_channels():
    refs, exam, series = inputs.build_task_arrays(
        [("short", np.arange(500), 0), ("valid", np.arange(960), 768)], seasonality=24)
    assert [f.entity_id for f in refs] == ["valid"]
    assert series[0]["reference_segments"] == []
    assert series[0]["exam_origins"] == []
    assert len(exam["windows"]) == 3
