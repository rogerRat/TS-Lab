import numpy as np
import pytest

from scripts.quick_crb_uniform_ref import ReadOnlyCachedComputer, accuracy, report


def test_accuracy_identical_and_zero_vectors():
    truth = np.array([[1., 0.], [0., 0.], [0., 2.]], dtype=np.float32)
    same = accuracy(truth, truth)
    assert same["relative_frobenius_rmse"] == 0
    assert same["windows"] == 3
    assert same["mean_window_cosine"] == pytest.approx(2 / 3)
    shifted = accuracy(truth, np.zeros_like(truth), np.array([True, False, True]))
    assert shifted["windows"] == 2
    assert shifted["relative_frobenius_rmse"] == pytest.approx(1)


def test_read_only_crb_reader_fails_closed_on_missing_native_row():
    class Dataset:
        _positions = {("reference:demo", 4, 1): 0}

        def lookup(self, dataset, ids):
            return np.ones((len(ids), 2), dtype=np.float32)

    class Fragment:
        fragment_id = 4

    reader = ReadOnlyCachedComputer(Dataset(), "reference:demo")
    assert reader(Fragment(), np.array([1])).shape == (1, 2)
    with pytest.raises(RuntimeError, match="missing production rows"):
        reader(Fragment(), np.array([2]))


def test_report_labels_truth_cost_and_reconstruction_limit():
    a = {"relative_frobenius_rmse": 0.2}
    payload = dict(per_trajectory=[dict(fragment_id=1, legal_windows=10, bp_each=2,
        native_crb=a, uniform=a, uniform_minus_native_rmse=0)],
        pooled=dict(native_crb=a, uniform=a, native_crb_off_union=a,
            uniform_off_union=a), total_legal_windows=10,
        full_truth_cost=dict(production_reused_windows=2, new_full_truth_bp=8),
        interpretation_limit="This is not downstream MASE")
    text = report(payload)
    assert "independent full-gradient truth" in text
    assert "This is not downstream MASE" in text
