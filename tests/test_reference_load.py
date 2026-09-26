import sys
from types import SimpleNamespace

import numpy as np
import pytest

from assets.dataset_processing.load.ref.fev import _asset_rows, _finite_reference_values


def test_reference_load_removes_non_finite_observations_without_filling_zero() -> None:
    values = _finite_reference_values([3.5, np.nan, np.inf, -np.inf, 0.0, -2.0])

    assert np.array_equal(values, np.asarray([3.5, 0.0, -2.0]))
    assert np.isfinite(values).all()


def test_reference_load_rejects_non_univariate_target() -> None:
    with pytest.raises(ValueError, match="non-univariate"):
        _finite_reference_values([[1.0, 2.0]])


def test_reference_load_warns_when_non_finite_values_are_skipped(monkeypatch, caplog) -> None:
    class FakeDataset:
        def with_format(self, _format: str):
            return [{"item_id": "series-1", "target": [1.0, np.nan, np.inf, 2.0]}]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_from_disk=lambda _path: FakeDataset()))

    rows = list(_asset_rows("unused", "fev-task"))

    assert rows[0]["values"].tolist() == [1.0, 2.0]
    assert "REFERENCE VALUE SKIPPED" in caplog.text
