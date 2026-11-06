import unittest

import pandas as pd
import numpy as np

from tsl.spec.load_dataframe import DataContractError, validate_series_rows


def _series_table():
    series = pd.DataFrame([
        {"series_id": "a", "series": [1.0, 2.0], "length": 2, "dataset": "d", "subset": "s"},
    ])
    return series


class DataContractTest(unittest.TestCase):
    def test_valid_series_table(self):
        series = _series_table()
        validate_series_rows(series)

    def test_length_and_key_errors_are_explicit(self):
        series = _series_table()
        series.loc[0, "length"] = 3
        with self.assertRaises(DataContractError):
            validate_series_rows(series)

    def test_non_finite_series_require_explicit_preservation(self):
        """Series validation is finite by default and explicit when relaxed."""
        series = _series_table()
        series.at[0, "series"] = [1.0, np.nan]
        with self.assertRaises(DataContractError):
            validate_series_rows(series)
        validate_series_rows(series, preserve_invalid_values=True)


if __name__ == "__main__":
    unittest.main()
