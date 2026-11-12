from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))

from high_precision_v2 import (
    QUALITY_FEATURES,
    REVERSAL_FEATURES,
    TREND_FEATURES,
    choose_ensemble,
    combine_probabilities,
    fit_submodels,
    predict_submodels,
)


def rows() -> pd.DataFrame:
    values = np.linspace(0.02, 0.98, 80)
    frame = pd.DataFrame({"date": pd.bdate_range("2025-01-02", periods=80)})
    all_features = TREND_FEATURES + REVERSAL_FEATURES + QUALITY_FEATURES
    for index, feature in enumerate(all_features):
        frame[feature] = np.clip(
            values + np.sin(np.arange(len(values)) + index) * 0.02,
            0.0,
            1.0,
        )
    frame["label_5"] = (
        frame[list(TREND_FEATURES)].mean(axis=1) > 0.52
    ).astype(int)
    frame["forward_5"] = np.where(frame["label_5"] == 1, 0.02, -0.01)
    return frame


class HighPrecisionV2ModelTests(unittest.TestCase):
    def test_submodels_and_predictions_are_reproducible(self) -> None:
        first = fit_submodels(rows(), l2=1.0)
        second = fit_submodels(rows(), l2=1.0)

        self.assertEqual(first, second)
        pd.testing.assert_frame_equal(
            predict_submodels(first, rows()),
            predict_submodels(second, rows()),
        )

    def test_combined_probability_is_exact_weighted_average(self) -> None:
        predictions = pd.DataFrame(
            {"trend": [0.8], "reversal": [0.4], "quality": [0.6]}
        )

        result = combine_probabilities(
            predictions,
            {"trend": 0.5, "reversal": 0.25, "quality": 0.25},
        )

        self.assertAlmostEqual(0.65, result[0])

    def test_choose_ensemble_is_deterministic_and_returns_calibrator(self) -> None:
        config = {
            "ensemble_weights": [
                {
                    "name": "trend",
                    "trend": 1.0,
                    "reversal": 0.0,
                    "quality": 0.0,
                },
                {
                    "name": "equal",
                    "trend": 1 / 3,
                    "reversal": 1 / 3,
                    "quality": 1 / 3,
                },
            ],
            "calibration_bins": 4,
            "beta_alpha": 1.0,
            "beta_beta": 1.0,
        }
        models = fit_submodels(rows().iloc[:60], l2=1.0)

        first = choose_ensemble(rows().iloc[60:], models, config)
        second = choose_ensemble(rows().iloc[60:], models, config)

        self.assertEqual(first, second)
        self.assertIn(first["name"], {"trend", "equal"})
        self.assertEqual(4, len(first["calibration"]))
        self.assertGreater(first["metrics"]["rows"], 0)


if __name__ == "__main__":
    unittest.main()
