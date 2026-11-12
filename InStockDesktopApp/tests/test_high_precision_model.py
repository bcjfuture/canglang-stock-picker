from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))

from high_precision_picker import (
    apply_probability_calibrator,
    fit_logistic,
    fit_probability_calibrator,
    predict_raw_probability,
    save_json_atomic,
)


class HighPrecisionModelTests(unittest.TestCase):
    def test_logistic_fit_is_deterministic_and_orders_examples(self) -> None:
        X = np.array(
            [
                [-2.0, -1.0],
                [-1.0, -0.5],
                [0.0, 0.0],
                [1.0, 0.5],
                [2.0, 1.0],
                [3.0, 1.5],
            ],
            dtype="float64",
        )
        y = np.array([0, 0, 0, 1, 1, 1], dtype="float64")

        first = fit_logistic(X, y, l2=1.0)
        second = fit_logistic(X, y, l2=1.0)
        probabilities = predict_raw_probability(first, X)

        np.testing.assert_allclose(first["coef"], second["coef"], atol=1e-12)
        self.assertAlmostEqual(first["intercept"], second["intercept"], places=12)
        self.assertTrue(np.all(np.diff(probabilities) > 0))
        self.assertTrue(np.all((probabilities > 0) & (probabilities < 1)))

    def test_beta_binomial_calibration_never_returns_unsmoothed_one(self) -> None:
        raw = np.array([0.91, 0.92, 0.93], dtype="float64")
        labels = np.array([1.0, 1.0, 1.0], dtype="float64")

        calibration = fit_probability_calibrator(
            raw, labels, bins=3, alpha=1.0, beta=1.0
        )
        calibrated = apply_probability_calibrator(
            np.array([0.925], dtype="float64"), calibration
        )

        self.assertGreater(calibrated[0], 0.5)
        self.assertLess(calibrated[0], 1.0)

    def test_json_writer_replaces_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.json"
            save_json_atomic(path, {"version": 1})
            save_json_atomic(path, {"version": 2})
            self.assertEqual(
                {"version": 2},
                json.loads(path.read_text(encoding="utf-8")),
            )


if __name__ == "__main__":
    unittest.main()
