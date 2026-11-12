from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from workers.sector_data import SectorDataBundle, SectorDataUnavailable
from workers.sector_forecast_core import build_feature_panel
from workers.sector_forecast_worker import MODEL_VERSION, run_sector_forecast
from workers.worker_cli import build_parser


def make_bundle(sector_type: str, days: int = 620, sectors: int = 12) -> SectorDataBundle:
    dates = pd.bdate_range("2023-01-02", periods=days)
    rows: list[dict[str, object]] = []
    signals: list[dict[str, object]] = []
    for sector in range(sectors):
        sector_id = f"{sector_type[:1].upper()}K{sector:04d}"
        sector_name = f"{sector_type}板块{sector}"
        for index, day in enumerate(dates):
            close = 100 * (1.001 + sector * 0.0001) ** index
            rows.append(
                {
                    "sector_type": sector_type,
                    "sector_id": sector_id,
                    "sector_name": sector_name,
                    "date": day,
                    "open": close * 0.998,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 1000,
                    "amount": close * 1000 * (1 + index / 10000),
                    "turnover": 2 + sector / 100,
                    "benchmark_close": 3000 * 1.0005**index,
                }
            )
        signals.append(
            {
                "sector_type": sector_type,
                "sector_id": sector_id,
                "sector_name": sector_name,
                "fund_score": sector / 10,
                "breadth_score": 0.5 + sector / 100,
                "heat_score": sector / 20,
                "up_ratio": 0.6,
                "current_data_completeness": 1.0,
            }
        )
    quality = {
        "valid_days": days,
        "latest_date": dates[-1].date().isoformat(),
        "board_count": sectors,
        "expected_board_count": sectors,
        "coverage_ratio": 1.0,
        "using_stale_cache": False,
        "missing_current_signal_ratio": 0.0,
        "runnable": True,
        "errors": [],
    }
    return SectorDataBundle(pd.DataFrame(rows), pd.DataFrame(signals), dates[-1], quality)


def passing_validation(_panel: pd.DataFrame) -> dict[str, object]:
    return {
        "passed": True,
        "quality": 0.9,
        "reasons": [],
        "folds": [],
        "model_metrics": {"top10_mean_excess": 0.01, "rank_ic": 0.1},
        "baseline_metrics": {"top10_mean_excess": 0.0, "rank_ic": 0.0},
    }


def failing_validation(_panel: pd.DataFrame) -> dict[str, object]:
    return {
        "passed": False,
        "quality": 0.2,
        "reasons": ["rank_ic"],
        "folds": [],
        "model_metrics": {"top10_mean_excess": -0.01, "rank_ic": -0.1},
        "baseline_metrics": {"top10_mean_excess": 0.0, "rank_ic": 0.0},
    }


def head_passing_validation(_panel: pd.DataFrame) -> dict[str, object]:
    validation = failing_validation(_panel)
    validation["head_passed"] = True
    validation["head_checks"] = {
        "top5_mean_excess": True,
        "rank_ic": True,
        "fold_stability": True,
    }
    return validation


class SectorForecastWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.paths = SimpleNamespace(data_dir=Path(self.temporary.name))
        self.events: list[tuple[str, str, int, dict | None]] = []

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def emit(self, kind: str, message: str, progress: int, payload: dict | None = None) -> None:
        self.events.append((kind, message, progress, payload))

    def test_result_describes_short_momentum_ranking_and_twenty_day_risk_horizon(self) -> None:
        result = run_sector_forecast(
            self.paths,
            self.emit,
            bundle_loader=lambda _paths, kind, refresh=False: make_bundle(kind),
            validator=passing_validation,
        )

        self.assertEqual(
            result.get("signal_horizons"),
            {
                "ranking_days": [3, 5, 10],
                "factor_weights": [0.45, 0.35, 0.20],
                "risk_days": [20],
            },
        )

    def test_derived_live_rows_are_ranked_but_excluded_from_validation(self) -> None:
        validation_inputs: list[bool] = []
        validation_targets: list[int] = []
        expected_targets: list[int] = []
        ranking_inputs: list[bool] = []

        def loader(_paths, sector_type: str, refresh: bool = False):
            bundle = make_bundle(sector_type)
            bundle.history["derived_latest"] = False
            latest = bundle.history["date"] == bundle.as_of
            bundle.history.loc[latest, "derived_latest"] = True
            expected_panel = build_feature_panel(
                bundle.history.loc[~bundle.history["derived_latest"]].copy()
            )
            expected_targets.append(
                int(expected_panel["target_excess_5"].notna().sum())
            )
            return bundle

        def validator(panel: pd.DataFrame) -> dict[str, object]:
            validation_inputs.append(bool(panel["derived_latest"].fillna(False).any()))
            validation_targets.append(int(panel["target_excess_5"].notna().sum()))
            return passing_validation(panel)

        def ranker(
            panel: pd.DataFrame, _validation: dict[str, object], _top_k: int
        ) -> list[dict[str, object]]:
            ranking_inputs.append(bool(panel["derived_latest"].fillna(False).any()))
            return []

        run_sector_forecast(
            self.paths,
            self.emit,
            bundle_loader=loader,
            validator=validator,
            ranker=ranker,
        )

        self.assertEqual([False, False], validation_inputs)
        self.assertEqual(expected_targets, validation_targets)
        self.assertEqual([True, True], ranking_inputs)

    def test_industry_can_publish_when_concept_is_unavailable(self) -> None:
        def loader(_paths, sector_type: str, refresh: bool = False):
            if sector_type == "concept":
                raise SectorDataUnavailable("concept offline")
            return make_bundle(sector_type)

        result = run_sector_forecast(
            self.paths, self.emit, bundle_loader=loader, validator=passing_validation
        )
        self.assertEqual(result["universes"]["industry"]["status"], "ready")
        self.assertEqual(result["universes"]["concept"]["status"], "unavailable")
        self.assertGreater(len(result["universes"]["industry"]["rows"]), 0)

    def test_failed_validation_publishes_factor_fallback(self) -> None:
        result = run_sector_forecast(
            self.paths,
            self.emit,
            bundle_loader=lambda _paths, kind, refresh=False: make_bundle(kind),
            validator=failing_validation,
        )
        concept = result["universes"]["concept"]
        self.assertEqual(concept["status"], "forced_fallback")
        self.assertEqual(len(concept["rows"]), 10)
        self.assertFalse(concept["validation"]["passed"])
        self.assertIn("参考榜单", concept["message"])

    def test_top_five_validation_publishes_head_ready_status(self) -> None:
        result = run_sector_forecast(
            self.paths,
            self.emit,
            bundle_loader=lambda _paths, kind, refresh=False: make_bundle(kind),
            validator=head_passing_validation,
        )
        concept = result["universes"]["concept"]
        self.assertEqual(concept["status"], "head_ready")
        self.assertIn("前5", concept["message"])

    def test_failed_validation_reaches_ranker(self) -> None:
        seen: list[bool] = []

        def ranker(_panel, validation, _top_k):
            seen.append(bool(validation["passed"]))
            return [{"rank": 1, "sector_id": "X", "sector_name": "测试"}]

        run_sector_forecast(
            self.paths,
            self.emit,
            bundle_loader=lambda _paths, kind, refresh=False: make_bundle(kind),
            validator=failing_validation,
            ranker=ranker,
        )
        self.assertEqual(seen, [False, False])

    def test_rerun_does_not_rewrite_snapshot(self) -> None:
        kwargs = {
            "bundle_loader": lambda _paths, kind, refresh=False: make_bundle(kind),
            "validator": passing_validation,
        }
        first = run_sector_forecast(self.paths, self.emit, **kwargs)
        snapshot = (
            self.paths.data_dir
            / "sector_forecast"
            / "history"
            / f"{first['as_of']}-{MODEL_VERSION}.json"
        )
        original = snapshot.read_bytes()
        run_sector_forecast(self.paths, self.emit, **kwargs)
        self.assertEqual(snapshot.read_bytes(), original)

    def test_cli_parser_accepts_sector_forecast(self) -> None:
        args = build_parser().parse_args(
            ["sector-forecast", "--result", "result.json", "--refresh-history"]
        )
        self.assertEqual(args.command, "sector-forecast")
        self.assertTrue(args.refresh_history)


if __name__ == "__main__":
    unittest.main()
