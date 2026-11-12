from __future__ import annotations

import sys
import unittest
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))

from high_precision_picker import (
    HighPrecisionPolicy,
    build_parser,
    deployment_checks,
    evaluate_selected_trades,
    fit_final_model,
    parameter_grid,
    recalculate_sample_returns,
    run_walk_forward_search,
    select_daily_candidates,
)
from picker_backtest import build_symbol_panel


class HighPrecisionBacktestTests(unittest.TestCase):
    def policy(self, **changes) -> HighPrecisionPolicy:
        values = {
            "breadth_floor": 0.4,
            "momentum_floor": -0.02,
            "drawdown_floor": -0.25,
            "strong_threshold": 0.75,
            "normal_threshold_delta": 0.05,
            "max_picks": 3,
        }
        values.update(changes)
        return HighPrecisionPolicy(**values)

    @staticmethod
    def prediction(
        code: str,
        probability: float,
        *,
        industry: str = "银行",
        date: str = "2026-01-05",
        breadth: float = 0.42,
        momentum: float = 0.0,
        drawdown: float = -0.2,
        forward: float = 0.01,
        benchmark: float = 0.002,
    ) -> dict:
        return {
            "date": date,
            "code": code,
            "industry": industry,
            "calibrated_probability": probability,
            "breadth_above_ma20_raw": breadth,
            "market_median_mom20_raw": momentum,
            "market_median_drawdown60_raw": drawdown,
            "forward_5": forward,
            "benchmark_forward_5": benchmark,
        }

    def test_weak_market_forces_cash(self) -> None:
        frame = pd.DataFrame(
            [
                self.prediction(
                    "000001",
                    0.99,
                    breadth=0.2,
                    momentum=0.05,
                    drawdown=-0.1,
                )
            ]
        )
        self.assertTrue(select_daily_candidates(frame, self.policy()).empty)

    def test_normal_market_uses_higher_threshold_and_max_three(self) -> None:
        rows = [
            self.prediction(
                f"{index + 1:06d}",
                probability,
                industry=f"行业{index}",
            )
            for index, probability in enumerate([0.84, 0.83, 0.82, 0.81])
        ]
        selected = select_daily_candidates(pd.DataFrame(rows), self.policy())
        self.assertEqual(3, len(selected))
        self.assertTrue((selected["probability_threshold"] == 0.80).all())

    def test_industry_deduplication_keeps_highest_probability(self) -> None:
        frame = pd.DataFrame(
            [
                self.prediction(
                    "000001",
                    0.91,
                    industry="软件",
                    breadth=0.6,
                    momentum=0.2,
                    drawdown=-0.05,
                ),
                self.prediction(
                    "000002",
                    0.89,
                    industry="软件",
                    breadth=0.6,
                    momentum=0.2,
                    drawdown=-0.05,
                ),
            ]
        )
        selected = select_daily_candidates(frame, self.policy())
        self.assertEqual(["000001"], selected["code"].tolist())

    def test_metrics_use_trade_wins_active_dates_benchmark_and_folds(self) -> None:
        scheduled = list(pd.bdate_range("2026-01-05", periods=20))
        rows = []
        active_dates = scheduled[:4]
        for date_index, signal_date in enumerate(active_dates):
            for stock_index in range(2):
                net = 0.02 if (date_index + stock_index) % 3 else -0.01
                rows.append(
                    self.prediction(
                        f"{date_index}{stock_index}".zfill(6),
                        0.9,
                        date=signal_date.date().isoformat(),
                        industry=f"行业{stock_index}",
                        forward=net,
                        benchmark=0.001,
                    )
                )
        folds = [
            {
                "valid_start": scheduled[0].date().isoformat(),
                "valid_end": scheduled[9].date().isoformat(),
            },
            {
                "valid_start": scheduled[10].date().isoformat(),
                "valid_end": scheduled[-1].date().isoformat(),
            },
        ]
        metrics = evaluate_selected_trades(
            pd.DataFrame(rows),
            scheduled,
            folds,
        )
        self.assertEqual(8, metrics["trades"])
        self.assertAlmostEqual(0.2, metrics["activity_rate"])
        self.assertEqual([4, 0], metrics["fold_active_dates"])
        self.assertAlmostEqual(5 / 8, metrics["win_rate"])
        self.assertAlmostEqual(
            pd.Series([row["forward_5"] - 0.001 for row in rows]).mean(),
            metrics["mean_excess_return"],
        )

    def test_deployment_rejects_oos_win_rate_below_target(self) -> None:
        config = {
            "minimum_win_rate": 0.85,
            "minimum_activity_rate": 0.2,
            "minimum_trades": 50,
            "minimum_active_dates_per_fold": 5,
            "minimum_fold_win_rate": 0.65,
            "minimum_majority_fold_win_rate": 0.8,
            "minimum_majority_fold_count": 3,
            "minimum_mean_net_return": 0.0,
            "minimum_mean_excess_return": 0.0,
            "maximum_drawdown": 0.12,
            "minimum_positive_neighbors": 2,
        }
        metrics = {
            "win_rate": 0.84,
            "activity_rate": 0.3,
            "trades": 60,
            "fold_active_dates": [8, 8, 8, 8, 8],
            "fold_win_rates": [0.9, 0.9, 0.8, 0.75, 0.65],
            "mean_net_return": 0.02,
            "mean_excess_return": 0.01,
            "max_drawdown": -0.05,
        }
        result = deployment_checks(
            metrics,
            [{"mean_net_return": 0.01}, {"mean_net_return": 0.005}],
            config,
        )
        self.assertFalse(result["passed"])
        self.assertIn("aggregate_win_rate", result["failed"])

    def test_parameter_grid_has_exact_finite_size(self) -> None:
        config = json.loads(
            (ROOT / "engine" / "high_precision_config.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(720, len(parameter_grid(config)))
        self.assertEqual(
            [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85],
            config["strong_thresholds"],
        )

    def test_optimizer_cli_accepts_required_artifact_paths(self) -> None:
        args = build_parser().parse_args(
            [
                "optimize",
                "--output",
                "report.json",
                "--candidate-model",
                "candidate.json",
                "--verification",
                "verification.json",
            ]
        )
        self.assertEqual("optimize", args.command)
        self.assertEqual(Path("report.json"), args.output)
        self.assertEqual(Path("candidate.json"), args.candidate_model)

    def test_walk_forward_separates_fit_calibration_and_validation(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=180)
        panels = []
        for stock_index in range(4):
            base = 10.0 + stock_index
            close = base + np.sin(np.arange(len(dates)) / (5 + stock_index))
            close += np.arange(len(dates)) * (0.002 + stock_index * 0.0005)
            bars = pd.DataFrame(
                {
                    "date": dates,
                    "open": close * 0.998,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 1_000_000 + np.arange(len(dates)) * 1000,
                    "amount": close
                    * (1_000_000 + np.arange(len(dates)) * 1000),
                }
            )
            panels.append(
                build_symbol_panel(
                    f"{stock_index + 1:06d}",
                    bars,
                    {
                        "name": f"股票{stock_index}",
                        "industry": f"行业{stock_index}",
                        "concept": "AI" if stock_index == 0 else "",
                    },
                )
            )
        panel = pd.concat(panels, ignore_index=True)
        panel["above_ma20"] = (panel["close"] > panel["ma_20"]).astype(float)
        grouped = panel.groupby("date")
        panel["breadth_above_ma20"] = grouped["above_ma20"].transform("mean")
        panel["market_median_mom20"] = grouped["mom_20"].transform("median")
        panel["market_median_drawdown60"] = grouped[
            "drawdown_60"
        ].transform("median")
        benchmark_close = 4000 + np.arange(len(dates)) * 2.0
        benchmark = pd.DataFrame(
            {
                "date": dates,
                "code": "000300",
                "open": benchmark_close * 0.999,
                "close": benchmark_close,
            }
        )
        profiles = json.loads(
            (ROOT / "engine" / "picker_factor_profiles.json").read_text(
                encoding="utf-8"
            )
        )
        config = json.loads(
            (ROOT / "engine" / "high_precision_config.json").read_text(
                encoding="utf-8"
            )
        )
        config.update(
            {
                "minimum_training_months": 0,
                "calibration_trading_days": 20,
                "calibration_minimum_activity_rate": 0.0,
                "calibration_minimum_trades": 0,
                "l2_values": [1.0],
                "strong_thresholds": [0.5],
                "normal_threshold_deltas": [0.0],
                "max_picks_values": [1],
                "market_policies": [
                    {
                        "breadth_floor": 0.0,
                        "momentum_floor": -1.0,
                        "drawdown_floor": -1.0,
                    }
                ],
            }
        )
        folds = [
            {
                "train_end": dates[109].date().isoformat(),
                "valid_start": dates[110].date().isoformat(),
                "valid_end": dates[149].date().isoformat(),
            }
        ]

        result = run_walk_forward_search(
            panel, profiles, config, benchmark, folds
        )

        fold = result["walk_forward"]["folds"][0]
        self.assertLess(fold["fit_max_date"], fold["calibration_min_date"])
        self.assertLess(
            fold["calibration_max_date"], fold["validation_min_date"]
        )
        self.assertEqual(1, result["attempt_count"])
        self.assertEqual(1, len(result["attempts"]))
        self.assertEqual(
            fold["validation_metrics"]["trades"],
            result["aggregate_metrics"]["trades"],
        )

    def test_final_model_contains_live_runtime_contract(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=80)
        close = 10 + np.sin(np.arange(len(dates)) / 4) + np.arange(len(dates)) * 0.01
        bars = pd.DataFrame(
            {
                "date": dates,
                "open": close * 0.998,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1_000_000,
                "amount": close * 1_000_000,
            }
        )
        panel = build_symbol_panel(
            "000001",
            bars,
            {"name": "测试股", "industry": "软件", "concept": "AI"},
        )
        profiles = json.loads(
            (ROOT / "engine" / "picker_factor_profiles.json").read_text(
                encoding="utf-8"
            )
        )
        config = json.loads(
            (ROOT / "engine" / "high_precision_config.json").read_text(
                encoding="utf-8"
            )
        )
        config["calibration_trading_days"] = 10
        chosen = {
            "l2": 1.0,
            "policy": self.policy(
                breadth_floor=0.0,
                momentum_floor=-1.0,
                drawdown_floor=-1.0,
            ).to_dict(),
        }

        model = fit_final_model(
            panel,
            profiles,
            config,
            chosen,
            run_id="test-run",
            deployment={"passed": False, "checks": {}, "failed": []},
        )

        self.assertEqual(
            {
                "schema_version",
                "run_id",
                "feature_names",
                "model",
                "market_statistics",
                "calibration",
                "policy",
                "deployment",
            },
            set(model),
        )

    def test_sample_return_recalculation_matches_stored_evidence(self) -> None:
        dates = pd.bdate_range("2026-01-05", periods=7)
        panel = pd.DataFrame(
            {
                "date": dates,
                "code": ["000001"] * 7,
                "open": [10.0, 10.0, 10.1, 10.2, 10.3, 10.4, 10.5],
                "close": [10.0, 10.1, 10.2, 10.3, 10.4, 11.0, 11.2],
            }
        )
        benchmark = pd.DataFrame(
            {
                "date": dates,
                "code": ["000300"] * 7,
                "open": [100.0] * 7,
                "close": [100.0, 100.1, 100.2, 100.3, 100.4, 101.0, 101.2],
            }
        )
        stored_return = 11.0 / 10.0 - 1.0 - 0.003
        stored_benchmark = 101.0 / 100.0 - 1.0 - 0.003
        report = {
            "walk_forward": {
                "folds": [
                    {
                        "validation_selections": [
                            {
                                "date": dates[0].date().isoformat(),
                                "code": "000001",
                                "forward_5": stored_return,
                                "benchmark_forward_5": stored_benchmark,
                            }
                        ]
                    }
                ]
            }
        }

        result = recalculate_sample_returns(
            report, panel, benchmark, sample_size=10, seed=20260717
        )

        self.assertTrue(result["passed"])
        self.assertEqual(1, result["sample_count"])
        self.assertLess(result["samples"][0]["return_abs_diff"], 1e-12)


if __name__ == "__main__":
    unittest.main()
