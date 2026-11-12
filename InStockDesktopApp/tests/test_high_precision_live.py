from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))

from high_precision_picker_data import (
    _display_row,
    high_precision_composite_score,
    _load_live_v3_features,
    build_live_payload,
    load_runtime_artifacts,
    predict_v2_live,
    predict_v3_live,
    predict_v4_live,
    predict_v5_live,
)
from high_precision_v2 import QUALITY_FEATURES, REVERSAL_FEATURES, TREND_FEATURES


class HighPrecisionLiveTests(unittest.TestCase):
    def test_shareholder_quality_has_fifteen_percent_high_precision_weight(self) -> None:
        self.assertEqual(high_precision_composite_score(0.80, 15.0), 83.0)
        self.assertEqual(high_precision_composite_score(0.80, 0.0), 75.5)
        self.assertEqual(high_precision_composite_score(0.80, -15.0), 68.0)

    def test_display_row_serializes_database_order_flow_date(self) -> None:
        row = _display_row(
            {
                "code": "600900",
                "fund_component_date": date(2026, 7, 30),
                "fund_amount_super": 130_012_240,
                "fund_amount_large": 506_381_024,
                "fund_amount_medium": -353_628_016,
                "fund_amount_small": -282_765_216,
                "fund_amount": 636_393_264,
                "fund_amount_3": 800_000_000,
                "fund_amount_5": 1_000_000_000,
            },
            probability=0.8,
            regime="normal",
            threshold=0.75,
            positive="",
            negative="",
            atr=None,
        )

        serialized = json.dumps(row, ensure_ascii=False)

        self.assertIn('"订单资金日期": "2026-07-30"', serialized)
        self.assertEqual(row["今日大单净额_万元"], 63639.33)
        self.assertEqual(row["今日散户净额_万元"], -63639.32)
        self.assertEqual(row["3日主力净流入_万元"], 80000.0)
        self.assertEqual(row["5日主力净流入_万元"], 100000.0)

    def test_v5_runtime_uses_fitted_relative_score_and_five_three_one_rule(self) -> None:
        model = json.loads(
            (ROOT / "engine" / "high_precision_model.json").read_text(encoding="utf-8")
        )
        features = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-07-15"] * 6),
                "code": [f"{index:06d}" for index in range(1, 7)],
                "mom_5": [0.001 * index for index in range(1, 7)],
                "mom_20": [0.002 * index for index in range(1, 7)],
                "mom_60": [0.003 * index for index in range(1, 7)],
                "trend_20_60": [0.001 * index for index in range(1, 7)],
                "drawdown_60": [-0.20 + 0.01 * index for index in range(1, 7)],
                "volatility_20": [0.08 - 0.005 * index for index in range(1, 7)],
                "liquidity_20": [1_000_000 * index for index in range(1, 7)],
                "overheat": [0.0] * 6,
                "breadth_above_ma20": [0.20] * 6,
                "market_median_mom20": [-0.03] * 6,
                "market_median_drawdown60": [-0.30] * 6,
            }
        )

        result = predict_v5_live(features, model)

        self.assertEqual(1, len(result))
        self.assertEqual({"poor"}, set(result["market_regime"]))
        self.assertEqual({1}, set(result["target_count"]))
        self.assertIn("v5_score", result)

    def test_v5_payload_exposes_distinct_probabilities_for_all_live_candidates(self) -> None:
        config = json.loads(
            (ROOT / "engine" / "high_precision_v5_config.json").read_text(encoding="utf-8")
        )
        profiles = json.loads(
            (ROOT / "engine" / "picker_factor_profiles.json").read_text(encoding="utf-8")
        )
        snapshot = {
            "selection_date": date(2026, 7, 17),
            "spot_date": date(2026, 7, 17),
            "fund_date": date(2026, 7, 17),
            "rows": [
                {
                    "code": f"{index:06d}",
                    "name": f"S{index}",
                    "industry": "I",
                    "concept": "C",
                    "new_price": 10.0,
                    "change_rate": 0.0,
                    "deal_amount": 1_000_000,
                }
                for index in range(1, 7)
            ],
        }
        features = pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-07-17")] * 6,
                "code": [f"{index:06d}" for index in range(1, 7)],
                "mom_5": [0.001 * index for index in range(1, 7)],
                "mom_20": [0.01 * index for index in range(1, 7)],
                "mom_60": [0.02 * index for index in range(1, 7)],
                "trend_20_60": [0.005 * index for index in range(1, 7)],
                "drawdown_60": [-0.2 + 0.01 * index for index in range(1, 7)],
                "volatility_20": [0.08 - 0.005 * index for index in range(1, 7)],
                "liquidity_20": [1_000_000 * index for index in range(1, 7)],
                "overheat": [0.0] * 6,
                "atr_14": [0.5] * 6,
                "breadth_above_ma20": [0.2] * 6,
                "market_median_mom20": [-0.03] * 6,
                "market_median_drawdown60": [-0.3] * 6,
            }
        )
        calibration = [
            {"lower": 0.0, "upper": 0.5, "count": 20, "wins": 6, "probability": 0.3},
            {"lower": 0.5, "upper": 1.0, "count": 20, "wins": 16, "probability": 0.8},
        ]
        model = {
            "schema_version": "high-precision-v5",
            "run_id": "v5-probability-test",
            "rule": config["rule"],
            "feature_names": ["mom_20_rank"],
            "regime_models": {
                regime: {"coefficients": [1.0]} for regime in ("good", "normal", "poor")
            },
            "probability_model": {
                "feature_names": ["mom_20_rank"],
                "regimes": {
                    regime: {
                        "model": {"intercept": -5.0, "coef": [10.0]},
                        "calibration": calibration,
                    }
                    for regime in ("good", "normal", "poor")
                },
            },
            "trade_policy": {
                "take_profit_pct": 0.05,
                "stop_loss_pct": 0.05,
                "position_fraction": 0.04,
            },
        }

        payload = build_live_payload(
            snapshot, {}, config, model, profiles, {}, v2_features=features
        )

        all_rows = payload["sheets"]["全部概率候选"]
        self.assertEqual(6, len(all_rows))
        self.assertGreater(len({row["5日上涨概率"] for row in all_rows}), 1)
        selected_codes = {row["代码"] for row in payload["sheets"]["高胜率精选"]}
        self.assertEqual(1, len(selected_codes))

    def test_live_loader_falls_back_to_instock_gzip_history(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            daily_cache = root / "daily"
            history_cache = root / "hist"
            daily_cache.mkdir()
            selection_date = pd.Timestamp("2026-07-17")
            date_start = (selection_date - pd.Timedelta(days=365 * 3)).strftime("%Y%m%d")
            history_dir = history_cache / date_start[:6] / date_start
            history_dir.mkdir(parents=True)
            dates = pd.bdate_range("2026-03-30", periods=80)
            close = pd.Series([10.0 * 1.002**index for index in range(len(dates))])
            pd.DataFrame(
                {
                    "date": dates,
                    "open": close * 0.999,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 1_000_000,
                    "amount": close * 1_000_000,
                }
            ).to_pickle(
                history_dir / "601016qfq.gzip.pickle", compression="gzip"
            )
            snapshot = {
                "selection_date": selection_date.date(),
                "rows": [
                    {"code": "601016", "name": "节能风电", "industry": "电力", "concept": "风能"}
                ],
            }
            from unittest.mock import patch

            with patch("high_precision_picker_data.DAILY_BARS_CACHE", daily_cache), patch(
                "high_precision_picker_data.INSTOCK_HISTORY_CACHE", history_cache
            ):
                result = _load_live_v3_features(snapshot)

        self.assertEqual(["601016"], result["code"].tolist())

    def test_live_loader_fetches_only_missing_history_from_secondary_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            daily_cache = root / "daily"
            history_cache = root / "hist"
            daily_cache.mkdir()
            history_cache.mkdir()
            dates = pd.bdate_range(end="2026-07-17", periods=80)
            close = pd.Series([10.0 * 1.001**index for index in range(len(dates))])
            fetched = pd.DataFrame(
                {
                    "date": dates,
                    "open": close * 0.999,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 1_000_000,
                    "amount": close * 1_000_000,
                }
            )
            snapshot = {
                "selection_date": date(2026, 7, 17),
                "rows": [
                    {"code": "000683", "name": "博源化工", "industry": "化学原料"}
                ],
            }
            from unittest.mock import patch

            with patch("high_precision_picker_data.DAILY_BARS_CACHE", daily_cache), patch(
                "high_precision_picker_data.INSTOCK_HISTORY_CACHE", history_cache
            ), patch(
                "high_precision_picker_data._fetch_sina_daily_bars",
                return_value=fetched,
            ) as fetch:
                result = _load_live_v3_features(snapshot)

        fetch.assert_called_once_with("000683", pd.Timestamp("2026-07-17"))
        self.assertEqual(["000683"], result["code"].tolist())

    def test_v4_payload_exposes_target_count_and_fixed_policy(self) -> None:
        config = json.loads(
            (ROOT / "engine" / "high_precision_v4_config.json").read_text(encoding="utf-8")
        )
        profiles = json.loads(
            (ROOT / "engine" / "picker_factor_profiles.json").read_text(encoding="utf-8")
        )
        snapshot = {
            "selection_date": date(2026, 7, 13),
            "spot_date": date(2026, 7, 13),
            "fund_date": date(2026, 7, 13),
            "rows": [
                {
                    "code": f"{index:06d}", "name": f"S{index}",
                    "industry": "I", "concept": "C", "new_price": 10.0,
                    "change_rate": 0.0, "deal_amount": 1_000_000,
                }
                for index in range(1, 7)
            ],
        }
        features = pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-07-13")] * 6,
                "code": [f"{index:06d}" for index in range(1, 7)],
                "mom_5": [0.01] * 6,
                "mom_20": [0.01 * index for index in range(1, 7)],
                "mom_60": [0.02 * index for index in range(1, 7)],
                "trend_20_60": [0.005 * index for index in range(1, 7)],
                "overheat": [0.0] * 6,
                "drawdown_60": [-0.10] * 6,
                "volatility_20": [0.03 - index * 0.001 for index in range(1, 7)],
                "atr_14": [0.5] * 6,
                "breadth_above_ma20": [0.60] * 6,
                "market_median_mom20": [0.04] * 6,
                "market_median_drawdown60": [-0.10] * 6,
            }
        )
        model = {
            "schema_version": "high-precision-v4",
            "run_id": "v4-test",
            "historical_win_rate": 0.719,
            "rule": config["rule"],
            "trade_policy": {
                "take_profit_pct": 0.03,
                "stop_loss_pct": 0.09,
                "position_fraction": 0.04,
            },
        }

        payload = build_live_payload(
            snapshot, {}, config, model, profiles, {}, v2_features=features
        )

        self.assertEqual(5, payload["summary"]["target_count"])
        self.assertEqual(5, len(payload["sheets"][payload["top_sheet_name"]]))
        self.assertEqual("high-precision-5d-v4", payload["trade_policy"]["policy_version"])
        self.assertEqual(0.03, payload["trade_policy"]["fixed_take_profit_pct"])
        self.assertEqual(0.09, payload["trade_policy"]["fixed_stop_loss_pct"])
        self.assertEqual(0.04, payload["trade_policy"]["position_fraction"])

    def test_v4_runtime_returns_five_three_one_by_regime(self) -> None:
        model = {
            "schema_version": "high-precision-v4",
            "historical_win_rate": 0.719,
            "rule": {
                "good": {
                    "breadth_floor": 0.50, "momentum_floor": 0.02,
                    "drawdown_floor": -0.15, "target_count": 5,
                    "score": "pullback_score",
                },
                "normal": {
                    "breadth_floor": 0.45, "momentum_floor": 0.005,
                    "drawdown_floor": -0.22, "target_count": 3,
                    "score": "low_vol_trend_score",
                },
                "poor": {"target_count": 1, "score": "rebound_score"},
            },
        }

        for market, expected_regime, expected_count in (
            ((0.60, 0.04, -0.10), "good", 5),
            ((0.47, 0.01, -0.18), "normal", 3),
            ((0.20, -0.03, -0.30), "poor", 1),
        ):
            breadth, momentum, drawdown = market
            features = pd.DataFrame(
                {
                    "date": [pd.Timestamp("2026-07-13")] * 6,
                    "code": [f"{index:06d}" for index in range(1, 7)],
                    "mom_5": [0.01] * 6,
                    "mom_20": [0.01 * index for index in range(1, 7)],
                    "mom_60": [0.02 * index for index in range(1, 7)],
                    "trend_20_60": [0.005 * index for index in range(1, 7)],
                    "overheat": [0.0] * 6,
                    "drawdown_60": [-0.10] * 6,
                    "volatility_20": [0.03 - index * 0.001 for index in range(1, 7)],
                    "breadth_above_ma20": [breadth] * 6,
                    "market_median_mom20": [momentum] * 6,
                    "market_median_drawdown60": [drawdown] * 6,
                }
            )

            result = predict_v4_live(features, model)

            self.assertEqual(expected_count, len(result))
            self.assertEqual({expected_regime}, set(result["market_regime"]))
            self.assertEqual({expected_count}, set(result["target_count"]))
            self.assertTrue((result["calibrated_probability"] == 0.719).all())

    def test_v3_live_loader_builds_raw_backtest_factors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            dates = pd.bdate_range("2026-03-30", periods=80)
            for code, drift in (("000001", 0.002), ("000002", 0.001)):
                close = pd.Series(
                    [10.0 * (1.0 + drift) ** index for index in range(len(dates))]
                )
                pd.DataFrame(
                    {
                        "date": dates,
                        "open": close * 0.999,
                        "high": close * 1.01,
                        "low": close * 0.99,
                        "close": close,
                        "volume": 1_000_000,
                        "amount": close * 1_000_000,
                    }
                ).to_csv(cache / f"{code}.csv", index=False)
            snapshot = {
                "selection_date": dates[-1].date(),
                "rows": [
                    {"code": "000001", "name": "A", "industry": "I", "concept": "C"},
                    {"code": "000002", "name": "B", "industry": "I", "concept": "C"},
                ],
            }

            from unittest.mock import patch

            with patch("high_precision_picker_data.DAILY_BARS_CACHE", cache):
                result = _load_live_v3_features(snapshot)

        self.assertEqual(2, len(result))
        for column in (
            "mom_5",
            "mom_60",
            "trend_20_60",
            "overheat",
            "breadth_above_ma20",
            "market_median_mom20",
            "market_median_drawdown60",
        ):
            self.assertIn(column, result.columns)
            self.assertTrue(result[column].notna().all())
        self.assertGreater(result.loc[result["code"] == "000001", "mom_60"].iloc[0], 0.0)

    def test_v3_runtime_applies_fixed_pullback_rule(self) -> None:
        features = pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-07-13")] * 2,
                "code": ["000002", "000001"],
                "industry": ["软件", "银行"],
                "mom_60": [0.2, 0.3],
                "trend_20_60": [0.1, 0.2],
                "mom_5": [0.03, -0.02],
                "overheat": [0.2, -0.1],
                "breadth_above_ma20": [0.6, 0.6],
                "market_median_mom20": [0.03, 0.03],
                "market_median_drawdown60": [-0.1, -0.1],
            }
        )
        model = {
            "schema_version": "high-precision-v3",
            "historical_win_rate": 0.716,
            "rule": {
                "breadth_floor": 0.5,
                "momentum_floor": 0.02,
                "drawdown_floor": -0.15,
                "max_picks": 1,
            },
        }

        result = predict_v3_live(features, model)

        self.assertEqual(["000001"], result["code"].tolist())
        self.assertAlmostEqual(0.716, result.iloc[0]["calibrated_probability"])

    def test_v2_runtime_predicts_ensemble_probability(self) -> None:
        features = pd.DataFrame(
            {
                **{
                    name: [0.5]
                    for name in (
                        TREND_FEATURES + REVERSAL_FEATURES + QUALITY_FEATURES
                    )
                },
                "date": [pd.Timestamp("2026-07-13")],
                "code": ["000001"],
                "industry": ["软件"],
                "breadth_above_ma20_raw": [0.7],
                "market_median_mom20_raw": [0.2],
                "market_median_drawdown60_raw": [-0.05],
            }
        )
        model = {
            "schema_version": "high-precision-v2",
            "submodels": {
                name: {
                    "feature_names": list(feature_names),
                    "model": {
                        "intercept": 10.0,
                        "coef": [0.0] * len(feature_names),
                    },
                }
                for name, feature_names in (
                    ("trend", TREND_FEATURES),
                    ("reversal", REVERSAL_FEATURES),
                    ("quality", QUALITY_FEATURES),
                )
            },
            "ensemble": {
                "name": "equal",
                "weights": {
                    "trend": 1 / 3,
                    "reversal": 1 / 3,
                    "quality": 1 / 3,
                },
                "calibration": [
                    {
                        "lower": 0.0,
                        "upper": 1.0,
                        "count": 10,
                        "wins": 9,
                        "probability": 0.9,
                    }
                ],
            },
        }

        result = predict_v2_live(features, model)

        self.assertAlmostEqual(0.9, result.iloc[0]["calibrated_probability"])
        self.assertGreater(result.iloc[0]["trend_probability"], 0.99)

    def test_runtime_artifacts_require_matching_passed_run_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_path = root / "model.json"
            report_path = root / "report.json"
            model_path.write_text(
                json.dumps(
                    {
                        "run_id": "model-run",
                        "deployment": {"passed": True},
                    }
                ),
                encoding="utf-8",
            )
            report_path.write_text(
                json.dumps(
                    {
                        "run_id": "report-run",
                        "deployment": {"passed": True},
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "运行编号不一致"):
                load_runtime_artifacts(model_path, report_path)

    def test_weak_market_returns_valid_empty_payload_with_reason(self) -> None:
        config = json.loads(
            (ROOT / "engine" / "high_precision_config.json").read_text(
                encoding="utf-8"
            )
        )
        profiles = json.loads(
            (ROOT / "engine" / "picker_factor_profiles.json").read_text(
                encoding="utf-8"
            )
        )
        snapshot = {
            "selection_date": date(2026, 7, 13),
            "spot_date": date(2026, 7, 13),
            "fund_date": date(2026, 7, 13),
            "spot_buy": set(),
            "indicator_buy": set(),
            "indicator_sell": set(),
            "strategy_hits": {},
            "rows": [
                {
                    "date": date(2026, 7, 13),
                    "code": "000001",
                    "name": "测试股",
                    "industry": "软件",
                    "concept": "AI",
                    "style": "",
                    "new_price": 10.0,
                    "change_rate": 1.0,
                    "deal_amount": 10000000,
                }
            ],
        }
        historical = {
            "000001": {
                **{name: 0.1 for name in config["stock_rank_features"]},
                "atr_14": 0.5,
                "breadth_above_ma20": 0.1,
                "market_median_mom20": -0.2,
                "market_median_drawdown60": -0.4,
            }
        }
        model = {
            "run_id": "run-1",
            "feature_names": config["features"],
            "model": {
                "intercept": 10.0,
                "coef": [0.0] * len(config["features"]),
            },
            "market_statistics": {
                "breadth_above_ma20": {"median": 0.5, "scale": 0.1},
                "market_median_mom20": {"median": 0.0, "scale": 0.1},
                "market_median_drawdown60": {
                    "median": -0.1,
                    "scale": 0.1,
                },
            },
            "calibration": [
                {
                    "lower": 0.0,
                    "upper": 1.0,
                    "count": 10,
                    "wins": 9,
                    "probability": 0.9,
                }
            ],
            "policy": {
                "breadth_floor": 0.4,
                "momentum_floor": -0.02,
                "drawdown_floor": -0.25,
                "strong_threshold": 0.75,
                "normal_threshold_delta": 0.05,
                "max_picks": 3,
            },
            "deployment": {"passed": True},
        }

        payload = build_live_payload(
            snapshot,
            historical,
            config,
            model,
            profiles,
            {"reject_name_keywords": ["ST", "退"]},
        )

        self.assertEqual([], payload["sheets"]["高胜率精选"])
        self.assertEqual(
            "市场处于弱势，高胜率模式保持空仓",
            payload["summary"]["abstain_reason"],
        )

    def test_selected_rows_contain_probability_regime_and_explanations(self) -> None:
        config = json.loads(
            (ROOT / "engine" / "high_precision_config.json").read_text(
                encoding="utf-8"
            )
        )
        profiles = json.loads(
            (ROOT / "engine" / "picker_factor_profiles.json").read_text(
                encoding="utf-8"
            )
        )
        snapshot = {
            "selection_date": date(2026, 7, 13),
            "spot_date": date(2026, 7, 13),
            "fund_date": date(2026, 7, 13),
            "spot_buy": set(),
            "indicator_buy": set(),
            "indicator_sell": set(),
            "strategy_hits": {},
            "rows": [
                {
                    "date": date(2026, 7, 13),
                    "code": "000001",
                    "name": "测试股",
                    "industry": "软件",
                    "concept": "AI",
                    "style": "",
                    "new_price": 10.0,
                    "change_rate": 1.0,
                    "deal_amount": 10000000,
                }
            ],
        }
        historical = {
            "000001": {
                **{name: 0.1 for name in config["stock_rank_features"]},
                "atr_14": 0.5,
                "breadth_above_ma20": 0.7,
                "market_median_mom20": 0.2,
                "market_median_drawdown60": -0.05,
            }
        }
        model = {
            "run_id": "run-1",
            "feature_names": config["features"],
            "model": {
                "intercept": 10.0,
                "coef": [0.1] * len(config["features"]),
            },
            "market_statistics": {
                "breadth_above_ma20": {"median": 0.5, "scale": 0.1},
                "market_median_mom20": {"median": 0.0, "scale": 0.1},
                "market_median_drawdown60": {
                    "median": -0.1,
                    "scale": 0.1,
                },
            },
            "calibration": [
                {
                    "lower": 0.0,
                    "upper": 1.0,
                    "count": 10,
                    "wins": 9,
                    "probability": 0.9,
                }
            ],
            "policy": {
                "breadth_floor": 0.4,
                "momentum_floor": -0.02,
                "drawdown_floor": -0.25,
                "strong_threshold": 0.75,
                "normal_threshold_delta": 0.05,
                "max_picks": 3,
            },
            "deployment": {"passed": True},
        }

        payload = build_live_payload(
            snapshot,
            historical,
            config,
            model,
            profiles,
            {"reject_name_keywords": ["ST", "退"]},
        )

        row = payload["sheets"]["高胜率精选"][0]
        for key in (
            "5日上涨概率",
            "市场状态",
            "概率门槛",
            "主要正向因子",
            "主要负向因子",
            "通过高胜率门槛",
        ):
            self.assertIn(key, row)


if __name__ == "__main__":
    unittest.main()
