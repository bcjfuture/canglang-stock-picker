from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))

from picker_news import (  # noqa: E402
    NewsFactorResult,
    NewsItem,
    evaluate_news_for_codes,
    score_news_items,
    shortlist_codes,
)


class FixtureProvider:
    def __init__(self, items: list[NewsItem]) -> None:
        self.items = items
        self.calls = 0

    def fetch(self, code: str, start_date: date, end_date: date) -> list[NewsItem]:
        self.calls += 1
        return [item for item in self.items if item.code == code]


class FailingProvider:
    def fetch(self, code: str, start_date: date, end_date: date) -> list[NewsItem]:
        raise RuntimeError("provider offline")


class PickerNewsScoringTests(unittest.TestCase):
    def test_future_and_expired_items_cannot_affect_signal(self) -> None:
        """Catches look-ahead leakage and an inclusive window wider than seven days."""
        items = [
            NewsItem("000001", datetime(2026, 8, 13, 9), "公司回购股份", "公告", "announcement", "u1"),
            NewsItem("000001", datetime(2026, 8, 4, 9), "公司被立案调查", "新闻", "news", "u2"),
            NewsItem("000001", datetime(2026, 8, 12, 9), "公司中标重大合同", "公告", "announcement", "u3"),
        ]

        result = score_news_items(items, signal_date=date(2026, 8, 12))

        self.assertGreater(result.score, 0)
        self.assertEqual(result.count, 1)
        self.assertFalse(result.hard_risk)

    def test_announcement_wins_when_duplicate_title_appears_as_news(self) -> None:
        """Catches double-counting syndicated copies of the same event."""
        items = [
            NewsItem("000001", datetime(2026, 8, 12, 8), "公司拟回购股份", "媒体", "news", "n"),
            NewsItem("000001", datetime(2026, 8, 12, 9), "公司拟回购股份", "交易所", "announcement", "a"),
        ]

        result = score_news_items(items, signal_date=date(2026, 8, 12))

        self.assertEqual(result.count, 1)
        self.assertEqual(result.positive_count, 1)
        self.assertEqual(result.score, 4.0)

    def test_hard_risk_overrides_positive_items(self) -> None:
        """Catches a serious disclosure being hidden by simultaneous positive news."""
        items = [
            NewsItem("000001", datetime(2026, 8, 12, 9), "公司回购股份", "公告", "announcement", "p"),
            NewsItem("000001", datetime(2026, 8, 12, 10), "公司因重大违法被立案调查", "公告", "announcement", "n"),
        ]

        result = score_news_items(items, signal_date=date(2026, 8, 12))

        self.assertLessEqual(result.score, -20.0)
        self.assertTrue(result.hard_risk)
        self.assertEqual(result.negative_count, 1)

    def test_positive_and_negative_totals_are_asymmetrically_capped(self) -> None:
        """Catches positive headlines overwhelming the configured risk asymmetry."""
        positives = [
            NewsItem("000001", datetime(2026, 8, 12, hour), f"第{hour}项回购并中标重大合同", "公告", "announcement", str(hour))
            for hour in range(1, 9)
        ]
        negatives = [
            NewsItem("000002", datetime(2026, 8, 12, hour), f"第{hour}项亏损并被处罚", "公告", "announcement", str(hour))
            for hour in range(1, 9)
        ]

        positive_result = score_news_items(positives, signal_date=date(2026, 8, 12))
        negative_result = score_news_items(negatives, signal_date=date(2026, 8, 12))

        self.assertEqual(positive_result.score, 8.0)
        self.assertEqual(negative_result.score, -15.0)

    def test_equity_incentive_cancellation_is_not_a_buyback_positive(self) -> None:
        """Catches employee-share cancellation being mistaken for a shareholder buyback."""
        result = score_news_items(
            [
                NewsItem(
                    "000001",
                    datetime(2026, 8, 12, 9),
                    "关于限制性股票激励计划部分限制性股票回购注销实施公告",
                    "交易所",
                    "announcement",
                    "u",
                )
            ],
            signal_date=date(2026, 8, 12),
        )

        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.positive_count, 0)

    def test_explicit_denial_of_delisting_risk_is_not_a_hard_risk(self) -> None:
        """Catches a risk-denial announcement triggering the phrase it denies."""
        result = score_news_items(
            [
                NewsItem(
                    "000001",
                    datetime(2026, 8, 12, 9),
                    "公司不存在退市风险的澄清公告",
                    "交易所",
                    "announcement",
                    "u",
                )
            ],
            signal_date=date(2026, 8, 12),
        )

        self.assertFalse(result.hard_risk)
        self.assertEqual(result.score, 0.0)


class PickerNewsEvaluationTests(unittest.TestCase):
    def test_shortlist_covers_top_sixty_and_bonus_range(self) -> None:
        """Catches skipping a candidate that could still enter Top N after the capped bonus."""
        rows = [{"代码": f"{index:06d}", "综合评分": 100 - index} for index in range(80)]

        codes = shortlist_codes(
            rows,
            code_key="代码",
            score_key="综合评分",
            top_n=20,
            min_shortlist=60,
            positive_cap=8.0,
        )

        self.assertEqual(len(codes), 60)
        self.assertIn("000059", codes)
        self.assertNotIn("000060", codes)

    def test_cache_prevents_second_provider_call(self) -> None:
        """Catches a repeated backtest silently depending on a changed remote response."""
        provider = FixtureProvider(
            [NewsItem("000001", datetime(2026, 8, 12, 9), "公司回购股份", "公告", "announcement", "u")]
        )
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)
            first = evaluate_news_for_codes(
                ["000001"],
                signal_date=date(2026, 8, 12),
                provider=provider,
                cache_dir=cache_dir,
                config={},
            )
            second = evaluate_news_for_codes(
                ["000001"],
                signal_date=date(2026, 8, 12),
                provider=provider,
                cache_dir=cache_dir,
                config={},
            )

        self.assertEqual(asdict(first["000001"]), asdict(second["000001"]))
        self.assertEqual(provider.calls, 1)

    def test_provider_failure_is_neutral_and_visible(self) -> None:
        """Catches a remote outage either aborting the picker or masquerading as evaluated news."""
        with tempfile.TemporaryDirectory() as temporary:
            result = evaluate_news_for_codes(
                ["000001"],
                signal_date=date(2026, 8, 12),
                provider=FailingProvider(),
                cache_dir=Path(temporary),
                config={},
            )["000001"]

        self.assertEqual(result, NewsFactorResult.unavailable())
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.status, "资讯不可用")


if __name__ == "__main__":
    unittest.main()
