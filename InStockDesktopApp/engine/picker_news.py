from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol


POSITIVE_KEYWORDS: dict[str, float] = {
    "中标": 4.0,
    "回购": 4.0,
    "增持": 3.0,
    "预增": 4.0,
    "扭亏": 4.0,
    "重大合同": 4.0,
    "签订合同": 3.0,
    "获批": 3.0,
    "分红": 2.0,
}
NEGATIVE_KEYWORDS: dict[str, float] = {
    "减持": -4.0,
    "亏损": -5.0,
    "下修": -5.0,
    "终止": -4.0,
    "诉讼": -4.0,
    "风险提示": -3.0,
    "处罚": -5.0,
    "违规": -5.0,
}
HARD_RISK_KEYWORDS: tuple[str, ...] = (
    "立案调查",
    "退市风险",
    "重大违法",
    "债务逾期",
    "破产",
    "被执行",
)
POSITIVE_EXCLUSIONS: dict[str, tuple[str, ...]] = {
    "回购": ("回购注销", "限制性股票回购", "股权激励回购"),
}
HARD_RISK_NEGATIONS: tuple[str, ...] = (
    "不存在退市风险",
    "不涉及退市风险",
    "无退市风险",
    "撤销退市风险",
    "申请撤销退市风险",
    "解除被执行",
)
CACHE_VERSION = 2


@dataclass(frozen=True)
class NewsItem:
    code: str
    published_at: datetime
    title: str
    source: str
    category: str
    url: str = ""


@dataclass(frozen=True)
class NewsFactorResult:
    score: float
    status: str
    count: int
    positive_count: int
    negative_count: int
    hard_risk: bool
    summary: str

    @classmethod
    def unavailable(cls) -> "NewsFactorResult":
        return cls(0.0, "资讯不可用", 0, 0, 0, False, "资讯不可用（按0分处理）")

    @classmethod
    def not_evaluated(cls) -> "NewsFactorResult":
        return cls(0.0, "未评估", 0, 0, 0, False, "未进入新闻评估范围")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NewsFactorResult":
        return cls(
            score=float(value.get("score", 0.0)),
            status=str(value.get("status", "资讯不可用")),
            count=int(value.get("count", 0)),
            positive_count=int(value.get("positive_count", 0)),
            negative_count=int(value.get("negative_count", 0)),
            hard_risk=bool(value.get("hard_risk", False)),
            summary=str(value.get("summary", "")),
        )


class NewsProvider(Protocol):
    def fetch(self, code: str, start_date: date, end_date: date) -> list[NewsItem]: ...


def _normal_code(value: Any) -> str:
    text = str(value or "").strip()
    digits = "".join(character for character in text if character.isdigit())
    return digits[-6:].zfill(6) if digits else text


def _normal_title(value: Any) -> str:
    return re.sub(r"[\W_]+", "", str(value or "").strip(), flags=re.UNICODE).lower()


def _as_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, date):
        return datetime.combine(value, time.min)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("/", "-").replace("Z", "+00:00"))
        return parsed.replace(tzinfo=None)
    except (TypeError, ValueError):
        pass
    for pattern in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(text, pattern)
        except ValueError:
            continue
    return None


def _decay(age_days: int) -> float:
    if age_days <= 1:
        return 1.0
    if age_days <= 3:
        return 0.7
    return 0.4


def _eligible_items(
    items: Iterable[NewsItem], signal_date: date, lookback_days: int
) -> list[NewsItem]:
    end_at = datetime.combine(signal_date, time.max)
    start_at = datetime.combine(signal_date - timedelta(days=max(1, lookback_days) - 1), time.min)
    eligible = [item for item in items if start_at <= item.published_at <= end_at and item.title.strip()]
    eligible.sort(
        key=lambda item: (
            _normal_title(item.title),
            0 if item.category == "announcement" else 1,
            -item.published_at.timestamp(),
        )
    )
    deduplicated: dict[str, NewsItem] = {}
    for item in eligible:
        key = _normal_title(item.title)
        if key not in deduplicated:
            deduplicated[key] = item
    return sorted(deduplicated.values(), key=lambda item: item.published_at, reverse=True)


def score_news_items(
    items: Iterable[NewsItem],
    *,
    signal_date: date,
    lookback_days: int = 7,
    positive_cap: float = 8.0,
    negative_cap: float = -15.0,
    hard_risk_penalty: float = -20.0,
) -> NewsFactorResult:
    eligible = _eligible_items(items, signal_date, lookback_days)
    if not eligible:
        return NewsFactorResult(0.0, "无合格资讯", 0, 0, 0, False, "近7日无合格资讯")

    positive_total = 0.0
    negative_total = 0.0
    positive_count = 0
    negative_count = 0
    hard_risk = False
    classified: list[tuple[str, str]] = []

    for item in eligible:
        title = item.title.strip()
        age_days = max(0, (signal_date - item.published_at.date()).days)
        weight = _decay(age_days)
        negated_hard_risk = any(phrase in title for phrase in HARD_RISK_NEGATIONS)
        hard_matches = [
            keyword
            for keyword in HARD_RISK_KEYWORDS
            if keyword in title and not negated_hard_risk
        ]
        if hard_matches:
            hard_risk = True
            negative_count += 1
            negative_total += hard_risk_penalty * weight
            classified.append(("重大风险", title))
            continue

        positive_score = sum(
            value
            for keyword, value in POSITIVE_KEYWORDS.items()
            if keyword in title
            and not any(
                phrase in title for phrase in POSITIVE_EXCLUSIONS.get(keyword, ())
            )
        )
        negative_score = sum(value for keyword, value in NEGATIVE_KEYWORDS.items() if keyword in title)
        if negative_score < 0:
            negative_count += 1
            negative_total += negative_score * weight
            classified.append(("风险", title))
        elif positive_score > 0:
            positive_count += 1
            positive_total += positive_score * weight
            classified.append(("利好", title))

    positive_total = min(max(0.0, positive_total), max(0.0, float(positive_cap)))
    negative_floor = min(0.0, float(negative_cap))
    negative_total = max(negative_floor, negative_total)
    score = positive_total + negative_total
    if hard_risk:
        score = min(score, float(hard_risk_penalty))
    score = round(score, 1)

    if classified:
        summary = "；".join(f"{label}：{title}" for label, title in classified[:3])
        if len(classified) > 3:
            summary += f"；等{len(classified)}项"
    else:
        summary = "近7日有资讯，但未命中评分规则"
    return NewsFactorResult(
        score=score,
        status="已评估",
        count=len(eligible),
        positive_count=positive_count,
        negative_count=negative_count,
        hard_risk=hard_risk,
        summary=summary,
    )


def shortlist_codes(
    rows: Iterable[Mapping[str, Any]],
    *,
    code_key: str,
    score_key: str,
    top_n: int,
    min_shortlist: int = 60,
    positive_cap: float = 8.0,
) -> list[str]:
    ordered = sorted(
        rows,
        key=lambda row: float(row.get(score_key, float("-inf")) or float("-inf")),
        reverse=True,
    )
    if not ordered:
        return []
    cutoff_index = min(max(1, int(top_n)), len(ordered)) - 1
    cutoff = float(ordered[cutoff_index].get(score_key, 0.0) or 0.0)
    base_count = min(max(int(min_shortlist), int(top_n)), len(ordered))
    selected: list[str] = []
    seen: set[str] = set()
    for index, row in enumerate(ordered):
        score = float(row.get(score_key, 0.0) or 0.0)
        if index >= base_count and score < cutoff - float(positive_cap):
            break
        code = _normal_code(row.get(code_key))
        if code and code not in seen:
            selected.append(code)
            seen.add(code)
    return selected


def _cache_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "version": CACHE_VERSION,
        "lookback_days": int(config.get("lookback_days", 7)),
        "positive_cap": float(config.get("positive_cap", 8.0)),
        "negative_cap": float(config.get("negative_cap", -15.0)),
        "hard_risk_penalty": float(config.get("hard_risk_penalty", -20.0)),
    }


def _cache_path(cache_dir: Path, signal_date: date, code: str) -> Path:
    return cache_dir / "picker-news" / signal_date.isoformat() / f"{_normal_code(code)}.json"


def _read_cache(path: Path, settings: Mapping[str, Any]) -> NewsFactorResult | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("settings") != dict(settings):
            return None
        return NewsFactorResult.from_dict(payload["result"])
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _write_cache(path: Path, settings: Mapping[str, Any], result: NewsFactorResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(
            {"settings": dict(settings), "result": asdict(result)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def evaluate_news_for_codes(
    codes: Iterable[str],
    *,
    signal_date: date,
    provider: NewsProvider,
    cache_dir: Path,
    config: Mapping[str, Any],
) -> dict[str, NewsFactorResult]:
    settings = _cache_settings(config)
    normalized_codes = list(dict.fromkeys(_normal_code(code) for code in codes if _normal_code(code)))
    results: dict[str, NewsFactorResult] = {}
    missing: list[str] = []
    for code in normalized_codes:
        cached = _read_cache(_cache_path(cache_dir, signal_date, code), settings)
        if cached is None:
            missing.append(code)
        else:
            results[code] = cached

    start_date = signal_date - timedelta(days=settings["lookback_days"] - 1)

    def evaluate(code: str) -> tuple[str, NewsFactorResult, bool]:
        try:
            items = provider.fetch(code, start_date, signal_date)
            result = score_news_items(
                items,
                signal_date=signal_date,
                lookback_days=settings["lookback_days"],
                positive_cap=settings["positive_cap"],
                negative_cap=settings["negative_cap"],
                hard_risk_penalty=settings["hard_risk_penalty"],
            )
            return code, result, True
        except Exception:
            return code, NewsFactorResult.unavailable(), False

    max_workers = max(1, min(int(config.get("max_workers", 6)), len(missing) or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate, code) for code in missing]
        for future in as_completed(futures):
            code, result, cacheable = future.result()
            results[code] = result
            if cacheable:
                _write_cache(_cache_path(cache_dir, signal_date, code), settings, result)
    return results


def _row_value(row: Mapping[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in row:
            value = row.get(name)
            if value is not None and str(value).strip() not in {"", "nan", "NaT"}:
                return value
    return None


def _frame_items(frame: Any, *, code: str, category: str) -> list[NewsItem]:
    if frame is None or getattr(frame, "empty", True):
        return []
    items: list[NewsItem] = []
    for row in frame.to_dict("records"):
        published_at = _as_datetime(
            _row_value(row, ("发布时间", "公告日期", "日期", "publish_time", "date"))
        )
        title = str(_row_value(row, ("新闻标题", "公告标题", "标题", "title")) or "").strip()
        if published_at is None or not title:
            continue
        items.append(
            NewsItem(
                code=_normal_code(code),
                published_at=published_at,
                title=title,
                source=str(_row_value(row, ("文章来源", "公告类型", "来源", "source")) or ""),
                category=category,
                url=str(_row_value(row, ("新闻链接", "公告链接", "链接", "url")) or ""),
            )
        )
    return items


class AkshareNewsProvider:
    def __init__(self, akshare_module: Any | None = None) -> None:
        if akshare_module is None:
            import akshare as akshare_module  # type: ignore[no-redef]

        self.akshare = akshare_module

    def fetch(self, code: str, start_date: date, end_date: date) -> list[NewsItem]:
        items: list[NewsItem] = []
        failures = 0
        try:
            news = self.akshare.stock_news_em(symbol=_normal_code(code))
            items.extend(_frame_items(news, code=code, category="news"))
        except Exception:
            failures += 1
        try:
            notices = self.akshare.stock_individual_notice_report(
                security=_normal_code(code),
                symbol="全部",
                begin_date=start_date.strftime("%Y%m%d"),
                end_date=end_date.strftime("%Y%m%d"),
            )
            items.extend(_frame_items(notices, code=code, category="announcement"))
        except Exception:
            failures += 1
        if failures == 2:
            raise RuntimeError(f"news and announcement providers failed for {code}")
        return items
