from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any


def _number(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _code(row: dict[str, object]) -> str:
    return str(row.get("代码", "")).strip().zfill(6)


def _normalized_sector_name(value: object) -> str:
    text = re.sub(r"[\s、，,;；/]+", "", str(value or "").strip())
    return text[:-2] if text.endswith("概念") and len(text) > 2 else text


def sector_context_from_payload(payload: dict, trade_date: str) -> dict[str, object]:
    if str(payload.get("as_of", ""))[:10] != str(trade_date)[:10]:
        return {}
    result: dict[str, object] = {
        "trade_date": str(trade_date)[:10],
        "industries": {},
        "concepts": {},
    }
    universes = payload.get("universes", {})
    if not isinstance(universes, dict):
        return result
    for universe_key, target_key in (("industry", "industries"), ("concept", "concepts")):
        universe = universes.get(universe_key, {})
        if not isinstance(universe, dict):
            continue
        status = str(universe.get("status", ""))
        if status == "ready":
            weight, limit = 0.15, 10
        elif status == "head_ready":
            weight, limit = 0.15, 5
        elif status == "forced_fallback":
            weight, limit = 0.05, 10
        else:
            continue
        target = result[target_key]
        if not isinstance(target, dict):
            continue
        rows = universe.get("rows", [])
        if not isinstance(rows, list):
            continue
        for position, row in enumerate(rows[:limit], start=1):
            if not isinstance(row, dict):
                continue
            name = str(row.get("sector_name", "")).strip()
            normalized = _normalized_sector_name(name)
            if not normalized:
                continue
            rank = max(1, int(_number(row.get("rank"), position)))
            strength = max(0.5, min(1.0, _number(row.get("score"), 100 - 5 * (rank - 1)) / 100.0))
            item = {
                "name": name,
                "rank": rank,
                "strength": strength,
                "weight": weight,
                "status": status,
            }
            target[name] = item
            if normalized != name:
                target[normalized] = item
    return result


def _sector_match(
    row: dict[str, object], sector_context: dict[str, object]
) -> tuple[float, float, list[str]]:
    industry = _normalized_sector_name(row.get("行业"))
    concepts = {
        _normalized_sector_name(item)
        for item in re.split(r"[、，,;；/]", str(row.get("概念", "")))
        if _normalized_sector_name(item)
    }
    matches: list[tuple[float, float, str]] = []
    industries = sector_context.get("industries", {})
    if isinstance(industries, dict):
        for key, item in industries.items():
            normalized = _normalized_sector_name(key)
            if industry and normalized and (industry == normalized or normalized in industry or industry in normalized):
                if isinstance(item, dict):
                    matches.append(
                        (_number(item.get("weight")), _number(item.get("strength"), 0.5), str(item.get("name") or key))
                    )
    sector_concepts = sector_context.get("concepts", {})
    if isinstance(sector_concepts, dict):
        for key, item in sector_concepts.items():
            normalized = _normalized_sector_name(key)
            if normalized and any(
                concept == normalized or normalized in concept or concept in normalized
                for concept in concepts
            ):
                if isinstance(item, dict):
                    matches.append(
                        (_number(item.get("weight")), _number(item.get("strength"), 0.5), str(item.get("name") or key))
                    )
    if not matches:
        return 0.0, 0.5, []
    weight = max(item[0] for item in matches)
    strength = max(item[1] for item in matches if item[0] == weight)
    names = list(dict.fromkeys(item[2] for item in matches if item[0] == weight))[:2]
    return min(0.15, max(0.0, weight)), min(1.0, max(0.0, strength)), names


def _is_hard_risk(row: dict[str, object]) -> bool:
    name = str(row.get("名称", "")).upper()
    risk = str(row.get("风险提示", ""))
    return bool(row.get("新闻重大风险")) or "ST" in name or "退" in name or "卖出" in risk


def _probability_percentiles(
    rows: list[dict[str, object]], probabilities: dict[str, object]
) -> dict[str, float]:
    available = [
        (_code(row), _number(probabilities.get(_code(row)), float("nan")))
        for row in rows
    ]
    available = [(code, value) for code, value in available if math.isfinite(value)]
    ordered = sorted(available, key=lambda item: item[1])
    if len(ordered) <= 1:
        return {code: 0.5 for code, _value in ordered}
    return {code: index / (len(ordered) - 1) for index, (code, _value) in enumerate(ordered)}


def rerank_top_five(
    source_rows: list[dict[str, object]],
    precision_context: dict[str, object],
    sector_context: dict[str, object],
    *,
    total_count: int = 20,
    head_count: int = 5,
    industry_cap: int = 2,
    candidate_limit: int | None = None,
    precision_weight: float = 0.25,
    probability_weight: float = 0.10,
    sector_weight_scale: float = 1.0,
    core_rows: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for source in source_rows:
        if not isinstance(source, dict):
            continue
        code = _code(source)
        if len(code) != 6 or not code.isdigit() or code in seen:
            continue
        seen.add(code)
        rows.append(dict(source))
    if not rows:
        return []
    precision_weight = min(1.0, max(0.0, float(precision_weight)))
    probability_weight = min(1.0, max(0.0, float(probability_weight)))
    if precision_weight + probability_weight > 1.0:
        raise ValueError("头部重排核心权重之和不能超过1")
    base_weight = 1.0 - precision_weight - probability_weight
    sector_weight_scale = min(1.0, max(0.0, float(sector_weight_scale)))

    probabilities = precision_context.get("probabilities_by_code", {})
    if not isinstance(probabilities, dict):
        probabilities = {}
    precision_scores = precision_context.get("precision_scores_by_code", {})
    if not isinstance(precision_scores, dict):
        precision_scores = {}
    probability_scores = _probability_percentiles(rows, probabilities)
    denominator = max(1, len(rows) - 1)
    scored: list[tuple[float, int, dict[str, object]]] = []
    for index, row in enumerate(rows):
        code = _code(row)
        base_score = 1.0 - index / denominator
        precision_score = min(1.0, max(0.0, _number(precision_scores.get(code), base_score)))
        probability_score = probability_scores.get(code, 0.5)
        core_score = (
            base_weight * base_score
            + precision_weight * precision_score
            + probability_weight * probability_score
        )
        sector_weight, sector_strength, sector_names = _sector_match(row, sector_context)
        sector_weight *= sector_weight_scale
        final_score = (1.0 - sector_weight) * core_score + sector_weight * sector_strength
        row["原策略排名"] = index + 1
        row["头部重排评分"] = round(final_score * 100.0, 4)
        row["板块因子权重"] = sector_weight
        row["热门板块"] = "、".join(sector_names)
        row["头部重排"] = False
        scored.append((final_score, index, row))

    industry_counts: Counter[str] = Counter()
    selected: list[dict[str, object]] = []
    selected_codes: set[str] = set()
    base_by_code = {_code(row): row for row in rows}
    for source in core_rows or []:
        if not isinstance(source, dict):
            continue
        core_code = _code(source)
        if len(core_code) != 6 or not core_code.isdigit() or core_code in selected_codes:
            continue
        core = dict(source)
        original = base_by_code.get(core_code)
        if original is not None:
            merged = dict(original)
            merged.update(core)
            core = merged
        if _is_hard_risk(core):
            continue
        industry = str(core.get("行业", "")).strip() or "未知行业"
        if industry_counts[industry] >= industry_cap:
            continue
        sector_weight, _sector_strength, sector_names = _sector_match(core, sector_context)
        sector_weight *= sector_weight_scale
        core["原策略排名"] = int(original.get("原策略排名", 0)) if original else 0
        core["头部重排评分"] = 100.0
        core["板块因子权重"] = sector_weight
        core["热门板块"] = "、".join(sector_names)
        core["高胜率核心席位"] = True
        core["头部重排"] = True
        selected.append(core)
        selected_codes.add(core_code)
        industry_counts[industry] += 1
        if len(selected) == head_count:
            break
    head_pool = scored[: max(head_count, int(candidate_limit or len(scored)))]
    for _score, _index, row in sorted(head_pool, key=lambda item: (-item[0], item[1])):
        if _is_hard_risk(row):
            continue
        industry = str(row.get("行业", "")).strip() or "未知行业"
        if industry_counts[industry] >= industry_cap:
            continue
        selected.append(row)
        selected_codes.add(_code(row))
        industry_counts[industry] += 1
        if len(selected) == head_count:
            break

    if len(selected) < head_count:
        for _score, _index, row in sorted(head_pool, key=lambda item: (-item[0], item[1])):
            if _code(row) in selected_codes or _is_hard_risk(row):
                continue
            selected.append(row)
            selected_codes.add(_code(row))
            if len(selected) == head_count:
                break

    for row in selected:
        row["头部重排"] = True
    tail = [row for row in rows if _code(row) not in selected_codes]
    result = (selected + tail)[:total_count]
    for rank, row in enumerate(result, start=1):
        row["排名"] = rank
    return result
