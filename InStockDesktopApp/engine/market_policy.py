from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class MarketPolicy:
    breadth_floor: float
    momentum_floor: float
    drawdown_floor: float
    strong_pick_count: int
    normal_pick_count: int

    def __post_init__(self) -> None:
        if self.strong_pick_count not in {12, 15}:
            raise ValueError("unsupported strong_pick_count")
        if self.normal_pick_count not in {5, 8, 10}:
            raise ValueError("unsupported normal_pick_count")

    @classmethod
    def from_dict(cls, value: dict) -> "MarketPolicy":
        return cls(**{key: value[key] for key in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)


def classify_regime(
    breadth_above_ma20: float,
    market_median_mom20: float,
    market_median_drawdown60: float,
    policy: MarketPolicy,
) -> str:
    if (
        breadth_above_ma20 < policy.breadth_floor
        or market_median_mom20 < policy.momentum_floor
        or market_median_drawdown60 < policy.drawdown_floor
    ):
        return "weak"
    if (
        breadth_above_ma20 >= policy.breadth_floor + 0.10
        and market_median_mom20 >= policy.momentum_floor + 0.10
        and market_median_drawdown60 >= policy.drawdown_floor + 0.05
    ):
        return "strong"
    return "normal"


def pick_count_for_regime(regime: str, policy: MarketPolicy) -> int:
    if regime == "strong":
        return policy.strong_pick_count
    if regime == "normal":
        return policy.normal_pick_count
    if regime == "weak":
        return 0
    raise ValueError(f"unknown regime: {regime}")


def participation_checks(
    scheduled_dates: int,
    active_counts: list[int],
    fold_active_dates: list[int],
) -> dict:
    rate = len(active_counts) / scheduled_dates if scheduled_dates > 0 else 0.0
    checks = [
        {
            "name": "活跃选股数量5至15只",
            "passed": bool(active_counts) and all(5 <= value <= 15 for value in active_counts),
            "value": list(active_counts),
        },
        {"name": "活跃覆盖率至少40%", "passed": rate >= 0.40, "value": rate},
        {
            "name": "每个半年阶段至少8个活跃调仓日",
            "passed": len(fold_active_dates) >= 5 and all(value >= 8 for value in fold_active_dates),
            "value": list(fold_active_dates),
        },
        {
            "name": "聚合样本外至少30个活跃调仓日",
            "passed": len(active_counts) >= 30,
            "value": len(active_counts),
        },
    ]
    return {
        "passed": all(check["passed"] for check in checks),
        "activity_rate": rate,
        "active_counts": list(active_counts),
        "fold_active_dates": list(fold_active_dates),
        "checks": checks,
    }


def neighbor_stability(candidate_drawdown: float, neighbors: list[dict]) -> dict:
    required = math.ceil(len(neighbors) / 2) if neighbors else 1
    stable = [
        item
        for item in neighbors
        if float(item.get("mean_return", 0.0)) > 0
        and abs(float(item.get("max_drawdown", -1.0))) <= abs(float(candidate_drawdown)) + 0.05
    ]
    return {
        "passed": len(stable) >= required,
        "stable_neighbors": len(stable),
        "required_neighbors": required,
        "neighbor_count": len(neighbors),
    }
