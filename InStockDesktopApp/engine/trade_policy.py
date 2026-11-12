from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class TradePolicy:
    policy_version: str
    holding_days: int
    atr_multiplier: float
    reward_risk: float
    stop_loss_floor: float = 0.04
    stop_loss_cap: float = 0.08
    take_profit_floor: float = 0.08
    take_profit_cap: float = 0.16
    fixed_take_profit_pct: float | None = None
    fixed_stop_loss_pct: float | None = None
    position_fraction: float | None = None

    def __post_init__(self) -> None:
        if self.holding_days not in {5, 7, 10, 15, 20}:
            raise ValueError("unsupported holding_days")
        if self.atr_multiplier not in {1.0, 1.25, 1.5, 1.75, 2.0}:
            raise ValueError("unsupported atr_multiplier")
        if self.reward_risk not in {1.5, 2.0, 2.5}:
            raise ValueError("unsupported reward_risk")
        if (self.fixed_take_profit_pct is None) != (self.fixed_stop_loss_pct is None):
            raise ValueError("fixed take-profit and stop-loss must be configured together")
        if self.fixed_take_profit_pct is not None:
            if not 0 < float(self.fixed_take_profit_pct) <= 0.20:
                raise ValueError("invalid fixed_take_profit_pct")
            if not 0 < float(self.fixed_stop_loss_pct) <= 0.20:
                raise ValueError("invalid fixed_stop_loss_pct")
        if self.position_fraction is not None and not 0 < float(self.position_fraction) <= 1.0:
            raise ValueError("invalid position_fraction")

    @classmethod
    def from_dict(cls, value: dict) -> "TradePolicy":
        return cls(**{key: value[key] for key in cls.__dataclass_fields__ if key in value})

    def to_dict(self) -> dict:
        return asdict(self)


def _finite_positive(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def calculate_advice(price: object, atr_14: object, policy: TradePolicy) -> dict:
    reference = _finite_positive(price)
    if reference is None:
        return {"status": "insufficient_data"}
    if policy.fixed_take_profit_pct is not None:
        stop_pct = float(policy.fixed_stop_loss_pct)
        take_pct = float(policy.fixed_take_profit_pct)
    else:
        atr = _finite_positive(atr_14)
        if atr is None:
            return {"status": "insufficient_data"}
        stop_pct = min(
            policy.stop_loss_cap,
            max(policy.stop_loss_floor, policy.atr_multiplier * atr / reference),
        )
        take_pct = min(
            policy.take_profit_cap,
            max(policy.take_profit_floor, policy.reward_risk * stop_pct),
        )
    return {
        "status": "ready",
        "advice_holding_days": policy.holding_days,
        "stop_loss_pct": round(stop_pct, 6),
        "take_profit_pct": round(take_pct, 6),
        "stop_loss_price": round(reference * (1.0 - stop_pct), 2),
        "take_profit_price": round(reference * (1.0 + take_pct), 2),
        "policy_version": policy.policy_version,
        "position_fraction": policy.position_fraction,
    }


def simulate_trade(
    bars,
    signal_pos: int,
    atr_14: object,
    policy: TradePolicy,
    cost: float = 0.003,
    hard_cap: int | None = None,
) -> dict | None:
    entry_pos = int(signal_pos) + 1
    if entry_pos >= len(bars):
        return None
    entry_price = _finite_positive(bars.iloc[entry_pos]["open"])
    advice = calculate_advice(entry_price, atr_14, policy)
    if entry_price is None or advice["status"] != "ready":
        return None
    stop = advice["stop_loss_price"]
    take = advice["take_profit_price"]
    holding_limit = min(policy.holding_days, hard_cap) if hard_cap else policy.holding_days
    final_pos = min(entry_pos + holding_limit, len(bars) - 1)
    exit_pos = final_pos
    exit_price = float(bars.iloc[final_pos]["close"])
    reason = "time_exit"
    for pos in range(entry_pos, final_pos + 1):
        row = bars.iloc[pos]
        open_price = float(row["open"])
        if open_price <= stop:
            exit_pos, exit_price, reason = pos, open_price, "gap_stop"
            break
        if open_price >= take:
            exit_pos, exit_price, reason = pos, open_price, "gap_take"
            break
        if float(row["low"]) <= stop:
            exit_pos, exit_price, reason = pos, stop, "stop_loss"
            break
        if float(row["high"]) >= take:
            exit_pos, exit_price, reason = pos, take, "take_profit"
            break
    gross = exit_price / entry_price - 1.0
    return {
        **advice,
        "entry_date": str(bars.iloc[entry_pos]["date"])[:10],
        "entry_price": entry_price,
        "exit_date": str(bars.iloc[exit_pos]["date"])[:10],
        "exit_price": round(exit_price, 4),
        "exit_reason": reason,
        "holding_days": exit_pos - entry_pos,
        "gross_return": gross,
        "net_return": gross - float(cost),
    }
