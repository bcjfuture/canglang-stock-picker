"""Transparent bounded scores; initial rules, not calibrated return forecasts."""
import math

import pandas as pd


def number(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def clip(value, low, high):
    return max(low, min(high, value))


# Signs from the full-market rank-IC study (reports/factor-research-*.json,
# 2023-09..2026-09, 5- and 20-session horizons): recent strength, unusual
# volume and volatility predicted underperformance in every year; a short
# pullback predicted outperformance.  +1 = higher is better.
EVIDENCE_FACTORS = {
    "mom_20": -1,
    "mom_5": -1,
    "volatility_20": -1,
    "abnormal_amount": -1,
    "recovery_from_low": -1,
    "ma20_gap": -1,
}


def ttm_roe_pct(row):
    """Trailing-twelve-month ROE implied by PB(MRQ) / PE(TTM).

    ``roe_weight`` is year-to-date (a Q1 value is about a quarter of the annual
    one), so thresholds on it drift with the reporting season.  A negative PE
    means a trailing loss and yields a negative ROE.
    """
    pe = number(row.get('pe9'), None)
    pb = number(row.get('pbnewmrq'), None)
    if pe is None or pb is None or pe == 0 or pb <= 0:
        return None
    return 100.0 * pb / pe


def operating_cash_flow_positive(row):
    """Sign of trailing operating cash flow; None when unknown.

    The price/cash-flow TTM ratio carries the trailing sign.  The per-share
    figure is year-to-date and often negative in Q1 for seasonal reasons, so it
    is only a fallback.
    """
    pcf = number(row.get('pcfjyxjl9'), None)
    if pcf is not None and pcf != 0:
        return pcf > 0
    cfo = number(row.get('per_netcash_operate'), None)
    return None if cfo is None or cfo == 0 else cfo > 0


def historical_features(bars):
    close = pd.to_numeric(bars['close'], errors='coerce')
    low = pd.to_numeric(bars['low'], errors='coerce')
    amount = pd.to_numeric(bars['amount'], errors='coerce')
    drawdown = close / close.rolling(60, min_periods=20).max() - 1
    recent_low = low.iloc[-3:].min()
    previous_low = low.iloc[-8:-3].min()
    low_20 = low.iloc[-20:].min()
    return {
        'selection_history_valid': bool(len(bars) >= 60 and close.iloc[-1] > 0),
        'signal_close': number(close.iloc[-1]),
        'mom_20': number(close.iloc[-1] / close.iloc[-21] - 1),
        'mom_60': number(close.iloc[-1] / close.iloc[-61] - 1) if len(bars) > 60 else None,
        'ma_10': number(close.iloc[-10:].mean()),
        'ma_20': number(close.iloc[-20:].mean()),
        'ma_60': number(close.iloc[-60:].mean()),
        'volatility_20': number(close.pct_change(fill_method=None).iloc[-20:].std()),
        'average_amount_20': number(amount.iloc[-20:].mean()),
        'mom_5': number(close.iloc[-1] / close.iloc[-6] - 1),
        'abnormal_amount': (
            number(amount.iloc[-5:].mean() / amount.iloc[-60:].mean(), None)
            if amount.iloc[-60:].mean() > 0 else None
        ),
        'prior_drawdown_20': number(drawdown.iloc[-21:-1].min()),
        'recent_low_3': number(recent_low),
        'preceding_low_5': number(previous_low),
        'recovery_from_low': number(close.iloc[-1] / low_20 - 1) if low_20 > 0 else None,
    }


def attach_peer_momentum(rows):
    """Industry-relative momentum from the same signal-date cross section."""
    frame = pd.DataFrame(rows)
    if frame.empty:
        return rows
    for column in ('mom_20', 'mom_60'):
        frame[column] = pd.to_numeric(frame.get(column, pd.Series(index=frame.index, dtype=float)), errors='coerce')
    ranks = []
    for column in ('mom_20', 'mom_60'):
        # Sparse industries fall back to the whole market; missing is neutral.
        market_rank = frame[column].rank(pct=True)
        grouped = frame.groupby('industry', dropna=False)[column]
        rank = grouped.rank(pct=True).where(grouped.transform('count').ge(5), market_rank)
        ranks.append(rank.fillna(0.5))
    frame['peer_momentum'] = (ranks[0] + ranks[1]) / 2
    for row, value in zip(rows, frame['peer_momentum']):
        row['peer_momentum'] = float(value)
    return rows


EVIDENCE_MIN_AMOUNT = 50_000_000


def attach_evidence_scores(rows, *, min_amount=EVIDENCE_MIN_AMOUNT):
    """Cross-sectional evidence composite (0..1, higher is better).

    Ranked over liquid names with a usable history, exactly as validated in
    factor_research.py; other rows get None.
    """
    frame = pd.DataFrame(rows)
    if frame.empty:
        return rows
    def column(name):
        return pd.to_numeric(frame.get(name, pd.Series(index=frame.index, dtype=float)), errors='coerce')
    valid = frame.get('selection_history_valid', pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    eligible = valid & (column('average_amount_20') >= min_amount)
    parts = {
        'mom_20': column('mom_20'),
        'mom_5': column('mom_5'),
        'volatility_20': column('volatility_20'),
        'abnormal_amount': column('abnormal_amount'),
        'recovery_from_low': column('recovery_from_low'),
        'ma20_gap': column('signal_close') / column('ma_20') - 1,
    }
    total = pd.Series(0.0, index=frame.index)
    count = pd.Series(0.0, index=frame.index)
    for name, sign in EVIDENCE_FACTORS.items():
        ranked = parts[name].where(eligible).rank(pct=True)
        contribution = ranked if sign > 0 else 1.0 - ranked
        count += contribution.notna().astype(float)
        total += contribution.fillna(0.0)
    score = (total / count).where(eligible & (count >= len(EVIDENCE_FACTORS) - 1))
    for row, value in zip(rows, score):
        row['evidence_score'] = None if pd.isna(value) else float(value)
    return rows


def _quality_score(row, shareholder_score):
    roe = number(ttm_roe_pct(row))
    growth = number(row.get('netprofit_yoy_ratio'))
    cash_positive = operating_cash_flow_positive(row)
    debt = number(row.get('debt_asset_ratio'), None)
    quality = clip(roe / 20, 0, 1)*8 + clip(growth / 30, 0, 1)*6
    quality += 4 if cash_positive else 0
    quality += 3 if debt is not None and 0 <= debt < 70 else 0
    quality += clip(shareholder_score / 15, 0, 1)*4
    return quality, growth, cash_positive


def score_groups_v2(row, config, *, shareholder_score=0, indicator_sell=False):
    """Evidence-weighted score: 60 evidence + 15 quality - risk penalties.

    Fund flow and liquidity size showed no positive predictive power in the
    study, so flow is reported but not scored and liquidity is a hard filter.
    """
    reasons, risks = [], []
    price = number(row.get('signal_close'), number(row.get('new_price')))
    average_amount = number(row.get('average_amount_20'))
    evidence = row.get('evidence_score')
    quality_raw, growth, cash_positive = _quality_score(row, shareholder_score)
    quality = quality_raw * 15 / 25
    evidence_points = 60 * number(evidence)
    liquid = average_amount >= number(config.get('minimum_average_amount_20'), EVIDENCE_MIN_AMOUNT)
    confirmed = bool(row.get('selection_history_valid')) and evidence is not None and liquid
    if not liquid:
        risks.append('20日均成交额不足，不参与严格入选')
    if evidence is None:
        risks.append('历史行情不足或缺口过多，无法计算证据分')
    else:
        reasons.append(f'证据分位{number(evidence):.0%}（低波动、未过热、短期回调）')

    if config.get('score_profile') == 'rebound_trend':
        rules = config['rebound_confirmation']
        # The relative-strength condition was dropped: strength predicted
        # underperformance.  The shape defines the pullback universe.
        conditions = (
            (number(row.get('prior_drawdown_20')) <= -rules['minimum_prior_drawdown'], '此前回撤不足'),
            (number(row.get('recent_low_3')) >= number(row.get('preceding_low_5')) > 0, '低点尚未企稳'),
            (price > number(row.get('ma_10')) > 0, '尚未站上10日线'),
            (0 <= number(row.get('recovery_from_low'), 1) <= rules['maximum_recovery'], '距低点反弹过大'),
        )
        shape_failures = [label for passed, label in conditions if not passed]
        confirmed = confirmed and not shape_failures
        reasons.append('回撤—企稳形态已确认' if not shape_failures else '反弹形态未完全确认')
        risks.extend(shape_failures)
    else:
        reasons.append('科技标签仅作候选范围，不按标签数量加分')

    ma20 = number(row.get('ma_20'))
    penalty = 0.0
    if number(row.get('mom_20')) > 0.25 or (ma20 > 0 and price/ma20 > 1.15):
        penalty += 15
        risks.append('涨幅或均线偏离过热')
    if number(row.get('volatility_20')) > 0.04:
        penalty += 5
        risks.append('近期波动较高')
    if number(row.get('turnoverrate')) > 20 or number(row.get('volume_ratio')) > 4:
        penalty += 5
        risks.append('换手或放量过热')
    if indicator_sell:
        penalty += 10
        risks.append('指标卖出信号')
    if growth < 0 or cash_positive is False:
        penalty += 5
        risks.append('盈利或现金流偏弱')
    penalty = min(30, penalty)
    groups = {'证据分': round(evidence_points, 2), '质量分': round(quality, 2),
              '资金分': 0.0, '流动性分': 0.0, '风险扣分': round(penalty, 2), '趋势分': 0.0}
    reasons.append('分组评分：' + '、'.join(f'{k}{v:g}' for k, v in groups.items() if k not in ('资金分', '流动性分', '趋势分')))
    return {'score': round(clip(evidence_points + quality - penalty, 0, 100), 1),
            'reasons': reasons, 'risks': risks, '形态达标': confirmed, **groups}


def score_groups(row, config, *, shareholder_score=0, indicator_sell=False):
    reasons, risks = [], []
    peer = clip(number(row.get('peer_momentum'), 0.5), 0, 1)
    price = number(row.get('signal_close'), number(row.get('new_price')))
    average_amount = number(row.get('average_amount_20'))
    flow_values = [number(row.get(k), None) for k in ('fund_amount', 'fund_amount_3', 'fund_amount_5')]
    flow = 0.0
    if average_amount > 0 and all(v is not None for v in flow_values):
        one, three, five = flow_values
        # Disjoint daily-average buckets: today, previous 2 days, earlier 2 days.
        buckets = (one, (three - one) / 2, (five - three) / 2)
        intensity = (buckets[0] + 2*buckets[1] + 2*buckets[2]) / (5*average_amount)
        flow = clip(intensity * 300, -20, 20)
        if all(v > 0 for v in buckets):
            flow += 5
            reasons.append('三个不重叠资金区间均净流入')
        elif all(v < 0 for v in buckets):
            flow -= 5
            risks.append('三个不重叠资金区间均净流出')
    else:
        risks.append('资金强度数据不足，资金分按0处理')

    roe = number(ttm_roe_pct(row))
    growth = number(row.get('netprofit_yoy_ratio'))
    cash_positive = operating_cash_flow_positive(row)
    debt = number(row.get('debt_asset_ratio'), None)
    quality = clip(roe / 20, 0, 1)*8 + clip(growth / 30, 0, 1)*6
    quality += 4 if cash_positive else 0
    quality += 3 if debt is not None and 0 <= debt < 70 else 0
    quality += clip(shareholder_score / 15, 0, 1)*4
    liquidity = min(10, average_amount/100_000_000*5)
    liquidity += 5 if 0.5 <= number(row.get('turnoverrate')) <= 15 else 0
    ma20, ma60 = number(row.get('ma_20')), number(row.get('ma_60'))
    trend = 25*peer + (5 if price > ma20 > 0 else 0) + (5 if ma20 > ma60 > 0 else 0)
    confirmed = bool(row.get('selection_history_valid'))
    shape_failures = []
    if config.get('score_profile') == 'rebound_trend':
        rules = config['rebound_confirmation']
        conditions = (
            (confirmed, '历史行情不足或日期不齐'),
            (number(row.get('prior_drawdown_20')) <= -rules['minimum_prior_drawdown'], '此前回撤不足'),
            (number(row.get('recent_low_3')) >= number(row.get('preceding_low_5')) > 0, '低点尚未企稳'),
            (price > number(row.get('ma_10')) > 0 and peer >= rules['minimum_peer_momentum'], '均线或相对强弱未确认'),
            (0 <= number(row.get('recovery_from_low'), 1) <= rules['maximum_recovery'], '距低点反弹过大'),
        )
        shape_failures = [label for passed, label in conditions if not passed]
        confirmed = not shape_failures
        trend = 15*peer + 20*sum(bool(passed) for passed, _ in conditions)/len(conditions)
        reasons.append('回撤—企稳—转强已确认' if confirmed else '反弹形态未完全确认')
        risks.extend(shape_failures)
    else:
        reasons.append('科技标签仅作候选范围，不按标签数量加分')

    penalty = 0.0
    if number(row.get('mom_20')) > 0.25 or (ma20 > 0 and price/ma20 > 1.15):
        penalty += 10
        risks.append('涨幅或均线偏离过热')
    if number(row.get('volatility_20')) > 0.04:
        penalty += 5
        risks.append('近期波动较高')
    if number(row.get('turnoverrate')) > 20 or number(row.get('volume_ratio')) > 4:
        penalty += 5
        risks.append('换手或放量过热')
    if indicator_sell:
        penalty += 10
        risks.append('指标卖出信号')
    if growth < 0 or cash_positive is False:
        penalty += 5
        risks.append('盈利或现金流偏弱')
    penalty = min(30, penalty)
    groups = {'趋势分': round(trend, 2), '资金分': round(flow, 2),
              '质量分': round(quality, 2), '流动性分': round(liquidity, 2), '风险扣分': round(penalty, 2)}
    reasons.append('分组评分：' + '、'.join(f'{k}{v:g}' for k, v in groups.items()))
    return {'score': round(clip(trend+flow+quality+liquidity-penalty, 0, 100), 1),
            'reasons': reasons, 'risks': risks, '形态达标': confirmed, **groups}
