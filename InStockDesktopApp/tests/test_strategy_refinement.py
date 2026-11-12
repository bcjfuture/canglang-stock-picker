import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'engine'))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
import instock_tech_picker_data as picker
from instock_app.settings import STRATEGIES
from workers.picker_worker import apply_head_reranking, normalize_picker_payload
from workers.recent_backtest_worker import find_stored_snapshot, _load_cached_snapshot
from selection_refinement import attach_peer_momentum


class StrategyRefinementTests(unittest.TestCase):
    def config(self, key, **overrides):
        config = json.loads((ROOT / 'engine' / f'{key}_config.json').read_text(encoding='utf-8'))
        config.update(overrides)
        return config

    def row(self, **changes):
        row = dict(code='000001', name='样本', industry='半导体', concept='芯片',
                   new_price=11, change_rate=1, turnoverrate=3, volume_ratio=1.5,
                   deal_amount=200_000_000, average_amount_20=200_000_000,
                   fund_amount=10_000_000, fund_amount_3=30_000_000,
                   fund_amount_5=50_000_000, pe9=25, pbnewmrq=2,
                   roe_weight=15, netprofit_yoy_ratio=20, debt_asset_ratio=40,
                   per_netcash_operate=1, historical_factor_score=0.5,
                   mom_20=0.1, mom_60=0.2, volatility_20=0.02,
                   ma_10=10.5, ma_20=10, ma_60=9, peer_momentum=0.75,
                   prior_drawdown_20=-0.12, recent_low_3=10,
                   preceding_low_5=9.8, recovery_from_low=0.05,
                   selection_history_valid=True, evidence_score=0.8, mom_5=-0.02,
                   abnormal_amount=0.9, signal_close=11)
        row.update(changes)
        return row

    def score(self, key, config_overrides=None, **changes):
        return picker.score_row(self.row(**changes), self.config(key, **(config_overrides or {})), set(), set(), set(), {})

    def test_tech_concept_synonyms_do_not_inflate_score(self):
        first = self.score('tech_attack', concept='芯片')
        many = self.score('tech_attack', concept='芯片 人工智能 算力 服务器 云计算')
        self.assertEqual(first['综合评分'], many['综合评分'])

    def test_tech_flow_score_is_scale_invariant(self):
        first = self.score('tech_attack')
        large = self.score('tech_attack', deal_amount=2_000_000_000,
                           average_amount_20=2_000_000_000, fund_amount=100_000_000,
                           fund_amount_3=300_000_000, fund_amount_5=500_000_000)
        self.assertEqual(first['资金分'], large['资金分'])

    def test_peer_comparison_is_within_industry(self):
        rows = [dict(code=f'{industry}{i}', industry=industry, mom_20=i/100+offset, mom_60=i/100+offset)
                for industry, offset in [('A', 0), ('B', 1)] for i in range(5)]
        attach_peer_momentum(rows)
        self.assertEqual(rows[4]['peer_momentum'], rows[9]['peer_momentum'])
        self.assertGreater(rows[4]['peer_momentum'], rows[5]['peer_momentum'])

    def test_technical_flags_cannot_double_count_grouped_trend(self):
        first = self.score('tech_attack')
        repeated = self.score('tech_attack', macd_golden_fork=1, kdj_golden_fork=1,
                              breakup_ma_5days=1, breakup_ma_10days=1, breakup_ma_20days=1)
        self.assertEqual(first['综合评分'], repeated['综合评分'])

    def test_old_selection_version_is_not_reused_for_current_backtest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / 'tech_attack'
            folder.mkdir()
            path = folder / 'old.json'
            payload = {'trade_date': '2026-08-24', 'rows': [], 'source_summary': {'selection_version': 'old'}}
            path.write_text(json.dumps(payload), encoding='utf-8')
            self.assertIsNone(find_stored_snapshot(root, 'tech_attack', '2026-08-24', required_selection_version='new'))
            self.assertIsNone(_load_cached_snapshot(path, '2026-08-24', required_selection_version='new'))
            payload['source_summary']['selection_version'] = 'new'
            path.write_text(json.dumps(payload), encoding='utf-8')
            self.assertIsNotNone(find_stored_snapshot(root, 'tech_attack', '2026-08-24', required_selection_version='new'))

    def test_one_day_spike_does_not_count_as_five_day_persistence(self):
        v1 = {'scoring_version': 'grouped-v1'}
        spike = self.score('tech_attack', v1, fund_amount=50_000_000,
                           fund_amount_3=50_000_000, fund_amount_5=50_000_000)
        steady = self.score('tech_attack', v1)
        self.assertLess(spike['资金分'], steady['资金分'])

    def test_rebound_requires_prior_drawdown_and_stabilization(self):
        self.assertTrue(self.score('rebound_trend')['形态达标'])
        for changes in (dict(prior_drawdown_20=-0.01), dict(recent_low_3=9),
                        dict(signal_close=10), dict(selection_history_valid=False),
                        dict(evidence_score=None), dict(average_amount_20=1_000_000)):
            with self.subTest(changes=changes):
                self.assertFalse(self.score('rebound_trend', **changes)['形态达标'])

    def test_trend_compares_prices_on_same_adjustment_basis(self):
        # Historical bars may be adjusted; a raw quote must not distort MA distance.
        first = self.score('tech_attack')
        adjusted = self.score('tech_attack', new_price=110, signal_close=11)
        self.assertEqual(first['综合评分'], adjusted['综合评分'])

    def test_twenty_slots_keep_qualified_first_and_label_fallback(self):
        for key in ('tech_attack', 'rebound_trend'):
            cfg = self.config(key)
            rows = [{'代码': '000001', '排名': 7, '综合评分': cfg['min_score']+1, '形态达标': True}]
            rows += [{'代码': f'{i:06d}', '排名': i, '综合评分': cfg['min_score']-1, '形态达标': False}
                     for i in range(2, 25)]
            selected = picker.select_output_rows(rows, {**cfg, 'fallback_fill': True}, active_count=20, regime='legacy')
            self.assertEqual(20, len(selected))
            self.assertEqual(1, selected[0]['排名'])
            self.assertEqual('严格入选', selected[0]['入选层级'])
            self.assertEqual('候补入选', selected[1]['入选层级'])
            self.assertIn('候补', selected[1]['风险提示'])

    def test_strict_only_strategies_do_not_fill_with_fallback(self):
        for key in ('tech_attack', 'rebound_trend'):
            cfg = self.config(key)
            rows = [{'代码': '000001', '排名': 1, '综合评分': cfg['min_score']+1, '形态达标': True}]
            rows += [{'代码': f'{i:06d}', '排名': i, '综合评分': cfg['min_score']-1, '形态达标': False}
                     for i in range(2, 25)]
            selected = picker.select_output_rows(rows, cfg, active_count=20, regime='legacy')
            self.assertEqual(['000001'], [row['代码'] for row in selected])

    def test_v2_penalizes_overheating_and_ignores_fund_flow(self):
        calm = self.score('tech_attack')
        hot = self.score('tech_attack', mom_20=0.35)
        inflow = self.score('tech_attack', fund_amount=500_000_000, fund_amount_3=900_000_000, fund_amount_5=1_500_000_000)
        self.assertLess(hot['综合评分'], calm['综合评分'])
        self.assertEqual(calm['综合评分'], inflow['综合评分'])

    def test_evidence_ranks_calm_pullback_above_extended_strength(self):
        from selection_refinement import attach_evidence_scores
        base = dict(selection_history_valid=True, average_amount_20=2e8, ma_20=10)
        rows = [dict(base, code=str(i), mom_20=0.01*i, mom_5=0.005*i, volatility_20=0.01+0.001*i,
                     abnormal_amount=0.8+0.05*i, recovery_from_low=0.01*i, signal_close=10+0.1*i)
                for i in range(10)]
        rows.append(dict(base, code='thin', average_amount_20=1e6, mom_20=0, mom_5=0, volatility_20=0.01,
                         abnormal_amount=0.8, recovery_from_low=0, signal_close=10))
        attach_evidence_scores(rows)
        self.assertGreater(rows[0]['evidence_score'], rows[9]['evidence_score'])
        self.assertIsNone(rows[-1]['evidence_score'])

    def test_empty_qualified_list_is_not_repopulated_by_worker(self):
        spec = STRATEGIES['tech_attack']
        payload = {'sheets': {spec.top_sheet: [], '全部': [{'代码': '000001', '综合评分': 1}]},
                   'summary': {'abstain_reason': '没有达标候选'}}
        apply_head_reranking(spec, payload, {}, {}, all_sheet_name='全部', policy={'enabled': False})
        self.assertEqual([], normalize_picker_payload(spec, payload)['rows'])

    def test_market_quality_priority_remains_visible_after_selection(self):
        base = {'行业': '银行', 'PE': 5, '历史最低ROE%': 15, '股息率%': 10, '综合评分': 50}
        healthy = dict(base, **{'代码': '000001', 'PB': 0.8, '净利同比%': 5, '三年净利增速%': 5})
        weak = dict(base, **{'代码': '000002', 'PB': 0.2, '净利同比%': -5, '三年净利增速%': -5})
        result = picker.select_market_earnings_v3([weak, healthy], self.config('all_market'))
        self.assertEqual(['000001', '000002'], [r['代码'] for r in result])
        self.assertIn('持续盈利走弱', result[1]['风险提示'])
        self.assertIn('质量', result[0]['加分理由'])

    def test_selection_history_is_point_in_time_and_requires_current_bar(self):
        with tempfile.TemporaryDirectory() as temp:
            dates = pd.bdate_range('2026-01-01', periods=80)
            prices = [10 + i/100 for i in range(80)]
            bars = pd.DataFrame(dict(date=dates, open=prices, high=prices,
                                     low=prices, close=prices, volume=1e6, amount=1e8))
            path = Path(temp) / '000001.csv'
            bars.to_csv(path, index=False)
            row = self.row(date=dates[69], new_price=prices[69])
            first = picker.load_historical_context([row], {'mom_20': 1}, Path(temp), selection_features=True)
            bars.loc[70:, 'close'] = 9999
            bars.to_csv(path, index=False)
            second = picker.load_historical_context([row], {'mom_20': 1}, Path(temp), selection_features=True)
            self.assertEqual(first, second)
            self.assertTrue(first['000001']['selection_history_valid'])
            bars.iloc[:65].to_csv(path, index=False)
            stale = picker.load_historical_context([row], {'mom_20': 1}, Path(temp), selection_features=True)
            self.assertFalse(stale.get('000001', {}).get('selection_history_valid', False))


if __name__ == '__main__':
    unittest.main()
