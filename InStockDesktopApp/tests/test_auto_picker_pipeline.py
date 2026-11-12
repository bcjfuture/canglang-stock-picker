from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from instock_app.controller import AppController
from instock_app.contracts import TaskResult
from instock_app.result_store import ResultStore
from instock_app.settings import STRATEGIES


class Runner:
    def __init__(self):
        self.commands = []
    def start(self, command, event, done):
        self.commands.append(command)
        return str(len(self.commands))
    def cancel(self, task_id):
        return True


class AutoPickerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.runner = Runner()
        self.done = []
        self.controller = AppController(
            SimpleNamespace(python_exe=Path(sys.executable), workers_dir=ROOT/'workers', data_dir=root),
            self.runner, ResultStore(root/'store'), SimpleNamespace(start_database=lambda: 1))
        self.controller.set_observers(lambda e: None, self.done.append)

    def test_update_runs_four_and_saves_switchable_results_then_backtests(self):
        c = self.controller
        c.start_update()
        c.on_done(TaskResult(0))
        for key in STRATEGIES:
            self.assertEqual(c.state.current_strategy, key)
            self.assertFalse(c.state.actions_enabled)
            payload = {'strategy': key, 'rows': [{'code': '000001'}]}
            c.on_done(TaskResult(0, payload))
            self.assertEqual(c.load_picker_result(key), payload)
        self.assertEqual(c.state.current_kind, 'backtest')
        c.on_done(TaskResult(0))
        self.assertEqual(c.state.current_kind, 'sector_forecast')
        c.on_done(TaskResult(0))
        self.assertTrue(c.state.actions_enabled)
        self.assertEqual(len(self.runner.commands), 7)
        self.assertIn('recent-backtest', self.runner.commands[5])

    def test_failed_update_stops_pipeline(self):
        self.controller.start_update()
        self.controller.on_done(TaskResult(1))
        self.assertEqual(len(self.runner.commands), 1)

    def test_failed_picker_stops_pipeline_without_overwriting_result(self):
        c = self.controller
        c.start_update()
        c.on_done(TaskResult(0))
        c.on_done(TaskResult(1))
        self.assertEqual(len(self.runner.commands), 2)
        self.assertTrue(c.state.actions_enabled)
        self.assertIsNone(c.load_picker_result('all_market'))

    def test_cancel_blocks_followup_even_if_worker_exits_successfully(self):
        c = self.controller
        c.start_update()
        self.assertTrue(c.cancel())
        c.on_done(TaskResult(0))
        self.assertEqual(len(self.runner.commands), 1)

    def test_manual_picker_does_not_run_other_strategies(self):
        c = self.controller
        c.run_picker('tech_attack')
        c.on_done(TaskResult(0, {'rows': []}))
        self.assertEqual(len(self.runner.commands), 1)

    def test_cancel_during_picker_stops_remaining_strategies(self):
        c = self.controller
        c.start_update()
        c.on_done(TaskResult(0))
        c.cancel()
        c.on_done(TaskResult(0, {'strategy': 'all_market', 'rows': []}))
        self.assertEqual(len(self.runner.commands), 2)
        self.assertTrue(c.state.actions_enabled)

    def test_backtest_failure_keeps_all_picker_results(self):
        c = self.controller
        c.start_update()
        c.on_done(TaskResult(0))
        for key in STRATEGIES:
            c.on_done(TaskResult(0, {'strategy': key, 'rows': []}))
        c.on_done(TaskResult(1))
        self.assertEqual(len(self.runner.commands), 6)
        self.assertTrue(all(c.load_picker_result(key) is not None for key in STRATEGIES))

    def test_save_failure_stops_pipeline_and_releases_actions(self):
        c = self.controller
        c.start_update()
        c.on_done(TaskResult(0))
        c.store.record_run = lambda *args: (_ for _ in ()).throw(OSError('disk full'))
        c.on_done(TaskResult(0, {'strategy': 'all_market', 'rows': []}))
        self.assertTrue(c.state.actions_enabled)
        self.assertEqual(len(self.runner.commands), 2)
        self.assertEqual(self.done[-1].returncode, 1)

    def test_completion_metadata_is_captured_before_next_task(self):
        self.controller.start_update()
        self.controller.on_done(TaskResult(0))
        self.assertEqual(self.done[0].payload['_completed_kind'], 'update')

    def test_start_failure_releases_pipeline(self):
        c = self.controller
        c.start_update()
        self.runner.start = lambda *args: (_ for _ in ()).throw(OSError('failed to launch'))
        c.on_done(TaskResult(0))
        self.assertTrue(c.state.actions_enabled)
        self.assertIsNone(c.state.current_kind)

    def test_manual_start_cannot_interrupt_automatic_handoff(self):
        c = self.controller
        blocked = []
        def done(result):
            try:
                c.run_picker('tech_attack')
            except RuntimeError:
                blocked.append(True)
        c.set_observers(lambda e: None, done)
        c.start_update()
        c.on_done(TaskResult(0))
        self.assertEqual(blocked, [True])
        self.assertEqual(c.state.current_strategy, 'all_market')

    def test_cancellation_during_handoff_stops_next_task(self):
        c = self.controller
        c.set_observers(lambda e: None, lambda result: c.cancel())
        c.start_update()
        c.on_done(TaskResult(0))
        self.assertEqual(len(self.runner.commands), 1)
        self.assertTrue(c.state.actions_enabled)


if __name__ == '__main__':
    unittest.main()
