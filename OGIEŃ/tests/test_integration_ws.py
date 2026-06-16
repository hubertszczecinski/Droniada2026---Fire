"""Testy parsowania zdarzeń WebSocket autonomii."""
from __future__ import annotations

import unittest

from release.integration_ws import (
    autonomy_analysis_pause_s,
    autonomy_hold_pause_s,
    hold_duration_s,
    parse_autonomy_event,
)


class TestAutonomyWsParse(unittest.TestCase):
    def test_hold_started(self) -> None:
        ev = parse_autonomy_event('{"event":"hold_started","timeout":12.5}')
        self.assertIsNotNone(ev)
        assert ev is not None
        self.assertEqual(ev['event'], 'hold_started')
        self.assertAlmostEqual(ev['timeout'], 12.5)

    def test_hold_stopped(self) -> None:
        ev = parse_autonomy_event('{"event":"hold_stopped","timeout":0.0}')
        self.assertIsNotNone(ev)
        assert ev is not None
        self.assertEqual(ev['event'], 'hold_stopped')
        self.assertAlmostEqual(ev['timeout'], 0.0)

    def test_hold_duration_from_host(self) -> None:
        self.assertAlmostEqual(hold_duration_s(5.0), 5.0)
        self.assertAlmostEqual(hold_duration_s(None), 8.0)

    def test_analysis_pause_from_env(self) -> None:
        import os

        prev = os.environ.get('DRONIADA_AUTONOMY_HOLD_PAUSE_S')
        os.environ['DRONIADA_AUTONOMY_HOLD_PAUSE_S'] = '12'
        try:
            self.assertAlmostEqual(autonomy_analysis_pause_s(), 12.0)
            self.assertAlmostEqual(autonomy_hold_pause_s(), 12.0)
        finally:
            if prev is None:
                os.environ.pop('DRONIADA_AUTONOMY_HOLD_PAUSE_S', None)
            else:
                os.environ['DRONIADA_AUTONOMY_HOLD_PAUSE_S'] = prev

    def test_ignores_other(self) -> None:
        self.assertIsNone(parse_autonomy_event('{"event":"tick"}'))
        self.assertIsNone(parse_autonomy_event('not json'))


if __name__ == '__main__':
    unittest.main()
