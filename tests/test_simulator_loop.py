"""The event loop's time limit and timestamp checks, on a bare simulator."""

import unittest
from types import SimpleNamespace

from vidur.simulator import Simulator


def bare(time_limit=float("inf")):
    sim = Simulator.__new__(Simulator)
    sim._time, sim._terminate, sim._time_limit = 0, False, time_limit
    sim._event_queue, sim._event_trace, sim._event_chrome_trace = [], [], []
    sim._cluster = None
    sim._scheduler = SimpleNamespace(is_empty=lambda: False)
    sim._metric_store = None
    flags = SimpleNamespace(write_json_trace=False, enable_chrome_trace=False)
    sim._config = SimpleNamespace(metrics_config=flags)
    return sim


def event(time, handled):
    e = SimpleNamespace(time=time, _priority_number=(time, 0, 0))
    e.handle_event = lambda *args: handled.append(time) or []
    return e


class SimulatorLoopTests(unittest.TestCase):
    def test_time_limit_stops_before_an_event_past_the_cutoff(self):
        sim, handled = bare(time_limit=1), []
        for time in (1, 2):
            sim._add_event(event(time, handled))
        sim.run()
        self.assertEqual(handled, [1])
        self.assertEqual(sim._time, 1)
        self.assertEqual(len(sim._event_queue), 1)

    def test_unfinished_requests_without_events_raise(self):
        with self.assertRaisesRegex(RuntimeError, "unfinished requests"):
            bare().run()

    def test_invalid_event_times_are_rejected(self):
        for time in (-1, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "timestamps"):
                bare()._add_event(SimpleNamespace(time=time, _priority_number=(0,)))


if __name__ == "__main__":
    unittest.main()
