"""Synthetic unit-capacity schedules, independent of GPU profiling data."""

import unittest

from vidur.entities.execution_plan import ExecutionActivity as A, ExecutionPlan as P
from vidur.scheduler.resource_executor import ResourceExecutor


class ResourceExecutorTests(unittest.TestCase):
    def test_later_arrival_uses_idle_compute_until_dependency_becomes_ready(self):
        ex = ResourceExecutor({"compute": 1, "memory": 1})
        ex.submit(
            "decode",
            P(
                (
                    A("attention", 4, ("memory",)),
                    A("output", 1, ("compute",), ("attention",)),
                )
            ),
            0,
        )
        self.assertEqual(len(ex.dispatch(0)), 1)
        ex.submit("prefill", P((A("prefill", 2, ("compute",)),)), 1)
        self.assertEqual(ex.dispatch(1)[0][3], 3)
        ex.complete("prefill", "prefill", 3)
        ex.retire("prefill")
        ex.complete("decode", "attention", 4)
        self.assertEqual(ex.dispatch(4)[0][3], 5)
        ex.complete("decode", "output", 5)
        ex.retire("decode")
        self.assertFalse(ex.plans)
        self.assertEqual(ex.used, {"compute": 0, "memory": 0})

    def test_all_resources_are_required_atomically(self):
        ex = ResourceExecutor({"compute": 1, "link": 1})
        ex.submit(0, P((A("transfer", 2, ("link",)),)), 0)
        ex.submit(1, P((A("combined", 3, ("compute", "link")),)), 0)
        self.assertEqual(len(ex.dispatch(0)), 1)
        self.assertEqual(ex.used["compute"], 0)
        ex.complete(0, "transfer", 2)
        self.assertEqual(ex.dispatch(2)[0][3], 5)

    def test_dependency_cycles_and_nonfinite_costs_fail(self):
        ex = ResourceExecutor({"r": 1})
        for plan in [
            P((A("a", float("nan"), ("r",)),)),
            P((A("a", 1, ("r",), ("a",)),)),
        ]:
            with self.assertRaises(ValueError):
                ex.submit(0, plan, 0)
        self.assertFalse(ex.plans)

    def test_retire_reports_schedule_and_critical_path(self):
        ex = ResourceExecutor({"compute": 1, "memory": 1})
        plan = P(
            (
                A("load", 3, ("memory",)),
                A("mix", 1, ("compute",)),
                A("tail", 2, ("compute",), ("mix",)),
                A("out", 1, ("compute",), ("load",)),
            )
        )
        ex.submit("p", plan, 10)
        for _key, activity, _start, finish in ex.dispatch(10):
            pass
        ex.complete("p", "mix", 11)
        ex.dispatch(11)
        ex.complete("p", "load", 13)
        ex.complete("p", "tail", 13)
        ex.dispatch(13)
        ex.complete("p", "out", 14)
        done = ex.retire("p")
        self.assertEqual(done.start, (10, 10, 11, 13))
        self.assertEqual(done.finish, (13, 11, 13, 14))
        # `out` waited on its dependency `load`, not on `tail` (same resource).
        self.assertEqual(done.binding, (None, None, 1, 0))
        self.assertEqual(done.critical_path, (0, 3))
        self.assertEqual((done.submitted_at, done.finished_at), (10, 14))
        self.assertEqual(done.wait_seconds, 0)

    def test_resource_release_binds_when_no_dependency_completes(self):
        ex = ResourceExecutor({"r": 1})
        ex.submit("p", P((A("a", 2, ("r",)), A("b", 3, ("r",)))), 0)
        ex.dispatch(0)
        ex.complete("p", "a", 2)
        ex.dispatch(2)
        ex.complete("p", "b", 5)
        done = ex.retire("p")
        self.assertEqual(done.binding, (None, 0))
        self.assertEqual(done.critical_path, (0, 1))

    def test_waiting_on_another_plan_is_reported_as_wait(self):
        ex = ResourceExecutor({"r": 1})
        ex.submit("first", P((A("a", 4, ("r",)),)), 0)
        ex.dispatch(0)
        ex.submit("second", P((A("b", 1, ("r",)),)), 1)
        ex.dispatch(1)
        ex.complete("first", "a", 4)
        ex.retire("first")
        ex.dispatch(4)
        ex.complete("second", "b", 5)
        done = ex.retire("second")
        self.assertEqual(done.binding, (None,))
        self.assertEqual(done.critical_path, (0,))
        self.assertEqual(done.wait_seconds, 3)


if __name__ == "__main__":
    unittest.main()
