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


if __name__ == "__main__":
    unittest.main()
