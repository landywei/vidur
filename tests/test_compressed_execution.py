"""Compressed execution reproduces event-by-event execution exactly."""

import random
import unittest

from vidur.entities.execution_plan import CompactPlan, PlanStructure
from vidur.scheduler.memory_pool_manager import MemoryPoolManager
from vidur.scheduler.resource_executor import CompressedState, ResourceExecutor

RESOURCES = ("gpu", "pim", "link")


def random_structure(rng, size):
    durations, resources, deps, reads, writes = [], [], [], [], []
    for i in range(size):
        durations.append(
            rng.choice([0.0, 1e-6, 3.3e-5, rng.random() * 1e-3, 0.1 + 0.2])
        )
        resources.append(tuple(rng.sample(RESOURCES, rng.randint(1, 2))))
        deps.append(tuple(sorted(rng.sample(range(i), min(i, rng.randint(0, 2))))))
        pool = rng.choice(("kv0", "kv1"))
        if rng.random() < 0.3:
            writes.append(((0, pool, rng.choice((64, 128))),))
            reads.append(())
        elif rng.random() < 0.3:
            reads.append(((rng.choice((0, 1)), pool, rng.choice((32, 128))),))
            writes.append(())
        else:
            reads.append(())
            writes.append(())
    return PlanStructure(
        tuple(durations),
        tuple(resources),
        tuple(deps),
        tuple(reads),
        tuple(writes),
        ("p",) * size,
        ("g",) * size,
        "plan",
    )


def executor(resident):
    manager = MemoryPoolManager({"kv0": 1024, "kv1": 1024})
    for request in (5, 9):
        manager.reserve(request, {"kv0": 256, "kv1": 256})
    for key, value in resident.items():
        manager.resident[key] = value
    return ResourceExecutor(dict.fromkeys(RESOURCES, 1), manager)


def by_events(structure, ids, time, resident):
    """The event loop's order for one plan: completions, then a dispatch."""
    ex = executor(resident)
    plan = CompactPlan(structure, ids).expand()
    ex.submit("p", plan, time)
    running = {}
    for _key, activity, _start, finish in ex.dispatch(time):
        running[activity.name] = finish
    while running:
        now = min(running.values())
        for name in [n for n, f in running.items() if f == now]:
            del running[name]
            ex.complete("p", name, now)
        for _key, activity, _start, finish in ex.dispatch(now):
            running[activity.name] = finish
    return ex.retire("p"), ex.memory_manager.resident


def compressed(ex, structure, ids, time, key="p"):
    finish = ex.submit(key, CompactPlan(structure, ids), time)
    assert isinstance(ex.plans[key], CompressedState)
    assert ex.finished(key, finish)
    assert finish == time or not ex.finished(key, finish * (1 - 1e-12))
    return ex.retire(key, finish)


FIELDS = ("start", "finish", "binding", "critical_path", "finished_at")


class CompressedExecutionTests(unittest.TestCase):
    def test_random_plans_match_event_execution(self):
        rng = random.Random(7)
        kinds = {"simulated": 0, "replayed": 0, "stalled": 0}
        for _case in range(300):
            structure = random_structure(rng, rng.randint(1, 25))
            resident = {(5, "kv0"): rng.choice((0, 64)), (9, "kv1"): 128}
            ids = (5, 9)
            ex = executor(resident)
            initial = dict(ex.memory_manager.resident)
            gaps = [0.0, 1e-9, 0.1 + 0.2, 1234.5678901, rng.random() * 1e4, 7e5]
            time = 0.0
            for n, gap in enumerate(gaps):
                # Submissions follow each other, as on one pipeline stage.
                time = ex.time + gap
                try:
                    expected, memory = by_events(structure, ids, time, resident)
                except Exception:  # a plan that stalls on memory
                    expected = None
                if expected is None:
                    # It stalls alone too: the executor runs it by events.
                    plan = CompactPlan(structure, ids)
                    self.assertIsNone(ex.submit(f"k{n}", plan, time))
                    kinds["stalled"] += 1
                    break
                got = compressed(ex, structure, ids, time, key=f"k{n}")
                for field in FIELDS:
                    self.assertEqual(getattr(got, field), getattr(expected, field))
                if n == 0:
                    self.assertEqual(got.execution, "simulated")
                kinds[got.execution] += 1
                # Residency after retirement, then restore the initial state.
                self.assertEqual(ex.memory_manager.resident, memory)
                ex.memory_manager.resident.clear()
                ex.memory_manager.resident.update(initial)
        self.assertTrue(all(kinds.values()), kinds)

    def test_replay_is_used_and_rejected_when_rounding_changes_ties(self):
        # Two chains whose sums tie at t = 0 but not at every t.
        structure = PlanStructure(
            (0.1, 0.2, 0.3, 1.0),
            (("gpu",), ("gpu",), ("pim",), ("link",)),
            ((), (0,), (), (1, 2)),
            ((), (), (), ()),
            ((), (), (), ()),
            ("p",) * 4,
            ("g",) * 4,
        )
        ex = ResourceExecutor(dict.fromkeys(RESOURCES, 1))
        kinds = []
        # (t + 0.1) + 0.2 is above t + 0.3 at t = 0, 2 and 1000, equal at 4
        # and 6, and below at 8.
        for n, time in enumerate((0.0, 2.0, 4.0, 6.0, 8.0, 1000.0)):
            expected, _ = by_events(structure, (), time, {})
            got = compressed(ex, structure, (), time, key=n)
            for field in FIELDS:
                self.assertEqual(getattr(got, field), getattr(expected, field))
            kinds.append(got.execution)
        self.assertEqual(
            kinds,
            ["simulated", "replayed", "simulated", "replayed", "simulated", "simulated"],
        )
        self.assertEqual(ex.compression["replay_rejected"], 3)

    def test_overlapping_plans_run_by_events(self):
        structure = PlanStructure(
            (1.0,), (("gpu",),), ((),), ((),), ((),), ("p",), ("g",)
        )
        ex = ResourceExecutor(dict.fromkeys(RESOURCES, 1))
        self.assertEqual(ex.submit("a", CompactPlan(structure, ()), 0.0), 1.0)
        # A plan overlapping a compressed one cannot run beside it: no
        # activity events hold the compressed plan's resources.
        with self.assertRaisesRegex(ValueError, "overlaps a plan running compressed"):
            ex.submit("b", CompactPlan(structure, ()), 0.5)
        self.assertEqual(ex.compression["expanded"], 1)
        self.assertFalse(ex.finished("a", 0.9))
        self.assertTrue(ex.finished("a", 1.0))
        self.assertEqual(ex.retire("a", 1.0).execution, "simulated")


if __name__ == "__main__":
    unittest.main()
