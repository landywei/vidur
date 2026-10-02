"""Byte accounting and residency semantics, independent of architecture models."""

import unittest

from vidur.entities.execution_plan import (
    ExecutionActivity as A,
)
from vidur.entities.execution_plan import (
    ExecutionPlan as P,
)
from vidur.entities.execution_plan import (
    MemoryAccess as M,
)
from vidur.scheduler.memory_pool_manager import MemoryPoolManager
from vidur.scheduler.resource_executor import ResourceExecutor


class MemoryPoolTests(unittest.TestCase):
    def test_atomic_reservation_and_release(self):
        pools = MemoryPoolManager({"a": 16, "b": 8})
        pools.reserve(0, {"a": 8, "b": 8})
        self.assertFalse(pools.can_reserve(1, {"a": 8, "b": 8}))
        with self.assertRaises(ValueError):
            pools.reserve(1, {"a": 8, "b": 8})
        self.assertEqual(pools.used, {"a": 8, "b": 8})
        pools.reserve(0, {"a": 8, "b": 8})
        self.assertEqual(pools.resident, {(0, "a"): 0, (0, "b"): 0})
        pools.free(0)
        pools.reserve(1, {"a": 8, "b": 8})
        self.assertNotIn((0, "a"), pools.resident)

    def test_transfer_publishes_destination_only_at_completion(self):
        pools = MemoryPoolManager({"a": 16, "b": 16})
        pools.reserve(0, {"a": 16, "b": 16})
        ex = ResourceExecutor({"compute": 1, "link": 1, "near": 1}, pools)
        ex.submit(
            0,
            P(
                (
                    A("produce", 2, ("compute",), memory_writes=(M(0, "a", 8),)),
                    A(
                        "copy",
                        3,
                        ("link",),
                        memory_reads=(M(0, "a", 8),),
                        memory_writes=(M(0, "b", 8),),
                    ),
                    A("consume", 1, ("near",), memory_reads=(M(0, "b", 8),)),
                )
            ),
            0,
        )
        self.assertEqual([x[1].name for x in ex.dispatch(0)], ["produce"])
        self.assertEqual(pools.resident[0, "a"], 0)
        ex.complete(0, "produce", 2)
        self.assertEqual([x[1].name for x in ex.dispatch(2)], ["copy"])
        self.assertEqual(pools.resident[0, "b"], 0)
        with self.assertRaises(ValueError):
            pools.free(0)
        ex.complete(0, "copy", 5)
        self.assertEqual([x[1].name for x in ex.dispatch(5)], ["consume"])
        ex.complete(0, "consume", 6)
        ex.retire(0)
        pools.free(0)
        self.assertEqual(pools.used, {"a": 0, "b": 0})

    def test_memory_hazards_serialize_writers_but_allow_readers(self):
        pools = MemoryPoolManager({"a": 16})
        pools.reserve(0, {"a": 16})
        write, read = (M(0, "a", 8),), (M(0, "a", 8),)
        pools.begin((), write)
        self.assertFalse(pools.ready(read, ()))
        pools.complete((), write)
        pools.begin(read, ())
        self.assertTrue(pools.ready(read, ()))
        self.assertFalse(pools.ready((), write))
        pools.complete(read, ())
        pools.begin(read, write)
        pools.complete(read, write)
        pools.free(0)
        pools.reserve(0, {"a": 16})
        self.assertFalse(pools.ready(read, ()))

    def test_invalid_capacity_footprint_and_unreserved_access(self):
        for size in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                MemoryPoolManager({"a": size})
        pools = MemoryPoolManager({"a": 8})
        for requirements in ({"b": 1}, {"a": 9}, {"a": -1}, {"a": True}):
            with self.assertRaises(ValueError):
                pools.reserve(0, requirements)
        pools.reserve(0, {"a": 8})
        for access in (M(1, "a", 1), M(0, "a", 9)):
            with self.assertRaises(ValueError):
                pools.validate_accesses((access,), ())
