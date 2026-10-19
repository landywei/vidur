"""Admission hook and cost probe: observation for policies, never accounting."""

import unittest
from types import SimpleNamespace

from vidur.execution_time_predictor.declarative_execution_provider import (
    NativeExecutionProvider,
)
from vidur.scheduler.execution_plan_builder import ExecutionPlanBuilder, PlanProbe
from vidur.scheduler.replica_scheduler.base_replica_scheduler import (
    AdmissionPolicy,
    AdmissionView,
    BaseReplicaScheduler,
)


def request(i, processed=0, prompt=8, done=False):
    return SimpleNamespace(
        id=i,
        num_processed_tokens=processed,
        num_prefill_tokens=prompt,
        is_prefill_complete=done,
    )


class Recording(AdmissionPolicy):
    def __init__(self, answer):
        self.answer, self.views = answer, []

    def admit(self, view):
        self.views.append(view)
        return self.answer


def scheduler(policy, resident):
    probe = PlanProbe(1.0, 2.0)
    return SimpleNamespace(
        _admission=policy,
        _resident={r.id: r for r in resident},
        _config=SimpleNamespace(batch_size_cap=4),
        _max_micro_batch_size=4,
        _num_stages=1,
        _request_generator_config=SimpleNamespace(max_tokens=64),
        probe=lambda work: probe,
    )


class AdmissionTests(unittest.TestCase):
    def test_default_policy_admits(self):
        self.assertTrue(AdmissionPolicy().admit(None))

    def test_no_policy_or_no_resident_request_always_admits(self):
        admits = BaseReplicaScheduler._admits
        self.assertTrue(admits(scheduler(None, [request(1)]), request(2), 8, [], []))
        policy = Recording(False)
        self.assertTrue(admits(scheduler(policy, []), request(2), 8, [], []))
        self.assertEqual(policy.views, [])

    def test_policy_sees_snapshots_and_decides(self):
        policy = Recording(False)
        resident = [request(1, 9, 8, True), request(2)]
        stub = scheduler(policy, resident)
        self.assertFalse(
            BaseReplicaScheduler._admits(stub, request(3), 8, [resident[1]], [8])
        )
        (view,) = policy.views
        self.assertIsInstance(view, AdmissionView)
        self.assertEqual([r.request_id for r in view.resident], [1, 2])
        self.assertEqual(view.selected[0][1], 8)
        self.assertEqual(view.candidate.request_id, 3)
        self.assertEqual((view.batch_size_cap, view.max_tokens), (4, 64))
        self.assertEqual(view.probe(None).makespan_seconds, 1.0)


class ProbeTests(unittest.TestCase):
    def test_provider_forwards_to_the_policy_probe(self):
        calls = []

        class Policy:
            def probe(self, work):
                calls.append(work)
                return PlanProbe(3.0, 4.0, (("pim", 3.0),))

        provider = NativeExecutionProvider(ExecutionPlanBuilder(Policy(), None))
        self.assertEqual(provider.probe("w").critical_path_seconds, (("pim", 3.0),))
        self.assertEqual(calls, ["w"])

    def test_a_policy_without_probe_cannot_be_probed(self):
        provider = NativeExecutionProvider(ExecutionPlanBuilder(object(), None))
        with self.assertRaises(ValueError):
            provider.probe("w")


if __name__ == "__main__":
    unittest.main()
