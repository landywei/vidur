"""Selected-work experiments using the serving engine's native events and heap.

This fixture replaces request arrival/selection and initial residency only.
It measures plan execution, not end-to-end request latency or TTFT. Dependencies,
resources, memory readiness and completion events are never mocked.
"""

from dataclasses import asdict, dataclass, field

from vidur.entities.execution_plan import MemoryAccess
from vidur.event_loop import EventLoop
from vidur.events.base_event import BaseEvent
from vidur.events.resource_execution_event import ResourceDispatchEvent
from vidur.scheduler.execution_plan_builder import BuildControls, SelectedWork
from vidur.scheduler.memory_pool_manager import MemoryPoolManager
from vidur.scheduler.resource_executor import ResourceExecutor
from vidur.types import EventType


@dataclass(frozen=True)
class SelectedExecutionFixture:
    work: SelectedWork
    selection_provenance: str
    reservations: tuple[MemoryAccess, ...] = ()
    initial_residency: tuple[MemoryAccess, ...] = ()
    memory_provenance: str = ""
    controls: BuildControls = field(default_factory=BuildControls)

    def __post_init__(self):
        if not self.selection_provenance.strip():
            raise ValueError("selected-work fixture requires selection provenance")
        if (
            self.reservations or self.initial_residency
        ) and not self.memory_provenance.strip():
            raise ValueError("memory fixtures require provenance")


class _SubmitSelectedWork(BaseEvent):
    def __init__(self, work, plan):
        super().__init__(0, EventType.BATCH_STAGE_ARRIVAL)
        self.work, self.plan = work, plan

    def handle_event(self, scheduler, metrics_store):
        replica = scheduler.get_replica_scheduler(0)
        replica._resource_executor.submit(
            (self.work.pipeline_stage, 0), self.plan, self.time
        )
        return [ResourceDispatchEvent(self.time, 0)]


class _SelectedReplica:
    def __init__(self, executor):
        self._resource_executor = executor
        self.completed_at = None

    def on_native_plan_complete(self, key, time):
        if self.completed_at is not None:
            raise RuntimeError("selected plan completed twice")
        self.completed_at = time
        return []

    def on_native_memory_release(self, time):
        # No admission queue exists in a selected-work fixture.
        return []


class _SelectedScheduler:
    def __init__(self, replica):
        self.replica = replica

    def get_replica_scheduler(self, replica_id):
        if replica_id != 0:
            raise ValueError("selected execution has one replica")
        return self.replica

    def is_empty(self):
        return (
            self.replica.completed_at is not None
            and not self.replica._resource_executor.plans
        )


class SelectedExecutionSimulation(EventLoop):
    def __init__(self, builder, fixture: SelectedExecutionFixture):
        super().__init__(write_json_trace=True, enable_chrome_trace=True)
        self.fixture = fixture
        self.plan = builder.build(fixture.work, fixture.controls)
        capacities = builder.policy.memory_pool_capacities()
        memory = MemoryPoolManager(capacities) if capacities else None
        ids = {r.request_id for r in fixture.work.requests}
        reservations = {}
        for access in fixture.reservations:
            pools = reservations.setdefault(access.request_id, {})
            if access.request_id not in ids or access.pool in pools:
                raise ValueError(
                    "memory fixture has foreign request or duplicate reservation"
                )
            pools[access.pool] = access.bytes
        if (reservations or fixture.initial_residency) and memory is None:
            raise ValueError("memory fixtures require declared pools")
        if memory is not None:
            for request_id, pools in reservations.items():
                memory.reserve(request_id, pools)
            memory.initialize_residency(fixture.initial_residency)
        executor = ResourceExecutor(builder.policy.resource_capacities(), memory)
        self._scheduler = _SelectedScheduler(_SelectedReplica(executor))
        self._metric_store = None
        self._add_event(_SubmitSelectedWork(fixture.work, self.plan))

    def diagnostics(self):
        """Preserve partial execution evidence even when the engine fails."""
        return {
            "plan": asdict(self.plan),
            "current_time_seconds": self._time,
            "completed_at_seconds": self._scheduler.replica.completed_at,
            "events": list(self._event_trace),
            "traceEvents": list(self._event_chrome_trace),
        }

    def report(self):
        replica = self._scheduler.replica
        if replica.completed_at is None:
            raise RuntimeError("selected execution is unfinished")
        memory = replica._resource_executor.memory_manager
        return {
            "schema_version": "vidur.selected-execution.v1",
            "scope": "selected-work execution; request admission and arrival are fixed",
            "unsupported_metrics": [
                "TTFT",
                "end-to-end request latency",
                "serving throughput",
            ],
            "fixture": asdict(self.fixture),
            "plan": asdict(self.plan),
            "completed_at_seconds": replica.completed_at,
            "memory": None
            if memory is None
            else {
                "capacity_bytes": memory.capacities,
                "used_bytes": memory.used,
                "peak_used_bytes": memory.peak_used,
                "resident_prefixes": [
                    {"request_id": r, "pool": p, "bytes": n}
                    for (r, p), n in sorted(memory.resident.items())
                ],
            },
            "events": self._event_trace,
            "traceEvents": self._event_chrome_trace,
        }
