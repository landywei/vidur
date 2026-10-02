"""Online, nonpreemptive list scheduling shared by all stages of a replica.

Only ready work reserves capacity. Completion releases capacity before the next
dispatch. No future calendar reservations or paper-specific overlap formulas.
"""

from dataclasses import dataclass, field
from math import isfinite

from vidur.entities.execution_plan import ExecutionPlan


@dataclass
class PlanState:
    plan: ExecutionPlan
    submitted_at: float
    started: set = field(default_factory=set)
    completed: set = field(default_factory=set)


class ResourceExecutor:
    def __init__(self, capacities, memory_manager=None):
        self.memory_manager = memory_manager
        self.capacities = dict(capacities)
        if not self.capacities or any(
            not isinstance(name, str) or not name or type(value) is not int or value < 1
            for name, value in self.capacities.items()
        ):
            raise ValueError("resource capacities must be named positive integers")
        self.used = dict.fromkeys(self.capacities, 0)
        self.plans = {}
        self.running = {}
        self.time = 0.0

    def _advance(self, time):
        if not isfinite(time) or time < self.time:
            raise ValueError("executor time must be finite and monotonic")
        self.time = time

    def submit(self, key, plan, time):
        self._advance(time)
        if key in self.plans:
            raise ValueError("plan already active")
        plan.validate(self.capacities)
        for activity in plan.activities:
            if (
                activity.memory_reads
                or activity.memory_writes
                or activity.memory_releases
            ):
                if self.memory_manager is None:
                    raise ValueError("memory accesses require native memory pools")
                self.memory_manager.validate_accesses(
                    activity.memory_reads,
                    activity.memory_writes,
                    activity.memory_releases,
                )
        self.plans[key] = PlanState(plan, time)

    def dispatch(self, time):
        self._advance(time)
        started = []
        # Dict insertion order is FIFO across plans; activity order breaks ties.
        for key, state in self.plans.items():
            for activity in state.plan.activities:
                if activity.name in state.started:
                    continue
                if not set(activity.dependencies) <= state.completed:
                    continue
                if any(self.used[r] >= self.capacities[r] for r in activity.resources):
                    continue
                if self.memory_manager is not None and not self.memory_manager.ready(
                    activity.memory_reads,
                    activity.memory_writes,
                    activity.memory_releases,
                ):
                    continue
                finish = time + activity.duration_seconds
                if not isfinite(finish):
                    raise ValueError("activity completion time overflow")
                if self.memory_manager is not None:
                    self.memory_manager.begin(
                        activity.memory_reads,
                        activity.memory_writes,
                        activity.memory_releases,
                    )
                state.started.add(activity.name)
                for resource in activity.resources:
                    self.used[resource] += 1
                self.running[key, activity.name] = (activity, time, finish)
                started.append((key, activity, time, finish))
        return started

    def complete(self, key, name, time):
        self._advance(time)
        activity, _start, finish = self.running[key, name]
        if time != finish:
            raise ValueError("activity completed at an unexpected time")
        if self.memory_manager is not None:
            self.memory_manager.complete(
                activity.memory_reads, activity.memory_writes, activity.memory_releases
            )
        del self.running[key, name]
        for resource in activity.resources:
            self.used[resource] -= 1
        self.plans[key].completed.add(name)

    def finished(self, key):
        state = self.plans[key]
        return len(state.completed) == len(state.plan.activities)

    def retire(self, key):
        if not self.finished(key):
            raise ValueError("cannot retire an unfinished plan")
        del self.plans[key]
