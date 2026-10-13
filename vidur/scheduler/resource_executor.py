"""Online, nonpreemptive list scheduling shared by all stages of a replica.

Only ready work reserves capacity. Completion releases capacity before the next
dispatch. No future calendar reservations or paper-specific overlap formulas.
"""

from dataclasses import dataclass, field
from math import isfinite

from vidur.entities.execution_plan import ExecutionPlan


@dataclass(frozen=True)
class PlanCompletion:
    """Observed schedule of one retired plan, indexed like its activities.

    ``binding[i]`` is the activity of the same plan whose completion let
    activity ``i`` start (its start equals that completion time), or None when
    it started at submission or after waiting on another plan. Among the
    plan's completions at that instant, a dependency is preferred, then an
    activity sharing a resource, then any; ties go to the later activity.
    ``critical_path`` follows binding activities back from the last
    completion (ties: the later activity); ``wait_seconds`` is the time
    between submission and the start of its first activity.
    """

    submitted_at: float
    finished_at: float
    start: tuple
    finish: tuple
    binding: tuple
    critical_path: tuple
    wait_seconds: float


@dataclass
class PlanState:
    plan: ExecutionPlan
    submitted_at: float
    started: set = field(default_factory=set)
    completed: set = field(default_factory=set)
    ready: dict = field(default_factory=dict)
    dependents: dict = field(default_factory=dict)
    remaining_dependencies: list = field(default_factory=list)
    index: dict = field(default_factory=dict)
    start: list = field(default_factory=list)
    finish: list = field(default_factory=list)
    binding: list = field(default_factory=list)
    # The plan's latest completion instant and the activities completed then.
    last_time: float = None
    last_completed: list = field(default_factory=list)

    def __post_init__(self):
        self.dependents = {a.name: [] for a in self.plan.activities}
        count = len(self.plan.activities)
        self.start = [None] * count
        self.finish = [None] * count
        self.binding = [None] * count
        for index, activity in enumerate(self.plan.activities):
            self.index[activity.name] = index
            self.remaining_dependencies.append(len(activity.dependencies))
            if not activity.dependencies:
                self.ready[index] = activity
            for name in activity.dependencies:
                self.dependents[name].append(index)

    def bind(self, index, time):
        """Record the start of ``index`` and the completion that released it."""
        self.start[index] = time
        if self.last_time != time:
            return
        activity = self.plan.activities[index]
        candidates = [
            d
            for d in (self.index[name] for name in activity.dependencies)
            if self.finish[d] == time
        ]
        if not candidates:
            resources = set(activity.resources)
            candidates = [
                j
                for j in self.last_completed
                if resources.intersection(self.plan.activities[j].resources)
            ]
        self.binding[index] = max(candidates or self.last_completed)

    def completion(self):
        count = len(self.plan.activities)
        if not count:
            return PlanCompletion(
                self.submitted_at, self.submitted_at, (), (), (), (), 0.0
            )
        last = max(range(count), key=lambda i: (self.finish[i], i))
        path = [last]
        while self.binding[path[-1]] is not None:
            path.append(self.binding[path[-1]])
        path.reverse()
        return PlanCompletion(
            self.submitted_at,
            self.finish[last],
            tuple(self.start),
            tuple(self.finish),
            tuple(self.binding),
            tuple(path),
            self.start[path[0]] - self.submitted_at,
        )


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
            # Only dependency-ready work can start. Sorting preserves the
            # original activity-order priority after out-of-order completions.
            for index in sorted(state.ready):
                activity = state.ready[index]
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
                state.bind(index, time)
                del state.ready[index]
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
        state = self.plans[key]
        state.completed.add(name)
        index = state.index[name]
        state.finish[index] = time
        if state.last_time != time:
            state.last_time = time
            state.last_completed = []
        state.last_completed.append(index)
        for index in state.dependents[name]:
            state.remaining_dependencies[index] -= 1
            if state.remaining_dependencies[index] == 0:
                state.ready[index] = state.plan.activities[index]

    def finished(self, key):
        state = self.plans[key]
        return len(state.completed) == len(state.plan.activities)

    def retire(self, key):
        """Remove a finished plan and return its observed `PlanCompletion`."""
        if not self.finished(key):
            raise ValueError("cannot retire an unfinished plan")
        return self.plans.pop(key).completion()
