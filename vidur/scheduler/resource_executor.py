"""Online, nonpreemptive list scheduling shared by all stages of a replica.

Only ready work reserves capacity. Completion releases capacity before the next
dispatch. No future calendar reservations or paper-specific overlap formulas.
"""

from dataclasses import dataclass, field
from math import isfinite
from weakref import WeakKeyDictionary

from vidur.entities.execution_plan import CompactPlan, ExecutionPlan
from vidur.scheduler.compressed_execution import replay, simulate


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
    # "events" (activity events), "simulated" or "replayed" (compressed).
    execution: str = "events"


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
        self.resource_set = {r for a in self.plan.activities for r in a.resources}
        self.memory_keys = {
            (x.request_id, x.pool)
            for a in self.plan.activities
            for x in a.memory_reads + a.memory_writes + a.memory_releases
        }
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
        first = self.start[path[0]]
        return PlanCompletion(
            self.submitted_at,
            self.finish[last],
            tuple(self.start),
            tuple(self.finish),
            tuple(self.binding),
            tuple(path),
            # Unknown when a subclass dispatches without recording starts.
            0.0 if first is None else first - self.submitted_at,
        )


class CompressedState:
    """A plan run compressed: its whole schedule is known at submission."""

    def __init__(self, plan, submitted_at, schedule, recipe, info, keys, writes, kind):
        self.plan = plan
        self.submitted_at = submitted_at
        self.finished_at = schedule.finished_at
        self.schedule = schedule
        self.recipe = recipe
        self.resource_set = info.resource_set
        self.memory_keys = keys
        self.writes = writes
        self.execution = kind
        # Never dispatched by the event loop.
        self.ready = {}

    def completion(self):
        path = self.recipe.critical_path
        return PlanCompletion(
            self.submitted_at,
            self.finished_at,
            self.schedule.start,
            self.schedule.finish,
            self.recipe.binding,
            path,
            self.schedule.start[path[0]] - self.submitted_at if path else 0.0,
            self.execution,
        )


class _StructureInfo:
    """Per-structure facts checked once: resources, accesses, validity."""

    def __init__(self, structure, capacities, pools):
        for duration in structure.durations:
            if not isfinite(duration) or duration < 0:
                raise ValueError("activity duration must be finite nonnegative seconds")
        for names in structure.resources:
            if not names or len(set(names)) != len(names):
                raise ValueError("activities require distinct resource names")
            if any(r not in capacities for r in names):
                raise ValueError("activity references an undeclared resource")
        accesses = []
        for activity in (structure.reads, structure.writes):
            for items in activity:
                seen = set()
                for p, pool, n in items:
                    if (p, pool) in seen:
                        raise ValueError("duplicate memory access in one activity")
                    seen.add((p, pool))
                    if type(n) is not int or n < 1 or type(p) is not int or p < 0:
                        raise ValueError("invalid memory access")
                    if pools is None:
                        raise ValueError("memory accesses require native memory pools")
                    if pool not in pools:
                        raise ValueError("memory access outside declared pools")
                    accesses.append((p, pool, n))
        self.resource_set = {r for names in structure.resources for r in names}
        self.accesses = tuple(accesses)
        self.positions = {(p, pool) for p, pool, _ in accesses}
        self.reads = tuple(x for items in structure.reads for x in items)
        self.writes = tuple(x for items in structure.writes for x in items)
        self.max_position = max((p for p, _, _ in accesses), default=-1)


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
        # Weak: a structure's recipes live as long as its owner keeps it.
        self._structures = WeakKeyDictionary()
        self._recipes = WeakKeyDictionary()
        self.compression = dict.fromkeys(
            ("expanded", "simulated", "replayed", "replay_rejected"), 0
        )

    def _advance(self, time):
        if not isfinite(time) or time < self.time:
            raise ValueError("executor time must be finite and monotonic")
        self.time = time

    def submit(self, key, plan, time):
        """Accept a plan; for a compressed plan return its completion time.

        A CompactPlan runs compressed when it is isolated (see
        compressed_execution); otherwise it is expanded and runs by events.
        """
        self._advance(time)
        if key in self.plans:
            raise ValueError("plan already active")
        if isinstance(plan, CompactPlan):
            state = self._compress(plan, time)
            if state is not None:
                self.plans[key] = state
                return state.finished_at
            self.compression["expanded"] += 1
            plan = plan.expand()
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
        state = PlanState(plan, time)
        for other in self.plans.values():
            if isinstance(other, CompressedState) and (
                state.resource_set & other.resource_set
                or state.memory_keys & other.memory_keys
            ):
                raise ValueError("a plan overlaps a plan running compressed")
        self.plans[key] = state
        return None

    def _compress(self, plan, time):
        structure, ids = plan.structure, plan.request_ids
        info = self._structures.get(structure)
        if info is None:
            pools = (
                None
                if self.memory_manager is None
                else self.memory_manager.capacities
            )
            info = self._structures[structure] = _StructureInfo(
                structure, self.capacities, pools
            )
        if info.max_position >= len(ids):
            raise ValueError("memory access outside selected work")
        for other in self.plans.values():
            if info.resource_set & other.resource_set:
                return None
        keys = {(ids[p], pool) for p, pool in info.positions}
        manager = self.memory_manager
        bits = ()
        if keys:
            for p, pool, n in info.accesses:
                reserved = manager.reservations.get(ids[p], {}).get(pool)
                if reserved is None or n > reserved:
                    raise ValueError("memory access exceeds a request's reserved bytes")
            if any(k in manager.writers or manager.readers.get(k) for k in keys):
                return None
            if any(keys & other.memory_keys for other in self.plans.values()):
                return None
            bits = tuple(
                manager.resident[ids[p], pool] >= n for p, pool, n in info.reads
            )
        recipes = self._recipes.setdefault(structure, {})
        recipe = recipes.get(bits)
        schedule = None
        if recipe is not None:
            schedule = replay(recipe, structure.durations, time)
            if schedule is None:
                self.compression["replay_rejected"] += 1
        execution = "replayed"
        if schedule is None:
            resident = {k: manager.resident[k] for k in keys} if keys else {}
            result = simulate(structure, ids, time, self.capacities, resident)
            if result is None:
                return None
            schedule, recipe = result
            recipes[bits] = recipe
            execution = "simulated"
        self.compression[execution] += 1
        if keys:
            manager._execution_started = True
        writes = [((ids[p], pool), n) for p, pool, n in info.writes]
        return CompressedState(
            plan, time, schedule, recipe, info, keys, writes, execution
        )

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

    def finished(self, key, time=None):
        state = self.plans[key]
        if isinstance(state, CompressedState):
            return state.finished_at <= (self.time if time is None else time)
        return len(state.completed) == len(state.plan.activities)

    def retire(self, key, time=None):
        """Remove a finished plan and return its observed `PlanCompletion`."""
        if time is not None:
            self._advance(time)
        if not self.finished(key):
            raise ValueError("cannot retire an unfinished plan")
        state = self.plans.pop(key)
        if isinstance(state, CompressedState) and state.writes:
            # Residency becomes visible to other plans only now; while the
            # plan ran, only its own activities could read these keys.
            resident = self.memory_manager.resident
            for k, n in state.writes:
                resident[k] = max(resident[k], n)
        return state.completion()
