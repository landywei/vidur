"""Costed analytical activities; seconds, explicit dependencies and resources.

Resources are replica-scoped names. Include a stage identifier in the name for
stage-private resources. Durations are fixed at submission, not cycle simulation.
"""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class MemoryAccess:
    """A valid prefix of a request's allocation in one named pool, in bytes."""

    request_id: int
    pool: str
    bytes: int

    def __post_init__(self):
        if (
            type(self.request_id) is not int
            or self.request_id < 0
            or not isinstance(self.pool, str)
            or not self.pool
            or type(self.bytes) is not int
            or self.bytes < 1
        ):
            raise ValueError(
                "memory access requires request ID, pool and positive integer bytes"
            )


@dataclass(frozen=True)
class ExecutionActivity:
    name: str
    duration_seconds: float
    resources: tuple[str, ...]
    dependencies: tuple[str, ...] = ()
    memory_reads: tuple[MemoryAccess, ...] = ()
    memory_writes: tuple[MemoryAccess, ...] = ()
    memory_releases: tuple[MemoryAccess, ...] = ()
    cost_provenance: str = ""
    cost_granularity: str = ""
    # Opaque identity of the plan this activity belongs to, for trace joins.
    plan_id: str = ""


@dataclass(frozen=True)
class ExecutionPlan:
    # Topological order also specifies deterministic ready-activity priority.
    activities: tuple[ExecutionActivity, ...]
    limitations: tuple[str, ...] = ()

    def validate(self, capacities):
        known = set()
        for activity in self.activities:
            if not activity.name or activity.name in known:
                raise ValueError("activity names must be nonempty and unique")
            if not isfinite(activity.duration_seconds) or activity.duration_seconds < 0:
                raise ValueError("activity duration must be finite nonnegative seconds")
            if not activity.resources or len(set(activity.resources)) != len(
                activity.resources
            ):
                raise ValueError("activities require distinct resource names")
            if any(resource not in capacities for resource in activity.resources):
                raise ValueError("activity references an undeclared resource")
            if not set(activity.dependencies) <= known:
                raise ValueError("dependencies must precede their consumers")
            known.add(activity.name)


@dataclass(frozen=True, eq=False)
class PlanStructure:
    """Activities shared by many plans; requests are given by position.

    Accesses are (request position, pool, bytes). Activity ``i`` is named
    ``a<i>`` and depends on earlier activities by index. Identity (not value)
    is the structure's key: equal structures may still be distinct objects.
    """

    durations: tuple
    resources: tuple
    dependencies: tuple
    reads: tuple
    writes: tuple
    cost_provenance: tuple
    cost_granularity: tuple
    plan_id: str = ""
    limitations: tuple = ()

    def __post_init__(self):
        count = len(self.durations)
        for values in (
            self.resources,
            self.dependencies,
            self.reads,
            self.writes,
            self.cost_provenance,
            self.cost_granularity,
        ):
            if len(values) != count:
                raise ValueError("plan structure fields must align by activity")
        for index, deps in enumerate(self.dependencies):
            if any(type(d) is not int or not 0 <= d < index for d in deps):
                raise ValueError("dependencies must precede their consumers")
        if not count and not self.limitations:
            raise ValueError("empty plans require an explicit omission boundary")

    def expand(self, request_ids):
        """The equivalent ExecutionPlan for one batch's request IDs."""
        return ExecutionPlan(
            tuple(
                ExecutionActivity(
                    f"a{i}",
                    self.durations[i],
                    self.resources[i],
                    tuple(f"a{d}" for d in self.dependencies[i]),
                    tuple(
                        MemoryAccess(request_ids[p], pool, n)
                        for p, pool, n in self.reads[i]
                    ),
                    tuple(
                        MemoryAccess(request_ids[p], pool, n)
                        for p, pool, n in self.writes[i]
                    ),
                    (),
                    self.cost_provenance[i],
                    self.cost_granularity[i],
                    self.plan_id,
                )
                for i in range(len(self.durations))
            ),
            self.limitations,
        )


@dataclass(frozen=True)
class CompactPlan:
    """A plan given as a shared structure and the batch's request IDs.

    The executor may run it compressed (vidur.scheduler.compressed_execution)
    or expand it into an ExecutionPlan; both give the same schedule.
    """

    structure: PlanStructure
    request_ids: tuple

    def expand(self):
        return self.structure.expand(self.request_ids)
