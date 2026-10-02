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


@dataclass(frozen=True)
class ExecutionPlan:
    # Topological order also specifies deterministic ready-activity priority.
    activities: tuple[ExecutionActivity, ...]

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
