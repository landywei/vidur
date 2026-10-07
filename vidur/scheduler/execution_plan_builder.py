"""Declarative analytical plan construction, separate from admission and dispatch.

Policies receive immutable selected work, never live requests or schedulers.
They declare placement and structure; bindings supply seconds and evidence.
Only ResourceExecutor and the event loop decide start/finish times.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Protocol

from vidur.entities.execution_plan import (
    CostAdjustment,
    ExecutionActivity,
    ExecutionPlan,
    MemoryAccess,
)


@dataclass(frozen=True)
class RequestState:
    request_id: int
    num_processed_tokens: int
    num_prefill_tokens: int
    is_prefill_complete: bool

    @classmethod
    def from_request(cls, request):
        return cls(
            request.id,
            request.num_processed_tokens,
            request.num_prefill_tokens,
            request.is_prefill_complete,
        )


@dataclass(frozen=True)
class SelectedRequest(RequestState):
    next_num_tokens: int


@dataclass(frozen=True)
class SelectedWork:
    requests: tuple[SelectedRequest, ...]
    pipeline_stage: int

    @classmethod
    def from_batch(cls, batch, pipeline_stage):
        if len(batch.requests) != len(batch.num_tokens):
            raise ValueError("selected requests and token counts must align")
        return cls(
            tuple(
                SelectedRequest(
                    r.id,
                    r.num_processed_tokens,
                    r.num_prefill_tokens,
                    r.is_prefill_complete,
                    count,
                )
                for r, count in zip(batch.requests, batch.num_tokens)
            ),
            pipeline_stage,
        )

    @property
    def num_prefill_tokens(self):
        return sum(
            r.next_num_tokens for r in self.requests if not r.is_prefill_complete
        )

    def uniform_decode_shape(self):
        decode = [r for r in self.requests if r.is_prefill_complete]
        if not decode:
            return None
        if len(decode) != len(self.requests):
            raise ValueError("expected a decode-only batch; got a mixed batch")
        contexts = {r.num_processed_tokens for r in decode}
        if len(contexts) != 1:
            raise ValueError("expected a uniform batch; KV lengths differ")
        return len(decode), contexts.pop()

    def __post_init__(self):
        if type(self.pipeline_stage) is not int or self.pipeline_stage < 0:
            raise ValueError("pipeline stage must be a nonnegative integer")
        if type(self.requests) is not tuple or any(
            not isinstance(r, SelectedRequest) for r in self.requests
        ):
            raise ValueError("selected requests must be an immutable tuple of snapshots")
        ids = [r.request_id for r in self.requests]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("selected work requires distinct requests")
        for r in self.requests:
            if (
                any(
                    type(n) is not int or n < 0
                    for n in (
                        r.request_id,
                        r.num_processed_tokens,
                        r.num_prefill_tokens,
                    )
                )
                or type(r.next_num_tokens) is not int
                or r.next_num_tokens < 1
            ):
                raise ValueError("invalid selected token counts")
            if type(r.is_prefill_complete) is not bool:
                raise ValueError("prefill completion must be a boolean")
            if r.is_prefill_complete:
                if r.num_processed_tokens <= r.num_prefill_tokens:
                    raise ValueError("completed prefill must include the first output token")
            elif (
                r.num_processed_tokens >= r.num_prefill_tokens
                or r.num_processed_tokens + r.next_num_tokens > r.num_prefill_tokens
            ):
                raise ValueError("prefill selection exceeds remaining prompt tokens")
            if r.is_prefill_complete and r.next_num_tokens != 1:
                raise ValueError("decode selection requires one token per request")


@dataclass(frozen=True)
class ActivityDeclaration:
    name: str
    cost_key: str
    resources: tuple[str, ...]
    dependencies: tuple[str, ...] = ()
    memory_reads: tuple[MemoryAccess, ...] = ()
    memory_writes: tuple[MemoryAccess, ...] = ()
    memory_releases: tuple[MemoryAccess, ...] = ()


@dataclass(frozen=True)
class ExecutionPhase:
    """Concurrent serial branches; the next phase waits for every branch end."""

    branches: tuple[tuple[ActivityDeclaration, ...], ...]


def expand_phases(phases):
    """Expand barriers without scheduling, costs, or architecture knowledge."""
    from dataclasses import replace

    declarations = []
    previous = ()
    for phase in phases:
        ends = []
        for branch in phase.branches:
            dependencies = previous
            for activity in branch:
                dependencies = tuple(
                    dict.fromkeys(dependencies + activity.dependencies)
                )
                declarations.append(replace(activity, dependencies=dependencies))
                dependencies = (activity.name,)
            if branch:
                ends.extend(dependencies)
        if ends:
            previous = tuple(ends)
    return tuple(declarations)


@dataclass(frozen=True)
class PlanDeclaration:
    activities: tuple[ActivityDeclaration, ...]
    # Empty/omitted work needs an explicit evidence boundary too.
    limitations: tuple[str, ...] = ()
    # Optional identity copied onto every activity and its trace events.
    plan_id: str = ""


@dataclass(frozen=True)
class BoundCost:
    seconds: float
    provenance: str
    granularity: str

    def __post_init__(self):
        if not isfinite(self.seconds) or self.seconds < 0:
            raise ValueError("bound costs must be finite nonnegative seconds")
        if not self.provenance.strip() or not self.granularity.strip():
            raise ValueError("bound costs require provenance and granularity")


@dataclass(frozen=True)
class CostMask:
    cost_key: str
    reason: str


@dataclass(frozen=True)
class CostFixture:
    cost_key: str
    seconds: float
    provenance: str
    reason: str


@dataclass(frozen=True)
class BuildControls:
    """Only explicit costs may be changed; topology and resources remain real."""

    masks: tuple[CostMask, ...] = ()
    fixtures: tuple[CostFixture, ...] = ()

    def apply(self, costs, declarations):
        declared = {a.cost_key for a in declarations}
        seen = set()
        adjusted = dict(costs)
        records = []
        for entry in self.masks + self.fixtures:
            if entry.cost_key in seen or entry.cost_key not in declared:
                raise ValueError("duplicate or unused cost control")
            if not entry.reason.strip():
                raise ValueError("cost controls require a reason")
            seen.add(entry.cost_key)
            if entry.cost_key not in costs:
                raise ValueError(f"missing cost binding: {entry.cost_key}")
            original = costs[entry.cost_key]
            masked = isinstance(entry, CostMask)
            replacement = BoundCost(
                0.0 if masked else entry.seconds,
                f"Explicit cost mask: {entry.reason}" if masked else entry.provenance,
                original.granularity,
            )
            adjusted[entry.cost_key] = replacement
            records.append(
                CostAdjustment(
                    entry.cost_key,
                    "mask" if masked else "fixture",
                    entry.reason,
                    original.seconds,
                    replacement.seconds,
                    original.provenance,
                    replacement.provenance,
                )
            )
        return adjusted, tuple(records)


class ArchitecturePolicy(Protocol):
    def declare(self, work: SelectedWork) -> PlanDeclaration: ...

    def resource_capacities(self) -> dict[str, int]: ...

    def memory_pool_capacities(self) -> dict[str, int]: ...


class CostBinding(Protocol):
    def bind(self, work: SelectedWork) -> dict[str, BoundCost]:
        """Validate the calibration domain and return evidenced costs by key."""
        ...


class MemoryPolicy(Protocol):
    """Demand only; allocation, readiness and eviction belong to Vidur.

    A None step demand retains the fixed reservation contract. No realized
    future output length is exposed by RequestState.
    """

    def request_memory_requirements(self, request: RequestState) -> dict[str, int]: ...

    def request_step_memory_requirements(
        self, request: RequestState, next_num_tokens: int
    ) -> dict[str, int] | None: ...


class ExecutionPlanBuilder:
    def __init__(self, policy: ArchitecturePolicy, costs: CostBinding):
        self.policy = policy
        self.costs = costs

    def build(
        self, work: SelectedWork, controls: BuildControls | None = None
    ) -> ExecutionPlan:
        declaration = self.policy.declare(work)
        # Domain validation always precedes masks/fixtures; controls cannot
        # manufacture support for an uncalibrated workload.
        costs = self.costs.bind(work)
        costs, adjustments = (controls or BuildControls()).apply(
            costs, declaration.activities
        )
        capacities = self.policy.resource_capacities()
        pools = self.policy.memory_pool_capacities()
        ids = {r.request_id for r in work.requests}
        activities = []
        for activity in declaration.activities:
            if activity.cost_key not in costs:
                raise ValueError(f"missing cost binding: {activity.cost_key}")
            for access in (
                activity.memory_reads
                + activity.memory_writes
                + activity.memory_releases
            ):
                if access.request_id not in ids or access.pool not in pools:
                    raise ValueError(
                        "memory access outside selected work or declared pools"
                    )
            cost = costs[activity.cost_key]
            activities.append(
                ExecutionActivity(
                    activity.name,
                    cost.seconds,
                    activity.resources,
                    activity.dependencies,
                    activity.memory_reads,
                    activity.memory_writes,
                    activity.memory_releases,
                    cost.provenance,
                    cost.granularity,
                    declaration.plan_id,
                )
            )
        if not activities and not declaration.limitations:
            raise ValueError("empty plans require an explicit omission boundary")
        plan = ExecutionPlan(tuple(activities), declaration.limitations, adjustments)
        plan.validate(capacities)
        return plan
