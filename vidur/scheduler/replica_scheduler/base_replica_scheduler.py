from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional

from vidur.config import (
    BaseReplicaSchedulerConfig,
    BaseRequestGeneratorConfig,
    ReplicaConfig,
)
from vidur.entities import Batch, Replica, Request
from vidur.execution_time_predictor import BaseExecutionTimePredictor
from vidur.logger import init_logger
from vidur.scheduler.replica_stage_scheduler import ReplicaStageScheduler
from vidur.scheduler.utils.memory_planner import MemoryPlanner

logger = init_logger(__name__)


@dataclass(frozen=True)
class AdmissionView:
    """What an admission policy sees when a queued request could join.

    `resident`: snapshots (RequestState) of the requests holding memory,
    including those already selected into the batch being formed.
    `selected`: the new requests already chosen for that batch, with their
    token counts. `candidate` and `candidate_tokens`: the queue head and the
    tokens it would run. `probe(work)` returns the execution provider's
    PlanProbe of a SelectedWork run alone (None if infeasible); it never
    changes the simulation. Live requests and schedulers are never exposed.
    """

    resident: tuple
    selected: tuple
    candidate: Any
    candidate_tokens: int
    batch_size_cap: int
    max_micro_batch_size: int
    num_stages: int
    max_tokens: int
    probe: Callable[[Any], Any]


class AdmissionPolicy:
    """Decides whether the queue head joins the batch being formed.

    Called only after the scheduler's own capacity, memory and cap checks
    pass, so a policy can only admit less. `admit` returning False stops new
    admissions for this batch (FIFO order is kept). A replica with no
    resident request always admits its queue head, so a policy cannot stall
    a run.
    """

    def admit(self, view: AdmissionView) -> bool:
        return True


class BaseReplicaScheduler(ABC):
    def __init__(
        self,
        replica_config: ReplicaConfig,
        replica_scheduler_config: BaseReplicaSchedulerConfig,
        request_generator_config: BaseRequestGeneratorConfig,
        replica: Replica,
        num_stages: int,
        execution_time_predictor: BaseExecutionTimePredictor,
    ) -> None:
        self._config = replica_scheduler_config
        self._replica_config = replica_config
        self._request_generator_config = request_generator_config
        self._replica_id = replica.id
        self._num_stages = num_stages
        self._max_inflight_batches = self._config.max_inflight_batches
        if self._max_inflight_batches is None:
            self._max_inflight_batches = num_stages
        for value in (self._max_inflight_batches, self._config.micro_batch_size):
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError("native batch limits must be positive integers")
        if (
            self._config.micro_batch_size is not None
            and self._config.micro_batch_size > self._config.batch_size_cap
        ):
            raise ValueError("micro batch size exceeds resident request cap")
        self._execution_time_predictor = execution_time_predictor
        self._admission: Optional[AdmissionPolicy] = None
        # Requests holding memory, by id (for admission views).
        self._resident = {}
        self._memory_manager = None
        self._memory_requirements = {}
        self._dynamic_memory = False
        self._memory_evictions = 0
        self._memory_recovery = False
        self._resource_executor = None
        if execution_time_predictor.uses_execution_plans:
            from vidur.scheduler.memory_pool_manager import MemoryPoolManager
            from vidur.scheduler.resource_executor import ResourceExecutor

            capacities = execution_time_predictor.memory_pool_capacities()
            if capacities:
                if str(self._config.get_type()) not in {"vllm", "sarathi"}:
                    raise ValueError(
                        "native memory admission supports vllm and sarathi only"
                    )
                self._memory_manager = MemoryPoolManager(capacities)
            self._resource_executor = ResourceExecutor(
                execution_time_predictor.resource_capacities(), self._memory_manager
            )
        elif (
            self._config.max_inflight_batches is not None
            or self._config.micro_batch_size is not None
        ):
            raise ValueError("native batch limits require an execution-plan predictor")

        self._max_blocks_per_sequence = (
            self._request_generator_config.max_tokens // self._config.block_size
        )

        memory_planner = MemoryPlanner(self._replica_config, replica)

        if not self._config.num_blocks:
            self._config.num_blocks = (
                self._max_blocks_per_sequence * memory_planner.get_max_request_slots()
            )
        self._max_batch_size = min(
            memory_planner.get_max_batch_size(),
            self._config.batch_size_cap,
        )

        logger.debug(
            f"Obtained max batch size of {self._max_batch_size} for replica {self._replica_id}"
        )

        self._request_queue = []
        self._num_allocated_blocks = 0
        self._allocation_map = {}

        self._replica_stage_schedulers = {
            stage_id: ReplicaStageScheduler(
                replica.id,
                stage_id,
                stage_id == num_stages - 1,
                execution_time_predictor,
                self._resource_executor,
            )
            for stage_id in range(num_stages)
        }

    def set_admission_policy(self, policy: Optional[AdmissionPolicy]) -> None:
        self._admission = policy

    def probe(self, work):
        """The execution provider's PlanProbe of `work` run alone, or None.

        Observation only: no plan is built, bound or scheduled.
        """
        probe = getattr(self._execution_time_predictor, "probe", None)
        if probe is None:
            raise ValueError("the execution-time predictor provides no cost probe")
        return probe(work)

    def _admits(self, request, next_num_tokens, requests, num_tokens) -> bool:
        """Whether the admission policy lets queued `request` join the batch."""
        if self._admission is None or not self._resident:
            return True
        from vidur.scheduler.execution_plan_builder import RequestState

        view = AdmissionView(
            tuple(RequestState.from_request(r) for r in self._resident.values()),
            tuple(
                (RequestState.from_request(r), n) for r, n in zip(requests, num_tokens)
            ),
            RequestState.from_request(request),
            next_num_tokens,
            self._config.batch_size_cap,
            self._max_micro_batch_size,
            self._num_stages,
            self._request_generator_config.max_tokens,
            self.probe,
        )
        return bool(self._admission.admit(view))

    def on_native_plan_complete(self, key, time, plan=None, completion=None):
        from vidur.events.batch_stage_end_event import BatchStageEndEvent

        stage_id, batch_id = key
        stage = self.get_replica_stage_scheduler(stage_id)
        batch, batch_stage = stage.native_batches.pop(batch_id)
        observer = getattr(self._execution_time_predictor, "on_plan_complete", None)
        if completion is not None and observer is not None:
            observer(self._replica_id, stage_id, batch_id, plan, completion)
        return [
            BatchStageEndEvent(
                time,
                self._replica_id,
                stage_id,
                stage.is_last_stage,
                batch,
                batch_stage,
            )
        ]

    def on_native_memory_release(self, time):
        from vidur.events.replica_schedule_event import ReplicaScheduleEvent

        return [ReplicaScheduleEvent(time, self._replica_id)]

    @property
    def num_pending_requests(self) -> int:
        return len(self._request_queue)

    @property
    def replica_id(self) -> int:
        return self._replica_id

    @property
    def num_allocated_blocks(self) -> int:
        return self._num_allocated_blocks

    @property
    def memory_usage_percent(self) -> int:
        return (self._num_allocated_blocks * 100) / self._config.num_blocks

    def memory_pool_stats(self):
        if self._memory_manager is None:
            return None
        manager = self._memory_manager
        return {
            "policy": "step_growth"
            if self._dynamic_memory
            else "full_request_reservation",
            "preemptions": self._memory_evictions,
            "units": "bytes",
            "capacity": dict(manager.capacities),
            "used": dict(manager.used),
            "peak_used": dict(manager.peak_used),
            "resident_bytes": sum(manager.resident.values()),
            "reserved_requests": len(manager.reservations),
        }

    def is_empty(self) -> bool:
        return (
            self.num_pending_requests == 0
            and len(self._allocation_map) == 0
            and all(
                stage_scheduler.is_empty()
                for stage_scheduler in self._replica_stage_schedulers.values()
            )
        )

    def _get_request_next_num_tokens(self, request: Request) -> int:
        assert not request.completed

        if request.is_prefill_complete:
            return 1

        return request.num_prefill_tokens

    def add_request(self, request: Request) -> None:
        if self._memory_manager is not None:
            requirements = dict(
                self._execution_time_predictor.request_memory_requirements(request)
            )
            self._memory_manager.validate_requirements(requirements)
            if not requirements or not any(requirements.values()):
                raise ValueError("pool-aware requests must declare a nonzero footprint")
            self._memory_requirements[request.id] = requirements
        self._request_queue.append(request)

    def _native_memory_demand(self, request, next_num_tokens):
        bounds = self._memory_requirements[request.id]
        demand = self._execution_time_predictor.request_step_memory_requirements(
            request, next_num_tokens
        )
        if demand is None:
            return bounds
        self._dynamic_memory = True
        demand = dict(demand)
        self._memory_manager.validate_requirements(demand)
        if (
            not demand
            or not any(demand.values())
            or any(p not in bounds or n > bounds[p] for p, n in demand.items())
        ):
            raise ValueError(
                "step memory demand must be nonzero and within declared bounds"
            )
        return demand

    def _can_reserve_native_memory(self, request, next_num_tokens):
        return self._memory_manager is None or self._memory_manager.can_grow(
            request.id, self._native_memory_demand(request, next_num_tokens)
        )

    def _reserve_native_memory(self, request, next_num_tokens):
        self._resident[request.id] = request
        if self._memory_manager is not None:
            self._memory_manager.grow(
                request.id, self._native_memory_demand(request, next_num_tokens)
            )
            if request.warm_start:
                # A warm start's context is already resident where the
                # provider says; seed it once its reservation exists.
                self._memory_manager.seed_residency(
                    request.id,
                    self._execution_time_predictor.request_initial_residency(request),
                )

    def _preempt_request(self, request):
        # Only callers holding an idle request may discard its resident KV.
        self.free(request.id)
        request.restart()
        self._memory_evictions += 1
        self._memory_recovery = self._memory_manager is not None
        self._request_queue.insert(0, request)

    def _native_recovery_blocks_admission(self):
        # Drain resident survivors before replaying evicted work. Without this
        # barrier, small prefill chunks can repeatedly evict each other.
        if not self._allocation_map:
            self._memory_recovery = False
        return self._memory_recovery

    def get_replica_stage_scheduler(self, stage_id: int):
        return self._replica_stage_schedulers[stage_id]

    def can_allocate(self, num_blocks: int) -> bool:
        return self._config.num_blocks - self._num_allocated_blocks >= num_blocks

    def allocate(self, request_id: int, num_blocks: int) -> None:
        self._num_allocated_blocks += num_blocks
        if request_id not in self._allocation_map:
            self._allocation_map[request_id] = num_blocks
        else:
            self._allocation_map[request_id] += num_blocks

        assert self._num_allocated_blocks <= self._config.num_blocks

    def free(self, *request_ids: int) -> None:
        for request_id in request_ids:
            self._resident.pop(request_id, None)
            if self._memory_manager is not None:
                self._memory_manager.free(request_id)
            num_blocks = self._allocation_map.pop(request_id)
            self._num_allocated_blocks -= num_blocks

        assert self._num_allocated_blocks >= 0

    def free_batch(self, batch: Batch) -> None:
        self.free(*batch.request_ids)

    @abstractmethod
    def on_batch_end(self, batch: Batch) -> None:
        pass

    @abstractmethod
    def _get_next_batch(self) -> Batch:
        pass

    def on_schedule(self) -> list[Batch]:
        scheduled_batches = []
        while self._num_running_batches < self._max_inflight_batches:
            evictions = self._memory_evictions
            batch = self._get_next_batch()
            if not batch:
                if self._memory_evictions > evictions and not self._num_running_batches:
                    batch = self._get_next_batch()
                if not batch:
                    break
            scheduled_batches.append(batch)
            self._num_running_batches += 1
        return scheduled_batches
