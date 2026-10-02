"""Native resource completion and dispatch events on Vidur's event heap."""

from vidur.entities import BatchStage
from vidur.events.base_event import BaseEvent
from vidur.types import EventType


def submit_native_batches(time, replica_id, stage_id, stage):
    executor = stage.resource_executor
    while stage._batch_queue:
        batch = stage._batch_queue[0]
        plan = stage._execution_time_predictor.get_execution_plan(batch, stage_id)
        for activity in plan.activities:
            if any(
                access.request_id not in batch.request_ids
                for access in activity.memory_reads + activity.memory_writes
            ):
                raise ValueError(
                    "activity memory access references a request outside its batch"
                )
        executor.submit((stage_id, batch.id), plan, time)
        stage._batch_queue.pop(0)
        batch_stage = BatchStage(
            batch.id, replica_id, stage_id, None, None, batch.requests, batch.num_tokens
        )
        batch_stage.on_schedule(time)
        stage.native_batches[batch.id] = (batch, batch_stage)
    return [ResourceDispatchEvent(time, replica_id)]


class ResourceDispatchEvent(BaseEvent):
    def __init__(self, time, replica_id):
        super().__init__(time, EventType.RESOURCE_DISPATCH)
        self.replica_id = replica_id

    def handle_event(self, scheduler, metrics_store):
        from vidur.events.batch_stage_end_event import BatchStageEndEvent

        replica = scheduler.get_replica_scheduler(self.replica_id)
        executor = replica._resource_executor
        events = []
        for key in list(executor.plans):
            if not executor.finished(key):
                continue
            stage_id, batch_id = key
            stage = replica.get_replica_stage_scheduler(stage_id)
            batch, batch_stage = stage.native_batches.pop(batch_id)
            executor.retire(key)
            events.append(
                BatchStageEndEvent(
                    self.time,
                    self.replica_id,
                    stage_id,
                    stage.is_last_stage,
                    batch,
                    batch_stage,
                )
            )
        for key, activity, start, finish in executor.dispatch(self.time):
            events.append(
                ResourceActivityEndEvent(finish, self.replica_id, key, activity, start)
            )
        return events

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": int(self._event_type),
            "replica_id": self.replica_id,
        }


class ResourceActivityEndEvent(BaseEvent):
    def __init__(self, time, replica_id, key, activity, start):
        super().__init__(time, EventType.RESOURCE_ACTIVITY_END)
        self.replica_id = replica_id
        self.key = key
        self.activity = activity
        self.start = start

    def handle_event(self, scheduler, metrics_store):
        executor = scheduler.get_replica_scheduler(self.replica_id)._resource_executor
        executor.complete(self.key, self.activity.name, self.time)
        return [ResourceDispatchEvent(self.time, self.replica_id)]

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": int(self._event_type),
            "replica_id": self.replica_id,
            "stage_id": self.key[0],
            "batch_id": self.key[1],
            "activity": self.activity.name,
            "start_seconds": self.start,
            "resources": list(self.activity.resources),
            "dependencies": list(self.activity.dependencies),
            "memory_reads": [vars(access) for access in self.activity.memory_reads],
            "memory_writes": [vars(access) for access in self.activity.memory_writes],
        }

    def to_chrome_trace(self):
        return {
            "name": self.activity.name,
            "ph": "X",
            "pid": self.replica_id,
            "tid": "+".join(self.activity.resources),
            "ts": self.start * 1e6,
            "dur": (self.time - self.start) * 1e6,
            "args": self.to_dict(),
        }
