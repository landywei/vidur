"""Native resource completion and dispatch events on Vidur's event heap."""

from vidur.entities import BatchStage
from vidur.events.base_event import BaseEvent
from vidur.types import EventType


def submit_native_batches(time, replica_id, stage_id, stage):
    executor = stage.resource_executor
    while stage._batch_queue:
        batch = stage._batch_queue[0]
        provider = stage._execution_time_predictor
        from vidur.scheduler.execution_plan_builder import SelectedWork

        plan = provider.execution_plan_builder.build(
            SelectedWork.from_batch(batch, stage_id)
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
        replica = scheduler.get_replica_scheduler(self.replica_id)
        executor = replica._resource_executor
        events = []
        for key in list(executor.plans):
            if not executor.finished(key):
                continue
            executor.retire(key)
            events.extend(replica.on_native_plan_complete(key, self.time))
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
        events = [ResourceDispatchEvent(self.time, self.replica_id)]
        if self.activity.memory_releases:
            replica = scheduler.get_replica_scheduler(self.replica_id)
            events.extend(replica.on_native_memory_release(self.time))
        return events

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
            "cost_provenance": self.activity.cost_provenance,
            "cost_granularity": self.activity.cost_granularity,
            "memory_reads": [vars(access) for access in self.activity.memory_reads],
            "memory_writes": [vars(access) for access in self.activity.memory_writes],
            "memory_releases": [
                vars(access) for access in self.activity.memory_releases
            ],
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
