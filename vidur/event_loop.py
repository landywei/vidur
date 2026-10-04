"""The shared Vidur event heap for serving and selected-work experiments."""

import heapq
from math import isfinite

from vidur.events.base_event import BaseEvent


class EventLoop:
    def __init__(self, time_limit=0, write_json_trace=False, enable_chrome_trace=False):
        self._time = 0
        self._terminate = False
        self._time_limit = time_limit or float("inf")
        self._event_queue = []
        self._event_trace = []
        self._event_chrome_trace = []
        self._write_json_trace = write_json_trace
        self._enable_chrome_trace = enable_chrome_trace

    def run(self):
        while self._event_queue and not self._terminate:
            _, event = self._event_queue[0]
            if event.time > self._time_limit:
                self._time = self._time_limit
                self._terminate = True
                break
            heapq.heappop(self._event_queue)
            self._set_time(event.time)
            new_events = event.handle_event(self._scheduler, self._metric_store)
            self._add_events(new_events)
            if self._write_json_trace:
                self._event_trace.append(event.to_dict())
            if self._enable_chrome_trace:
                trace = event.to_chrome_trace()
                if trace:
                    self._event_chrome_trace.append(trace)
        if not self._scheduler.is_empty() and not self._terminate:
            raise RuntimeError(
                "simulation has unfinished requests with no events; check admission "
                "capacity, execution dependencies and memory residency"
            )

    def _add_event(self, event: BaseEvent):
        if not isfinite(event.time) or event.time < self._time:
            raise ValueError("events must have finite, nondecreasing timestamps")
        heapq.heappush(self._event_queue, (event._priority_number, event))

    def _add_events(self, events):
        for event in events:
            self._add_event(event)

    def _set_time(self, time):
        self._time = time
        if self._time > self._time_limit:
            self._terminate = True
