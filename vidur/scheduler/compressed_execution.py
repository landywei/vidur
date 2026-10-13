"""Exact compressed execution of an isolated plan.

A plan is isolated while no other active plan uses its resources or the
memory (request, pool) keys it accesses. Its schedule is then a function of
its submission time, its structure and, for each read, whether the initial
resident prefix already covers it. `simulate` applies ResourceExecutor's
dispatch rules to the plan alone, with the same float operations (finish =
dispatch instant + duration), and records the control flow as a `Recipe`:
the instant at which each activity starts and completes. `replay` recomputes
the times for another submission time and accepts them only if the instants
stay strictly increasing and every completion falls exactly on its recorded
instant. Every dispatch decision depends only on which activities have
completed at each instant, so an accepted replay is the schedule `simulate`
(and the event loop) would produce.
"""

import heapq
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class Recipe:
    start_instant: tuple
    end_instant: tuple
    # Per instant after submission: an activity completing at it.
    representative: tuple
    binding: tuple
    critical_path: tuple


@dataclass(frozen=True)
class Schedule:
    start: tuple
    finish: tuple
    finished_at: float


def simulate(structure, request_ids, time, capacities, resident):
    """Schedule and recipe of an isolated plan, or None if it cannot finish
    alone (the event loop then reports the stall)."""
    durations = structure.durations
    resources = structure.resources
    deps = structure.dependencies
    count = len(durations)
    dependents = [[] for _ in range(count)]
    for index, required in enumerate(deps):
        for d in required:
            dependents[d].append(index)
    remaining = [len(required) for required in deps]
    ready = {i for i in range(count) if not remaining[i]}
    reads = [
        [((request_ids[p], pool), n) for p, pool, n in accesses]
        for accesses in structure.reads
    ]
    writes = [
        [((request_ids[p], pool), n) for p, pool, n in accesses]
        for accesses in structure.writes
    ]
    resident = dict(resident)
    readers, writers, used = {}, set(), {}
    start, finish, binding = [None] * count, [None] * count, [None] * count
    start_instant, end_instant = [None] * count, [None] * count
    instants, representative = [time], [None]
    last = {"time": None, "completed": []}
    heap = []

    def dispatch(now, instant):
        for i in sorted(ready):
            if any(used.get(r, 0) >= capacities[r] for r in resources[i]):
                continue
            if any(k in writers or resident[k] < n for k, n in reads[i]):
                continue
            if any(k in writers or readers.get(k) for k, _ in writes[i]):
                continue
            end = now + durations[i]
            if not isfinite(end):
                raise ValueError("activity completion time overflow")
            for k, _ in reads[i]:
                readers[k] = readers.get(k, 0) + 1
            for k, _ in writes[i]:
                writers.add(k)
            for r in resources[i]:
                used[r] = used.get(r, 0) + 1
            start[i], start_instant[i] = now, instant
            if last["time"] == now:
                chosen = [d for d in deps[i] if finish[d] == now]
                if not chosen:
                    mine = set(resources[i])
                    chosen = [
                        j for j in last["completed"] if mine.intersection(resources[j])
                    ]
                binding[i] = max(chosen or last["completed"])
            ready.discard(i)
            heapq.heappush(heap, (end, i))

    dispatch(time, 0)
    instant = 0
    while heap:
        now = heap[0][0]
        if now != instants[instant]:
            instants.append(now)
            representative.append(heap[0][1])
            instant += 1
        while heap and heap[0][0] == now:
            end, i = heapq.heappop(heap)
            finish[i], end_instant[i] = end, instant
            for k, _ in reads[i]:
                readers[k] -= 1
                if not readers[k]:
                    del readers[k]
            for k, n in writes[i]:
                writers.remove(k)
                resident[k] = max(resident[k], n)
            for r in resources[i]:
                used[r] -= 1
            if last["time"] != now:
                last["time"], last["completed"] = now, []
            last["completed"].append(i)
            for j in dependents[i]:
                remaining[j] -= 1
                if not remaining[j]:
                    ready.add(j)
        dispatch(now, instant)
    if any(f is None for f in finish):
        return None
    path = []
    if count:
        path.append(max(range(count), key=lambda i: (finish[i], i)))
        while binding[path[-1]] is not None:
            path.append(binding[path[-1]])
        path.reverse()
    recipe = Recipe(
        tuple(start_instant),
        tuple(end_instant),
        tuple(representative),
        tuple(binding),
        tuple(path),
    )
    return Schedule(tuple(start), tuple(finish), instants[-1]), recipe


def replay(recipe, durations, time):
    """The schedule at another submission time, or None if the recorded
    control flow does not hold there."""
    starts, ends = recipe.start_instant, recipe.end_instant
    instants = [time]
    for k in range(1, len(recipe.representative)):
        r = recipe.representative[k]
        now = instants[starts[r]] + durations[r]
        if not now > instants[-1]:
            return None
        instants.append(now)
    if not isfinite(instants[-1]):
        return None
    finish = []
    for i, duration in enumerate(durations):
        end = instants[starts[i]] + duration
        if end != instants[ends[i]]:
            return None
        finish.append(end)
    return Schedule(tuple(instants[s] for s in starts), tuple(finish), instants[-1])
