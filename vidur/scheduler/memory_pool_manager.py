"""Replica-local byte reservations and valid-prefix residency.

Admission reserves a request's declared maximum footprint atomically across
pools. Residency becomes visible only when a writing activity completes.
Step admission can atomically grow allocations. Explicit activity releases
reclaim source pools at completion; scheduler preemption discards whole requests.
"""


class MemoryPoolManager:
    def __init__(self, capacities):
        self.capacities = dict(capacities)
        if any(
            not isinstance(p, str) or not p or type(n) is not int or n < 1
            for p, n in self.capacities.items()
        ):
            raise ValueError("memory pool capacities must be positive integer bytes")
        self.used = dict.fromkeys(self.capacities, 0)
        self.peak_used = dict(self.used)
        self.reservations = {}
        self.resident = {}
        self.readers = {}
        self.writers = set()
        self._execution_started = False
        self._residency_initialized = False

    def initialize_residency(self, accesses):
        """Seed an explicit initial-state fixture before any memory execution.

        Validate the entire fixture before publishing any prefix. Reservations
        remain mandatory; this does not bypass capacity or access readiness.
        """
        if self._execution_started or self._residency_initialized:
            raise ValueError("initial residency requires pristine memory state")
        self.validate_accesses((), accesses)
        self._residency_initialized = True
        for access in accesses:
            self.resident[access.request_id, access.pool] = access.bytes

    def seed_residency(self, request_id, prefixes):
        """Publish a warm-start request's resident prefixes once, before it runs."""
        reserved = self.reservations.get(request_id, {})
        for pool, size in prefixes.items():
            if type(size) is not int or size < 0 or size > reserved.get(pool, -1):
                raise ValueError("warm-start residency exceeds the reservation")
        for pool, size in prefixes.items():
            key = (request_id, pool)
            if not self.resident.get(key):
                self.resident[key] = size

    def validate_requirements(self, requirements):
        if any(
            p not in self.capacities or type(n) is not int or n < 0
            for p, n in requirements.items()
        ):
            raise ValueError(
                "invalid memory reservation: pool and integer bytes required"
            )
        if any(n > self.capacities[p] for p, n in requirements.items()):
            raise ValueError("request footprint exceeds a memory pool capacity")

    def can_reserve(self, request_id, requirements):
        self.validate_requirements(requirements)
        if request_id in self.reservations:
            if self.reservations[request_id] != requirements:
                raise ValueError("request memory footprint changed while reserved")
            return True
        return all(
            self.used[p] + n <= self.capacities[p] for p, n in requirements.items()
        )

    def reserve(self, request_id, requirements):
        if not self.can_reserve(request_id, requirements):
            raise ValueError("insufficient memory pool capacity")
        if request_id in self.reservations:
            return
        self.reservations[request_id] = dict(requirements)
        for pool, size in requirements.items():
            self.used[pool] += size
            self.peak_used[pool] = max(self.peak_used[pool], self.used[pool])
            self.resident[request_id, pool] = 0

    def can_grow(self, request_id, requirements):
        """Check incremental demand without shrinking existing allocations."""
        self.validate_requirements(requirements)
        current = self.reservations.get(request_id, {})
        return all(
            self.used[p] + max(0, n - current.get(p, 0)) <= self.capacities[p]
            for p, n in requirements.items()
        )

    def grow(self, request_id, requirements):
        if not self.can_grow(request_id, requirements):
            raise ValueError("insufficient memory pool capacity")
        current = self.reservations.setdefault(request_id, {})
        for pool, size in requirements.items():
            previous = current.get(pool, 0)
            current[pool] = max(previous, size)
            self.used[pool] += max(0, size - previous)
            self.peak_used[pool] = max(self.peak_used[pool], self.used[pool])
            self.resident.setdefault((request_id, pool), 0)

    def free(self, request_id):
        pools = self.reservations[request_id]
        if any(
            (request_id, p) in self.writers or self.readers.get((request_id, p), 0)
            for p in pools
        ):
            raise ValueError("cannot free memory used by an active activity")
        for pool, size in pools.items():
            self.used[pool] -= size
            self.resident.pop((request_id, pool))
        del self.reservations[request_id]

    def validate_accesses(self, reads, writes, releases=()):
        for accesses in (reads, writes, releases):
            seen = set()
            for access in accesses:
                key = (access.request_id, access.pool)
                if key in seen:
                    raise ValueError("duplicate memory access in one activity")
                seen.add(key)
                reserved = self.reservations.get(access.request_id, {}).get(access.pool)
                if (
                    type(access.bytes) is not int
                    or access.bytes < 1
                    or reserved is None
                    or access.bytes > reserved
                ):
                    raise ValueError("memory access exceeds a request's reserved bytes")

        for access in releases:
            key = (access.request_id, access.pool)
            if access not in reads or any(
                (w.request_id, w.pool) == key for w in writes
            ):
                raise ValueError("release requires a matching read and no source write")
            if self.reservations[access.request_id][access.pool] != access.bytes:
                raise ValueError("release must cover the whole pool allocation")

    def ready(self, reads, writes, releases=()):
        self.validate_accesses(reads, writes, releases)
        for access in reads:
            key = (access.request_id, access.pool)
            if key in self.writers or self.resident[key] < access.bytes:
                return False
        for access in writes + releases:
            key = (access.request_id, access.pool)
            if key in self.writers or self.readers.get(key, 0):
                return False
        return True

    def begin(self, reads, writes, releases=()):
        if not self.ready(reads, writes, releases):
            raise ValueError("memory activity is not ready")
        self._execution_started = True
        for access in reads:
            key = (access.request_id, access.pool)
            self.readers[key] = self.readers.get(key, 0) + 1
        for access in writes + releases:
            self.writers.add((access.request_id, access.pool))

    def complete(self, reads, writes, releases=()):
        for access in reads:
            key = (access.request_id, access.pool)
            self.readers[key] -= 1
            if not self.readers[key]:
                del self.readers[key]
        for access in writes:
            key = (access.request_id, access.pool)
            self.writers.remove(key)
            self.resident[key] = max(self.resident[key], access.bytes)
        for access in releases:
            key = (access.request_id, access.pool)
            self.writers.remove(key)
            self.used[access.pool] -= self.reservations[access.request_id].pop(
                access.pool
            )
            del self.resident[key]
