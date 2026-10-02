"""Replica-local byte reservations and valid-prefix residency.

Admission reserves a request's declared maximum footprint atomically across
pools. Residency becomes visible only when a writing activity completes.
This conservative policy does not implement paging, dynamic growth or eviction.
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

    def validate_accesses(self, reads, writes):
        for accesses in (reads, writes):
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

    def ready(self, reads, writes):
        self.validate_accesses(reads, writes)
        for access in reads:
            key = (access.request_id, access.pool)
            if key in self.writers or self.resident[key] < access.bytes:
                return False
        for access in writes:
            key = (access.request_id, access.pool)
            if key in self.writers or self.readers.get(key, 0):
                return False
        return True

    def begin(self, reads, writes):
        if not self.ready(reads, writes):
            raise ValueError("memory activity is not ready")
        for access in reads:
            key = (access.request_id, access.pool)
            self.readers[key] = self.readers.get(key, 0) + 1
        for access in writes:
            self.writers.add((access.request_id, access.pool))

    def complete(self, reads, writes):
        for access in reads:
            key = (access.request_id, access.pool)
            self.readers[key] -= 1
            if not self.readers[key]:
                del self.readers[key]
        for access in writes:
            key = (access.request_id, access.pool)
            self.writers.remove(key)
            self.resident[key] = max(self.resident[key], access.bytes)
