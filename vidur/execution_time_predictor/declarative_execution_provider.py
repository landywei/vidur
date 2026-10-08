"""Native provider for declarative execution and immutable memory demand.

No paper knowledge, admission decisions, request mutation or slot timing here.
Memory policies receive immutable request state without future output lengths.
"""

from vidur.scheduler.execution_plan_builder import RequestState


class NativeExecutionProvider:
    uses_execution_plans = True

    def __init__(self, builder, memory_policy=None):
        self.execution_plan_builder = builder
        self.memory_policy = memory_policy

    def get_execution_time(self, batch, pipeline_stage):
        raise RuntimeError("native execution requires activity events, not slot timing")

    def resource_capacities(self):
        return self.execution_plan_builder.policy.resource_capacities()

    def memory_pool_capacities(self):
        return self.execution_plan_builder.policy.memory_pool_capacities()

    def request_memory_requirements(self, request):
        if self.memory_policy is None:
            if self.memory_pool_capacities():
                raise ValueError("declared memory pools require a memory policy")
            return {}
        return self.memory_policy.request_memory_requirements(
            RequestState.from_request(request)
        )

    def request_initial_residency(self, request):
        """Resident prefixes of a warm-start request, by pool."""
        policy = self.memory_policy
        if policy is None or not hasattr(policy, "request_initial_residency"):
            raise ValueError("warm-start requests need a memory policy with residency")
        return policy.request_initial_residency(RequestState.from_request(request))

    def request_step_memory_requirements(self, request, next_num_tokens):
        if self.memory_policy is None:
            return None
        return self.memory_policy.request_step_memory_requirements(
            RequestState.from_request(request), next_num_tokens
        )
