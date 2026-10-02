"""Activity-cost provider interface, independent of the legacy operator slots."""

from abc import ABC, abstractmethod


class BaseExecutionPlanPredictor(ABC):
    uses_execution_plans = True

    def __init__(
        self, predictor_config, replica_config, replica_scheduler_config, metrics_config
    ):
        self._config = predictor_config
        self._replica_config = replica_config

    @abstractmethod
    def resource_capacities(self):
        """Return replica-wide resource names and integer capacities."""
        raise NotImplementedError

    @abstractmethod
    def get_execution_plan(self, batch, pipeline_stage):
        """Return a costed plan; no scheduling or request mutation here."""
        raise NotImplementedError

    def memory_pool_capacities(self):
        """Return usable replica-wide byte capacities; empty disables pool admission."""
        return {}

    def request_memory_requirements(self, request):
        """Declare a stable maximum byte footprint per pool; do not allocate here."""
        return {}

    def get_execution_time(self, batch, pipeline_stage):
        raise RuntimeError("execution-plan predictors require native activity events")
