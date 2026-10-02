# Native analytical resource execution

This fork adds an opt-in execution-plan provider alongside legacy operator-slot
predictors. Register a `BaseExecutionPlanPredictor` implementing
`resource_capacities()` and `get_execution_plan(batch, pipeline_stage)`.
Plans contain topologically ordered `ExecutionActivity` objects, durations in
seconds, resource names and dependencies. Each activity requires one unit of
every named resource. Capacities are replica-scoped and shared across stages;
stage-private resources require distinct names.

Activity completions and resource dispatch run on the normal event heap. A
replica-wide online executor starts ready activities in FIFO plan/activity order,
atomically reserves capacity, and releases it on completion. It does not reserve
resources for future work or infer interference costs. Costs remain fixed for an
activity once submitted. This is an analytical model, not hardware execution or
cycle simulation.

Native vLLM and Sarathi configurations can decouple `max_inflight_batches` and
`micro_batch_size` from pipeline depth. Defaults preserve original admission
limits. Legacy slot predictors reject these options. All activities in a stage
must complete before existing stage/batch events advance tokens and release KV.
There is no independent sub-batch token completion or separate device KV pool.

Chrome traces include activity intervals, resources, dependencies and batch/stage
IDs. Native elapsed stage time includes internal resource waiting. GPU-specific
MFU and legacy operator-slot attribution are omitted on this path. Resource
capacities model exclusion/parallel slots, not fluid bandwidth sharing. The heap
tie-breaker is corrected to event-type priority before event ID; completions run
before dispatch at equal timestamps.

Run core tests with `python -m unittest discover -s tests`. PIMScope carries
end-to-end serving, request-lifecycle, pipeline and calibrated-model regressions.
