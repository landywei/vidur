# Native analytical resource execution

This fork adds an opt-in execution path alongside Vidur's operator-slot
predictors. A registered predictor returns a `NativeExecutionProvider` that
wraps an `ExecutionPlanBuilder(policy, costs)` and an optional memory policy:

- `policy.declare(work)` turns immutable selected work (`SelectedWork`: request
  snapshots and the pipeline stage) into a `PlanDeclaration` of activities,
  each with resources, dependencies and memory accesses, plus limitations and
  an optional plan identity.
- `costs.bind(work)` returns seconds with provenance and granularity per cost
  key, and raises outside its calibration domain.
- `policy.resource_capacities()` and `memory_pool_capacities()` name the
  replica's resources and byte pools.

Activity completions and resource dispatch run on the normal event heap. A
stage runs one batch's plan at a time, as the slot path does; batches overlap
across stages. A replica-wide executor starts dependency-ready activities in
plan order, atomically reserves one unit of every named resource, and releases
it on completion. It does not reserve resources for future work or infer
interference costs; costs are fixed once a plan is submitted. This is an
analytical model, not hardware execution or cycle simulation.

Memory pools admit requests by declared byte demand (a fixed reservation or
step growth) and track resident prefixes: an activity starts only when its
reads are resident, and its writes become resident on completion. A request
that arrives with tokens already processed (a warm start) is seeded with its
resident prefix.

Native vLLM and Sarathi configurations can decouple `max_inflight_batches` and
`micro_batch_size` from pipeline depth. All activities in a stage complete
before stage and batch events advance tokens and release KV.

Chrome traces include activity intervals, resources, dependencies, declared
duration, plan identity, and stage envelopes with per-request prefill status.
GPU MFU and operator-slot attribution are omitted on this path. The heap orders
events by type before ID, so completions run before dispatch at equal times. A
simulation that stops with unfinished requests raises instead of asserting.

Run core tests with `python -m unittest discover -s tests`. PIMScope carries
end-to-end serving, request-lifecycle, pipeline and calibrated-model regressions.
