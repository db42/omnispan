# Omnispan Perf Lab Design

## Goal

Build a tiny "Token Factory Perf Lab" that demonstrates inference-serving performance behavior clearly.

The first version is intentionally narrow:

- One model
- One machine
- One Rust engine process
- One Python MLX worker process
- A small number of synthetic tenants
- A small number of serving modes that can be benchmarked cleanly

This is not a product build. It is a performance lab.

## What We Want To Learn

- How end-to-end latency decomposes into queue wait time and model execution time
- How throughput changes when requests are scheduled differently
- Whether an engine-controlled path beats direct request execution under load
- How much batching helps on the current MLX runtime
- What metrics matter for a future Token Factory-style control plane

## Non-Goals For The First Iteration

- No dashboard yet
- No billing system
- No full OpenAI API compatibility
- No multi-node routing
- No production auth system
- No full tenant management UI
- No speculative decoding yet
- No prefix caching yet

Those can come later after the serving engine behavior is understood.

## High-Level Architecture

There are three conceptual layers, but only two processes in the first version.

### Layer 1: Edge + Serving Engine

Language: Rust

Responsibilities:

- Accept external requests
- Validate request shape
- Assign request IDs
- Track request lifecycle
- Own the pending queue
- Run the scheduling loop
- Choose execution mode
- Record metrics
- Forward work to the Python worker
- Route responses back to callers

This layer should contain almost all performance-critical orchestration logic.

### Layer 2: Model Worker

Language: Python

Responsibilities:

- Load the MLX model once
- Own tokenizer and runtime state
- Execute inference requests
- Return response text and token counts
- Expose timing data for worker-side execution

This layer should not own scheduling policy.

### Optional Future Split

Later, the Rust process can be split into two conceptual services:

- API gateway
- engine

That split is not needed for the first perf lab.

## Process Model

Initial deployment:

- `omnispan-engine` in Rust
- `worker/worker.py` in Python

Single machine only.

The engine communicates with one worker over internal RPC.

## Internal RPC Boundary

The internal RPC should be small and explicit.

Use gRPC between Rust and Python.

Reason:

- Strongly typed contract
- Easy streaming extension later
- Familiar from `ftrie`
- Clean future path to multiple workers

### Initial Worker RPC

Unary is enough for version 1.

`Generate`

Request fields:

- `request_id`
- `tenant_id`
- `prompt`
- `max_tokens`
- `submitted_at_ms`

Response fields:

- `request_id`
- `response_text`
- `input_tokens`
- `output_tokens`
- `worker_latency_ms`
- `status`
- `error_message`

Later extensions:

- streaming token chunks
- batched request execution
- prefix-cache metadata
- TTFT and decode timing split

## Serving Modes

The engine will support three modes first.

### 1. `direct`

Behavior:

- request enters the engine
- the engine immediately calls the worker for that single request
- no queue ownership beyond the active request

Purpose:

- baseline for comparison
- measure current steady-state path with Rust edge in front

Expected behavior:

- simplest control path
- poor behavior under concurrency when many requests compete for one worker

### 2. `queued`

Behavior:

- request enters a shared queue
- a background scheduler loop inside the engine pulls one request at a time
- exactly one request is sent to the worker at a time

Purpose:

- establish explicit queue ownership
- separate queue wait time from worker execution time
- create the correct shape for later batching

Expected behavior:

- same or similar worker execution latency as `direct`
- clearer queueing metrics
- improved architectural clarity, not necessarily better raw latency

### 3. `micro_batch`

Behavior:

- request enters a shared queue
- engine waits for a short batching window
- engine collects up to `batch_size` pending requests
- engine dispatches them as a batch if worker supports it
- if worker does not support true batching yet, the engine executes the collected set as a grouped scheduling unit and records the batch attempt

Purpose:

- measure batching behavior cleanly
- create the first real performance experiment
- prepare for true continuous batching later

Expected behavior:

- better throughput under concurrency if batching is real and effective
- slightly worse per-request latency at low load
- queue wait time becomes an intentional tradeoff

## Why `micro_batch` Before True Continuous Batching

True continuous batching is token-step scheduling across in-flight requests.

That is harder because it requires:

- prefill/decode phase awareness
- request state tracking across decode steps
- more detailed worker runtime control
- often a more specialized inference backend

`micro_batch` is the correct first approximation because it:

- teaches the engine shape
- preserves a clear latency/throughput tradeoff
- can be implemented with simpler worker primitives

## Request Lifecycle

Every request should move through explicit states.

States:

- `received`
- `queued`
- `scheduled`
- `dispatched`
- `running`
- `completed`
- `failed`

Timestamps to record:

- `received_at`
- `queued_at`
- `scheduled_at`
- `worker_started_at`
- `worker_completed_at`
- `responded_at`

Derived metrics:

- queue wait time
- engine overhead
- worker execution time
- end-to-end latency

This state model matters more than feature breadth.

## Metrics To Capture

The lab should capture the following in all modes.

Per-request:

- request ID
- tenant ID
- serving mode
- status
- input tokens
- output tokens
- queue wait ms
- worker latency ms
- end-to-end latency ms

Aggregate:

- total requests
- success count
- failure count
- requests per second
- tokens per second
- p50 latency
- p95 latency
- p99 latency
- average queue wait
- average worker latency
- batch size distribution
- engine queue depth over time

Future metrics:

- TTFT
- decode tokens/sec
- prefix cache hit rate
- prefill tokens saved

## Tenants In The First Perf Lab

Tenants exist only to model contention and future isolation policies.

Keep this minimal:

- `shared-basic`
- `reserved-pro`

For the first pass, tenants are labels and metrics dimensions.

Do not build full quota enforcement yet unless it directly helps a benchmark.

## Benchmark Plan

The first benchmark matrix should be small and repeatable.

Dimensions:

- mode: `direct`, `queued`, `micro_batch`
- concurrency: `1`, `5`, `10`, `20`
- prompt shape: short and medium

Outputs:

- JSON result artifacts
- one markdown comparison note

The benchmark harness should produce comparable output across all modes.

## Suggested Module Layout

Rust side:

- `engine/src/bin/omnispan-engine.rs`
- `engine/src/config.rs`
- `engine/src/lib.rs`
- `engine/src/types.rs`
- `engine/src/engine.rs`
- `engine/src/queue.rs`
- `engine/src/metrics.rs`
- `engine/src/worker_client.rs`

Python side:

- `worker/worker.py`
- `worker/worker_types.py`
- `worker/worker_runtime.py`

Keep engine policy independent from the transport layer.

The API handler should not know how the worker is called.

## Implementation Order

### Milestone 1: Direct Mode Through Rust

- Rust engine accepts requests
- Rust engine forwards unary RPC to Python worker
- End-to-end request works through the new boundary
- Metrics are emitted

### Milestone 2: Queued Mode

- Add explicit queue
- Add background scheduler loop
- Serialize requests through the queue
- Measure queue wait separately from worker time

### Milestone 3: Micro-Batch Mode

- Add batching window and max batch size
- Group pending requests
- Dispatch grouped work
- Compare throughput and latency against direct and queued

### Milestone 4: Benchmark Artifacts

- Save results for all three modes
- Write comparison notes
- Identify the next worthwhile optimization

## Design Rules

- The worker owns model state, not the engine.
- The engine owns queueing and execution policy.
- Every request must be traceable by request ID.
- Every serving mode must produce the same benchmark schema.
- Do not hide queue time inside worker time.
- Do not prematurely add product features that do not change performance learning.

## Open Questions

These need answering before implementation goes deep:

- Does the Python MLX worker support true multi-request batch execution in a form useful here?
- If not, should `micro_batch` start as grouped scheduling with sequential execution, or should the worker API be designed for future batched execution immediately?
- Do we want the Rust edge to expose HTTP first, or start with gRPC end-to-end for faster internal iteration?
- Do we want streaming responses in milestone 1, or keep everything unary until queueing is stable?

## Recommended Answer To The Open Questions

For now:

- assume unary worker calls first
- assume no true batching primitive until proven
- expose a simple HTTP edge later, not first
- keep engine-to-worker transport gRPC
- keep milestone 1 synchronous and unary

That keeps the design honest and minimizes fake sophistication.
