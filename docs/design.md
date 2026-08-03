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

## Status (current)

Built beyond the original plan below:

- **Backends:** MLX (Apple Silicon) and vLLM (Linux/GPU). vLLM adds continuous batching and optional prefix caching (APC).
- **Streaming:** `SubmitGenerateStream` (server-streaming) in `direct`/`queued`; `micro_batch` returns `UNIMPLEMENTED`. MLX streams via `stream_generate`; vLLM via `AsyncLLMEngine` (`VLLM_ASYNC=1`), validated on a RunPod A40.
- **TTFT/TPOT:** worker/engine/client decomposition, reported only where valid — MLX on the streaming path only (unary and static batch return the whole response at once, so a first-token time isn't client-observable; judge those by throughput + latency). vLLM reports per-request TTFT/TPOT natively.
- **SLO auto-tuner** (`bench/autotune.py`): sweeps engine configs, recommends the max-throughput config meeting a latency SLO, and refuses to pass a config on a metric it cannot measure.

Key findings: on MLX, throughput (batching) and per-request streaming (TTFT) can't coexist without continuous batching. On vLLM the same engine policies invert — serializing in front of its continuous batcher costs 7.6x throughput and 120x client TTFT versus letting concurrency through. The correct control-plane policy is a property of the runtime beneath it, which is the argument for folding scheduling into the inference engine (vLLM/SGLang).

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
- No prefix-aware routing yet (vLLM's own APC is available; the engine does not route by prefix)

Those can come later after the serving engine behavior is understood.

## Related Notes

- [batch-tier.md](batch-tier.md) — design note on an offline batch tier (not implemented).

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

Later extensions (now implemented, except where noted):

- streaming token chunks — `GenerateStream` (worker) / `SubmitGenerateStream` (engine)
- batched request execution — `GenerateBatch`
- TTFT and decode timing split — `ttft_ms` / `tpot_ms` fields
- prefix-cache metadata — not yet (vLLM handles APC internally)

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

Current status:

- debug-only
- concurrent direct load has triggered native crashes in the Python MLX worker
- keep this mode for single-request debugging, not for throughput comparisons

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

Current status:

- this is now the primary execution path for performance work
- queued mode is the safe concurrency boundary for the current Python MLX worker

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

Current status:

- implemented as a first real batch path
- the engine waits for a short batching window, drains pending requests up to a maximum batch size, and dispatches them through the worker batch RPC
- worker-side `mlx_lm.batch_generate` has shown a meaningful throughput gain over serial execution in local measurement

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

## Request Flow: Two Layers of Scheduling

There are **two schedulers**, at two layers, and conflating them is the easiest
mistake to make here. The engine decides *how many requests get in*; the runtime
decides *how they interleave once inside*. The engine never implements batching —
it can only bound the batch the runtime is able to form.

### Streaming off

```
CLIENT
  │  8 concurrent requests
  ▼
ENGINE (our Rust)          ← ADMISSION control
  │  "how many do I let through?"
  │  queued  = Semaphore(1) → 1 at a time
  │  direct  = no limit     → all 8
  ▼
WORKER / vLLM              ← CONTINUOUS BATCHING
     "of what I've been given, how do I
      interleave them across decode steps?"
     maintains running set, one fused kernel per step
```

### Streaming on

Same two layers. What changes is the **return path**: tokens flow back
continuously instead of once at the end.

```
        DOWN: admission (once)                    UP: tokens (continuous)
        ──────────────────────                    ────────────────────────
CLIENT
  │  8 × SubmitGenerateStream                ▲  8 independent gRPC streams
  │                                          │  GenerateChunk{text_delta} … then
  ▼                                          │  GenerateChunk{finished, metrics}
                                             │  ⏱ CLIENT TTFT = first chunk on wire
ENGINE (Rust)         ADMISSION              │
  │  Semaphore(N)                            │  proxies chunk-by-chunk, no buffering
  │   N=1 → 1 in, 7 wait (whole stream!)     │  ⏱ ENGINE TTFT = first chunk forwarded
  │   N=∞ → all 8 in                         │  client hangs up → drop worker stream
  ▼                                          │
WORKER (Python, async)                       │  WorkerChunk per token
  │  N concurrent agenerate_stream()         │  ⏱ WORKER TTFT + TPOT timed here
  ▼                                          │
vLLM AsyncLLMEngine   CONTINUOUS BATCHING    │
     running set: [s1 s2 … sN]               │
     step k   → one fused kernel → N tokens ─┘   ← one token per sequence,
     step k+1 → one fused kernel → N tokens ─┘     each pushed to its own stream
     (sequences join/leave mid-flight)
```

**One decode step yields N tokens, fanning out to N different client streams.**
Batching and streaming are perpendicular: the batch runs *across* sequences, the
stream runs *along* one sequence. They do not trade off against each other.

This is only true of *continuous* batching. Static batching (`mlx_lm.batch_generate`,
our `micro_batch`) groups at the **request** level and returns only final texts,
so it genuinely cannot stream — which is why `micro_batch` returns `UNIMPLEMENTED`
for streaming and why TTFT/TPOT are unmeasurable there.

### Why the admission limit dominates

Under streaming, a request holds its admission slot for its **entire lifetime**,
not just a dispatch. So `N` directly bounds the runtime's batch size.

Measured on an A40 with `Qwen/Qwen3-32B-AWQ`, 8 concurrent streams, identical
worker, varying only `ENGINE_MODE`:

| | `queued` (N=1) | `direct` (N=∞) |
|---|---|---|
| Throughput | 32.9 tokens/s | 250.6 tokens/s |
| Worker TTFT p50 | 47 ms | 133 ms |
| Client TTFT p50 | 17,369 ms | 144 ms |
| TPOT p50 | 32.9 ms | 33.9 ms |

At N=1 the worker still answers in 47 ms; the other 17.3 s is requests waiting
their turn, and vLLM's batcher sits permanently at batch size 1. Aggregate
throughput equal to a single sequence's decode rate is the signature of that.

Two consequences worth keeping:

- **The gate is a workaround for a runtime that cannot handle concurrency.** It is
  mandatory for MLX (concurrent native calls segfault) and costly for vLLM.
  The right policy is a property of the runtime beneath, not a global default.
- **Per-request TTFT and system TTFT move in opposite directions.** Restricting
  concurrency makes any *individual* prefill faster (48 ms at N=1 vs 330 ms at
  N=32) while making the *population* far slower, because a queue forms behind
  it. This is why TTFT is measured at three vantage points; the worker/client gap
  is exactly the queueing.

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

Implemented since (streaming path): TTFT and TPOT, each with worker / engine / client vantage points.

Still future:

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

**Resolved since:** MLX's `batch_generate` is a usable batch primitive (micro-batch is real, ~1.6× throughput); transport is gRPC end-to-end; streaming was added after queueing was stable, as server-streaming on top of the unary contract rather than a milestone-1 feature.
