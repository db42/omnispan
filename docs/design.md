# Omnispan Perf Lab Design

This document is both the original design plan and a record of what building it
actually taught us. Sections carry a **Status** line where the outcome differs
from the plan. Where a measurement contradicted the original reasoning, the
original reasoning is kept and marked rather than quietly deleted — the delta is
the most useful part of the document.

## Goal

Build a tiny "Token Factory Perf Lab" that demonstrates inference-serving
performance behavior clearly.

The first version was intentionally narrow:

- One model
- One machine
- One Rust engine process
- One Python worker process
- A small number of synthetic tenants
- A small number of serving modes that can be benchmarked cleanly

This is not a product build. It is a performance lab.

## Status (current)

Built beyond the original plan:

- **Backends:** MLX (Apple Silicon) and vLLM (Linux/GPU). vLLM brings continuous
  batching and optional prefix caching (APC).
- **Streaming:** `SubmitGenerateStream` (server-streaming) in `direct`/`queued`;
  `micro_batch` returns `UNIMPLEMENTED`. MLX streams via `stream_generate`; vLLM
  via `AsyncLLMEngine` (`VLLM_ASYNC=1`), validated on a RunPod A40.
- **Admission limit:** `MAX_CONCURRENT_STREAMS` (0 = unlimited) generalizes the
  streaming gate. `queued` and `direct` are its endpoints (N=1 and N=∞).
- **TTFT/TPOT:** worker/engine/client decomposition, reported only where valid —
  MLX on the streaming path only (unary and static batch return the whole
  response at once, so a first-token time is not client-observable; judge those
  by throughput and total latency). vLLM reports per-request TTFT/TPOT natively.
- **SLO auto-tuner** (`bench/autotune.py`) and **admission sweep**
  (`bench/sweep_concurrency.py`). Both refuse to pass a config on a metric they
  could not measure.

### The three findings that changed the design

1. **On MLX, throughput and per-request streaming cannot coexist.** Static
   batching (`batch_generate`) has no per-sequence token events, so batching and
   TTFT are mutually exclusive without continuous batching.
2. **On vLLM the same engine policies invert.** Serializing in front of its
   continuous batcher costs 7.6× throughput and 120× client TTFT versus letting
   concurrency through. The correct control-plane policy is a property of the
   runtime beneath it, not a global default.
3. **Admission saturates, then stops mattering.** Sweeping N on an A40:
   throughput plateaus at N≈128 (~1,260 tokens/s) because vLLM's KV cache and
   `max_num_seqs` bind before the engine's gate does. Past that the control plane
   has no influence. The live tradeoff is TPOT (33 ms at N=1 → 104 ms at N=128),
   so the throughput-optimal and SLO-optimal N are different numbers.

## What We Wanted To Learn — and what we found

| Question | Answer |
|---|---|
| How end-to-end latency decomposes into queue wait and model execution | Cleanly, once measured at three vantage points. Under load, queueing dominates: at N=1, 99.6% of client TTFT was queue wait. |
| How throughput changes with scheduling | Enormously, and in a backend-dependent direction: batching helps MLX ~1.6×; *not* gating helps vLLM 7.6×. |
| Whether an engine-controlled path beats direct execution under load | For MLX yes (direct segfaults). For vLLM no — direct wins decisively. |
| How much batching helps on MLX | ~1.6× (117 vs 72 tokens/s), at the cost of losing TTFT/TPOT observability entirely. |
| What metrics matter for a control plane | TTFT and TPOT per vantage point, queue/gate wait, and — the one we lack — KV-cache pressure, which is the real admission signal. |

## Non-Goals For The First Iteration

- No dashboard
- No billing system
- No full OpenAI API compatibility
- No multi-node routing
- No production auth system
- No full tenant management UI
- No speculative decoding
- No prefix-aware routing (vLLM's own APC is available; the engine does not route
  by prefix)
- No offline batch tier — see [batch-tier.md](batch-tier.md)

## Related Notes

- [batch-tier.md](batch-tier.md) — design note on an offline batch tier (not implemented).

## High-Level Architecture

Two processes: a Rust control plane and a Python model worker.

### Layer 1: Edge + Serving Engine (Rust)

Responsibilities:

- Accept external requests, validate shape, assign request IDs
- **Admission control** — decide how many requests reach the worker, and later
  which ones
- Own the pending queue and the scheduler loop (unary path)
- Own the streaming admission gate (streaming path)
- Record metrics and route responses back to callers

> **Status: revised.** The original plan said this layer "should contain almost
> all performance-critical orchestration logic." That is true only when the
> worker has no scheduler of its own. With vLLM, the performance-critical
> scheduling — which sequences advance each decode step, KV allocation,
> preemption — lives in the runtime, and the engine's remaining job is admission.
> Measured consequence: applying engine-side serialization to vLLM cost 7.6×
> throughput. See [Request Flow](#request-flow-two-layers-of-scheduling).

### Layer 2: Model Worker (Python)

Responsibilities:

- Load the model once; own tokenizer and runtime state
- Execute inference requests (unary, batch, streaming)
- Expose timing data for worker-side execution

> **Status: revised.** The original rule was "this layer should not own
> scheduling policy." That holds for MLX, which has no scheduler. It is false for
> vLLM, which owns continuous batching — genuinely a scheduling policy, and the
> one that determines throughput. The accurate rule is: *the worker owns
> intra-worker scheduling; the engine owns admission.*

### Optional Future Split

The Rust process could later split into an API gateway and an engine. Not needed
at this scale.

## Process Model

- `omnispan-engine` (Rust)
- `worker/worker.py` (Python)

Single machine; the engine talks to one worker over gRPC.

## Internal RPC Boundary

gRPC between Rust and Python — strongly typed, streaming-capable, and a clean
path to multiple workers.

### Worker RPC

**Status: implemented, extended beyond the original unary plan.**

| RPC | Shape | Notes |
|---|---|---|
| `Generate` | unary | original v1 contract |
| `GenerateBatch` | unary, N requests | static batch (MLX `batch_generate`) |
| `GenerateStream` | server-streaming | one `WorkerChunk` per token, then a terminal summary chunk |

Metric fields added since: `ttft_ms`, `tpot_ms` on replies and terminal chunks;
`engine_ttft_ms` and `gate_wait_ms` on the engine's `GenerateChunk`.

Still not carried on the wire: prefix-cache metadata (vLLM handles APC
internally and we do not read its per-request cache stats).

## Serving Modes

**Status: all three implemented.** Selected with `ENGINE_MODE`. Streaming adds an
orthogonal dial, `MAX_CONCURRENT_STREAMS`.

### 1. `direct`

The engine calls the worker immediately; no queue.

- **MLX:** debug-only. Concurrent direct load triggers native crashes in the MLX
  runtime.
- **vLLM:** the *best* mode. Concurrency passes straight through to the
  continuous batcher — 250.6 tokens/s vs 32.9 for `queued` at concurrency 8.

### 2. `queued`

A background scheduler loop pulls one request at a time; exactly one request is
in the worker at a time. Streaming uses a 1-permit gate instead of the loop.

- **MLX:** the primary execution path — the safe concurrency boundary for a
  runtime that segfaults under parallel native calls.
- **vLLM:** actively harmful. It pins the running batch at 1 and idles the
  batcher.

> **Status: original claim corrected.** The plan called queued "the primary
> execution path for performance work" without qualification. It is
> backend-dependent: mandatory for MLX, a 7.6× penalty for vLLM.

### 3. `micro_batch`

The engine waits a short window, drains up to `MAX_BATCH_SIZE` pending requests,
and dispatches them through the worker batch RPC.

- **MLX:** real and effective — ~117 tokens/s peak (w20/b4), ~1.6× over queued.
- **Streaming:** returns `UNIMPLEMENTED`; a static batch cannot stream per request.
- **Observability cost:** TTFT and TPOT are unmeasurable on this path.

## What `micro_batch` Actually Taught Us

> **Status: premise superseded.** This section originally argued micro-batch was
> a stepping stone toward building continuous batching in the engine. It is not.
> The original reasoning is preserved below because the conclusion it led to was
> wrong in an instructive way.

**Original reasoning.** True continuous batching is token-step scheduling across
in-flight requests, which needs prefill/decode phase awareness, per-request state
across decode steps, and detailed worker runtime control. `micro_batch` was the
correct first approximation because it teaches the engine shape, preserves a
clear latency/throughput tradeoff, and can be built on simpler worker primitives.

**What we learned.** Micro-batch is a *workaround for a runtime without a
scheduler*, not a step toward one. Continuous batching cannot be implemented in
the control plane at all: it requires advancing every active sequence one decode
step inside a shared kernel, plus KV-block allocation, eviction, and mid-flight
admission. **It must live where the KV cache lives.** A control plane above it can
only decide who gets in and when.

This is why the industry consolidated on vLLM/SGLang rather than on smart
gateways, and it is the strongest single finding in this project.

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

> **Status: aspirational.** The explicit state machine below is *not*
> implemented. In practice the engine records `received_at` and `scheduled_at`
> and derives queue/gate wait from them; the worker times its own execution. The
> full model is retained as the target shape if per-request tracing is added.

States: `received`, `queued`, `scheduled`, `dispatched`, `running`, `completed`,
`failed`.

Timestamps: `received_at`, `queued_at`, `scheduled_at`, `worker_started_at`,
`worker_completed_at`, `responded_at`.

Derived: queue wait, engine overhead, worker execution time, end-to-end latency.

## Metrics To Capture

**Implemented** — per request: request ID, tenant ID, serving mode, status, input
and output tokens, queue/gate wait ms, worker latency ms, end-to-end latency ms,
and on the streaming path `ttft_ms` / `tpot_ms` at worker, engine, and client
vantage points.

**Implemented** — aggregate: total/success/failure counts, requests per second,
tokens per second, p50/p95/p99 and mean for every latency series.

**Not implemented:**

- batch size distribution
- engine queue depth over time (only an instantaneous `queue_depth` in
  `GetEngineStats`)
- prefix cache hit rate, prefill tokens saved
- **KV-cache pressure** (`gpu_cache_usage_perc` from vLLM) — the sweep showed this
  is the signal a real admission controller should close the loop on, rather than
  a static N

## Tenants In The First Perf Lab

Tenants exist only to model contention and future isolation policies. They are
labels and metrics dimensions; no quota enforcement is implemented.

The benchmark rotates synthetic tenants (`tenant-1`, `tenant-2`, … via
`--tenant-prefix` / `--tenant-count`). The originally planned `shared-basic` /
`reserved-pro` classes were never built, and would only be worth adding alongside
real fairshare scheduling.

## Benchmark Plan

**Status: implemented and exceeded.**

| Planned | Actual |
|---|---|
| modes `direct`, `queued`, `micro_batch` | all three, plus streaming and an admission-limit sweep |
| concurrency 1, 5, 10, 20 | 1–10 locally (MLX); 8–256 on GPU |
| prompt shape short and medium | short/medium, plus a prefix-cache-heavy shape (`--shared-prefix-repeats`) |
| JSON artifacts + one markdown note | JSON artifacts in `bench/` and `bench/runpod/`; results tables in the README |

Harnesses: `bench/benchmark.py` (load generator, `--stream`),
`bench/autotune.py` (SLO-driven config search), `bench/sweep_concurrency.py`
(admission-limit sweep).

## Module Layout

**Status: as built** (the planned `types.rs`, `metrics.rs`, and
`worker_types.py` were never needed).

Rust:

- `engine/src/bin/omnispan-engine.rs` — config wiring and startup
- `engine/src/config.rs`, `engine/src/lib.rs`
- `engine/src/engine.rs` — gRPC service, streaming proxy, admission gate
- `engine/src/queue.rs` — scheduler loop and micro-batch formation
- `engine/src/worker_client.rs` — worker RPC, unary/batch/streaming

Python:

- `worker/worker.py` — gRPC service
- `worker/worker_runtime.py` — backend selection and the runtime interface
- `worker/mlx_runtime.py`, `worker/vllm_runtime.py`, `worker/vllm_async_runtime.py`

Engine policy stays independent of transport; the API handler does not know how
the worker is called.

## Implementation Order — as completed

| Milestone | Outcome |
|---|---|
| 1. Direct mode through Rust | Done. Exposed the MLX concurrency crash, which forced the queued path. |
| 2. Queued mode | Done. Separated queue wait from worker time — the decomposition everything else rests on. |
| 3. Micro-batch mode | Done. ~1.6× on MLX; also revealed that static batching forfeits TTFT/TPOT. |
| 4. Benchmark artifacts | Done. `bench/` and `bench/runpod/`, with results tables in the README. |
| 5. *(added)* TTFT/TPOT + streaming | Done. Three-vantage decomposition; `SubmitGenerateStream`. |
| 6. *(added)* vLLM backend + async streaming | Done. Validated on a RunPod A40. |
| 7. *(added)* Admission limit + sweep | Done. Found saturation at N≈128 and the TPOT tradeoff. |

Next candidates, in order of value: adaptive admission driven by KV-cache
pressure; multi-tenant fairshare over GPU-time; prefix-affinity routing (needs
multiple workers).

## Design Rules

- The worker owns model state and **intra-worker scheduling**; the engine owns
  **admission**.
- Every request must be traceable by request ID.
- Every serving mode must produce the same benchmark schema.
- Do not hide queue time inside worker time.
- **Do not report a metric the execution path cannot actually measure.** Report
  it only where it is observable, and let an unmeasured metric fail an SLO check
  rather than silently pass it.
- Do not prematurely add product features that do not change performance learning.

> The original second rule read "the engine owns queueing and execution policy."
> Execution policy belongs to the runtime whenever the runtime has one.

## Open Questions — resolved

| Question | Answer |
|---|---|
| Does the MLX worker support useful multi-request batch execution? | Yes. `mlx_lm.batch_generate` is real and gives ~1.6×, but returns only final texts, so it forfeits TTFT/TPOT. |
| Should `micro_batch` be grouped scheduling or a real batch API? | A real batch RPC, and it was the right call — it made the observability cost of static batching measurable. |
| HTTP edge first, or gRPC end-to-end? | gRPC end-to-end. Server-streaming later came almost free as a result. |
| Streaming in milestone 1, or unary until queueing is stable? | Unary first was correct. Streaming layered cleanly on top of a stable queue, and the unary path still serves benchmarks and `grpcurl`. |

The original guidance — assume unary first, assume no batching primitive until
proven, keep transport gRPC, keep milestone 1 synchronous — held up. It kept the
design honest and avoided fake sophistication.

## Open Questions — current

- What is the right control signal for admission? The sweep says a static N is
  the wrong shape; KV-cache pressure plus a TTFT/TPOT SLO is likely the right
  closed loop.
- How should the engine learn a backend's capability, rather than being told?
  `direct` for vLLM and `queued` for MLX are opposite defaults, currently chosen
  by hand.
- What breaks first at multi-worker scale — routing, or fairness?
