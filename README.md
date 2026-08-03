# Omnispan

A small LLM-serving perf lab I built to study, hands-on, how the serving layer
shapes latency and throughput. A Rust **admission-control engine** sits in front
of a Python model worker (vLLM or MLX) over gRPC. The point of the lab is to
isolate one question that's easy to conflate: **engine-level admission control
vs. runtime-level continuous batching** — two schedulers at two layers — and
measure how the boundary between them drives TTFT, TPOT, and throughput.

Single node, one worker: the goal is a legible control plane you can read end to
end, not a distributed system.

## Architecture

```
CLIENT ──8 concurrent streams──▶ ENGINE (Rust)   ──gRPC──▶ WORKER (Python) ──▶ vLLM / MLX
                                 admission control         per-token stream      continuous
                                 "how many get in?"        TTFT / TPOT timed     batching
```

The engine decides *how many requests get in*; the runtime decides *how they
interleave once inside*. The engine never implements batching — it only bounds
the batch the runtime can form. TTFT is timed at three vantage points (worker,
engine, client) so queueing is visible separately from compute.

## A measured result

One A40, `Qwen/Qwen3-32B-AWQ`, 8 concurrent streams of a 13-token prompt with
150-token outputs. Identical worker, identical load — varying **only** the
engine's concurrency gate:

| | `queued` (N=1) | `direct` (N=∞) |
|---|---|---|
| Throughput | 32.9 tokens/s | **250.6 tokens/s** |
| Client TTFT p50 | 17,369 ms | **144 ms** |
| Worker TTFT p50 | 47 ms | 133 ms |
| TPOT p50 | 32.9 ms | 33.9 ms |

The instructive part: **per-request and population TTFT move in opposite
directions.** Restricting concurrency makes any *individual* prefill faster — the
worker still answers in 47 ms, since that one sequence has the GPU to itself —
while making the *population* far slower, because that 47 ms now sits behind a
~17 s queue. The gate pins vLLM's batcher at batch size 1, so aggregate
throughput collapses to a single sequence's decode rate. In `direct`, prefill
competes with seven other sequences' decode steps and slows to 133 ms; every
client still sees its first token ~120× sooner.

Meanwhile TPOT barely moves (32.9 → 33.9 ms) while throughput grows 7.6×. That
is the signature of continuous batching: eight sequences share each decode step,
so per-token latency is nearly unchanged while aggregate output multiplies.

The takeaway I keep: the right admission policy is a property of the runtime
beneath, not a global default. (MLX *needs* the gate — concurrent native calls
segfault; vLLM is punished by it.)

*Small sample — 8 requests — but an independent 32-request sweep reproduced both
cells to within 0.1% (32.86 and 250.63 tokens/s), and this A/B is the cleanest
controlled comparison: same offered load, one variable.*

## Where the control plane stops mattering

Sweeping the admission limit shows the gate has a ceiling — and that there are
**two inflection points, at different N**:

| N | 1 | 8 | 32 | 64 | 128 | 256 | ∞ |
|---|---|---|---|---|---|---|---|
| tokens/s | 32.9 | 250.6 | 860.2 | 1,170.4 | **1,258.6** | 1,260.2 | 1,266.4 |
| TPOT p50 (ms) | 33.0 | 34.0 | **38.7** | 55.3 | 103.7 | 103.8 | 102.0 |

- **Throughput saturation at N≈128** (~1,260 tokens/s). Beyond it, N=256 and
  unlimited are identical — vLLM's KV cache and `max_num_seqs` bind before the
  engine's gate does, so the control plane has no further influence.
- **Latency knee at N≈32–64.** TPOT is flat to 32 (33 → 39 ms), then degrades
  sharply: 55 ms at 64, 104 ms at 128.

Those are separate curves bending at separate points, which is the whole reason
throughput-optimal and SLO-optimal disagree: **throughput keeps improving well
after per-token latency has started falling apart.** Maximizing throughput picks
N=128 (1,259 tokens/s, TPOT 104 ms). A 50 ms TPOT target — roughly reading speed
— picks **N=32**: 68% of peak throughput at usable per-user latency. And at this
load *no* N met a 2 s TTFT p95 target, which is the signal to add replicas rather
than keep tuning.

**The workload these numbers describe.** 13-token prompts, 150-token outputs
(11.5:1 output-to-input), uniform length, Qwen3-32B-AWQ on one A40. That is a
**decode-bound workload with negligible prefill**, which is exactly why batching
scaled so cheaply — adding sequences to a memory-bandwidth-bound decode step is
nearly free until bandwidth or KV memory runs out. Long prompts would shift
saturation much earlier (prefill is compute-bound and stalls decode for
everyone); longer or mixed-length outputs would too. **N≈128 is a property of
this workload, not a transferable constant** — which is itself the argument that
a static N is the wrong control variable, and the loop should close on measured
KV pressure instead.

*Table splices two sweeps at different offered loads (N≤16 at 32 requests, N≥32 at 512). N=32 ran in both at 859.9 vs 860.2 tokens/s — the join's justification and a reproducibility check.*

I expected a single clean inflection and initially found none: the first sweep
was capped by the benchmark client's own concurrency, not the GPU. Re-running at
higher offered load produced the curve above.

## What's here

- `engine/` — Rust control plane: intake, admission gate, scheduler loop, worker RPC
- `worker/` — Python worker with pluggable backends (`mlx_runtime`, `vllm_runtime`, `vllm_async_runtime`)
- `proto/` — shared gRPC contract (unary, batch, and server-streaming)
- `bench/` — load generator, SLO auto-tuner, admission sweep, result artifacts
- `docs/` — [`design.md`](docs/design.md) (request-flow diagrams, what the measurements changed), [`batch-tier.md`](docs/batch-tier.md)
- `scripts/` — [`RUNPOD.md`](scripts/RUNPOD.md) and GPU-pod provisioning

The worker gRPC contract is identical across backends; the backend switch lives
entirely inside `worker/`.

## Scheduling policies

The engine implements three request-admission policies, selected with
`ENGINE_MODE`. Each one exists to isolate a specific tradeoff.

| Mode | Behavior | What it demonstrates |
|---|---|---|
| `direct` | Execute inline on the request thread, no queue | Baseline — and why unmanaged concurrency is unsafe (the MLX runtime segfaults under concurrent native calls) |
| `queued` | Single in-engine queue, one request executed at a time | Safe serialization; isolates queue-wait cost under concurrency |
| `micro_batch` | Accumulate requests for a short window, dispatch as one batch | Throughput gains from batching, and the latency/observability cost they carry |

## Inference metrics

Omnispan measures request execution as distinct phases rather than one latency
number, because that decomposition is what drives serving decisions:

- **TTFT (time to first token)** — prefill cost the user feels before anything
  appears. Measured inside the worker by streaming generation token-by-token.
- **TPOT (time per output token)** — mean inter-token latency during decode.
- **Queue wait** — time a request sits in the engine queue before dispatch.
- **Worker latency** — total time inside the worker (prefill + decode).
- **Throughput** — aggregate tokens/second across all successful requests.

In **streaming mode** (see [Streaming](#streaming)) TTFT is measured at three
vantage points on the same request, which localizes where latency actually comes
from:

- **worker TTFT** — model prefill time, measured at the runtime.
- **engine TTFT** — dispatch to first token forwarded onward (adds engine overhead).
- **client TTFT** — first token on the wire at the client; this is what the user
  feels, and under concurrency it also absorbs queue/serialization wait.

A large client-vs-engine gap points at queueing, not prefill — the kind of
distinction that changes whether you scale workers or tune the scheduler.

TPOT is captured both worker-side and client-side (`client_tpot_ms`). Unlike
TTFT, the two tend to match closely even under load: queue wait is paid *before*
the first token, so it inflates TTFT but not the steady-state decode rate. If
TPOT is healthy while TTFT is not, the bottleneck is admission/queueing, not the
model.

> **Measurement honesty:** TTFT and TPOT are serving metrics tied to a token
> stream the client actually receives. For MLX they are reported **only on the
> streaming path** — the unary and synchronous-batch paths return the whole
> response at once (batch via `mlx_lm.batch_generate` returns only final texts),
> so a first-token time there would describe an internal decode property, not a
> latency anyone observed. Those paths are judged by throughput and total
> latency. vLLM is different: its continuous batching streams each sequence
> independently, so per-request TTFT/TPOT stay valid even on its unary/batch
> paths, read from vLLM's native metrics.

## Prerequisites

- Rust toolchain
- `python`
- Python environment with backend-specific worker dependencies installed
- `grpcurl` for manual testing

Install local MLX worker dependencies:

```bash
python -m pip install -r worker/requirements.txt
```

Install vLLM worker dependencies on your Linux/CUDA box instead:

```bash
python -m pip install -r worker/requirements-vllm.txt
```

## Worker backends

- `WORKER_BACKEND=mlx`
  - default backend
  - intended for Apple Silicon local development
  - default model: `mlx-community/Qwen2.5-7B-Instruct-4bit`

- `WORKER_BACKEND=vllm`
  - intended for Linux + NVIDIA GPU environments such as RunPod
  - default model: `Qwen/Qwen2.5-7B-Instruct`
  - useful env vars: `MODEL_ID`, `VLLM_TENSOR_PARALLEL_SIZE`,
    `VLLM_GPU_MEMORY_UTILIZATION`, `VLLM_MAX_MODEL_LEN`, `VLLM_TRUST_REMOTE_CODE`,
    `VLLM_ENFORCE_EAGER`, `VLLM_ENABLE_PREFIX_CACHING`, `WORKER_DEBUG_BATCH_LOGGING`,
    `VLLM_DTYPE`, `VLLM_QUANTIZATION` (see [Configuration](#configuration))

## Run

Start the local MLX worker:

```bash
python worker/worker.py
```

Start a vLLM worker on RunPod/Linux:

```bash
WORKER_BACKEND=vllm \
MODEL_ID=Qwen/Qwen2.5-7B-Instruct \
VLLM_GPU_MEMORY_UTILIZATION=0.9 \
python worker/worker.py
```

Example for an AWQ model on a single NVIDIA GPU:

```bash
WORKER_BACKEND=vllm \
MODEL_ID=Qwen/Qwen3-32B-AWQ \
VLLM_QUANTIZATION=AWQ \
VLLM_GPU_MEMORY_UTILIZATION=0.85 \
VLLM_MAX_MODEL_LEN=4096 \
VLLM_ENFORCE_EAGER=1 \
VLLM_ENABLE_PREFIX_CACHING=1 \
python worker/worker.py
```

Start the engine in a second terminal (use `queued` for any concurrent test):

```bash
cd engine
ENGINE_MODE=queued WORKER_ENDPOINT=http://127.0.0.1:50071 cargo run --bin omnispan-engine
```

Run micro-batch mode:

```bash
cd engine
ENGINE_MODE=micro_batch WORKER_ENDPOINT=http://127.0.0.1:50071 BATCH_WINDOW_MS=20 MAX_BATCH_SIZE=4 cargo run --bin omnispan-engine
```

Submit a request with `grpcurl` from the repo root:

```bash
grpcurl -plaintext -import-path ./proto -proto omnispan.proto \
  -d '{"tenant_id":"shared-basic","prompt":"Explain transformer attention in 3 sentences.","max_tokens":150}' \
  127.0.0.1:50061 omnispan.Engine/SubmitGenerate
```

Fetch engine stats:

```bash
grpcurl -plaintext -import-path ./proto -proto omnispan.proto \
  -d '{}' \
  127.0.0.1:50061 omnispan.Engine/GetEngineStats
```

## Configuration

### Serving engine environment variables

| Variable | Description | Default |
|---|---|---|
| `ENGINE_MODE` | Serving mode: `direct` (debug-only), `queued`, or `micro_batch` | `direct` |
| `BIND_HOST` | Host address to bind the engine server | `127.0.0.1` |
| `BIND_PORT` | Port number to bind the engine server | `50061` |
| `WORKER_ENDPOINT` | The endpoint of the model worker process | `http://127.0.0.1:50071` |
| `WORKER_RPC_TIMEOUT_MS` | Timeout for gRPC calls to the worker (in ms) | `30000` |
| `QUEUE_CAPACITY` | Capacity of the request scheduler queue | `1024` |
| `BATCH_WINDOW_MS` | Waiting window to accumulate requests in `micro_batch` mode (in ms) | `10` |
| `MAX_BATCH_SIZE` | Maximum number of requests grouped in a batch in `micro_batch` mode | `4` |

### Model worker environment variables

| Variable | Description | Default |
|---|---|---|
| `WORKER_BACKEND` | Backend model runtime: `mlx` (local Apple Silicon) or `vllm` (Linux GPU) | `mlx` |
| `WORKER_HOST` | Host address to bind the worker server | `127.0.0.1` |
| `WORKER_PORT` | Port number to bind the worker server | `50071` |
| `MODEL_ID` | Model identifier or path (HuggingFace) | Backend dependent (see below) |

#### MLX backend
- `MODEL_ID` defaults to `mlx-community/Qwen2.5-7B-Instruct-4bit`.

#### vLLM backend
- `MODEL_ID` defaults to `Qwen/Qwen2.5-7B-Instruct`.
- `VLLM_TENSOR_PARALLEL_SIZE`: Number of GPUs to partition the model across. Default: `1`.
- `VLLM_GPU_MEMORY_UTILIZATION`: Target GPU memory fraction. Default: `0.9`.
- `VLLM_MAX_MODEL_LEN`: Cap on the model context length.
- `VLLM_TRUST_REMOTE_CODE`: Set `1` or `true` to trust HuggingFace model code. Default: `false`.
- `VLLM_ENFORCE_EAGER`: Set `1` or `true` to disable CUDA graph capture. Default: `false`.
- `VLLM_ENABLE_PREFIX_CACHING`: Set `1` or `true` to turn on automatic prefix caching. Default: `false`.
- `VLLM_DTYPE`: Data type of weights (e.g. `half`, `float16`, `bfloat16`).
- `VLLM_QUANTIZATION`: Quantization type (e.g. `awq`, `gptq`, `squeezellm`).
- `VLLM_ASYNC`: Set `1` or `true` to use the `AsyncLLMEngine` runtime, which adds streaming (per-request tokens with real TTFT/TPOT) on top of vLLM's continuous batching. Default: `false` (synchronous `LLM`, no streaming). Requires GPU — validated on RunPod.
- `WORKER_DEBUG_BATCH_LOGGING`: Set `1` or `true` to enable verbose batch logging in worker (synchronous path only). Default: `false`.

## Benchmarking

[`bench/benchmark.py`](bench/benchmark.py) runs a concurrent load test against a
running engine and prints a JSON summary including TTFT and TPOT percentiles.

```bash
python bench/benchmark.py \
  --target 127.0.0.1:50061 \
  --requests 10 \
  --concurrency 10 \
  --max-tokens 150 \
  --mode queued
```

Options:

- `--target`: gRPC endpoint of the engine (default: `127.0.0.1:50061`).
- `--requests` / `--concurrency`: total requests and max in-flight (default: `10` / `10`).
- `--tenant-prefix` / `--tenant-count`: mock multiple tenants (default: `tenant` / `2`).
- `--max-tokens`: `max_tokens` per request (default: `150`).
- `--shared-prefix-repeats` / `--suffix-template`: build a prefix-cache-sensitive workload.
- `--mode`: label for the output JSON.

Save results to a JSON artifact:

```bash
python bench/benchmark.py --requests 10 --concurrency 10 --mode micro_batch_w10_b4 > bench/micro_batch_w10_b4.json
```

## Streaming

The engine exposes a server-streaming RPC, `SubmitGenerateStream`, that emits one
chunk per output token followed by a terminal summary chunk. This is what makes
**client-observed TTFT** a real measurement rather than a value reported after the
fact. Streaming is supported in `direct` and `queued` modes; in `micro_batch`
mode it returns `UNIMPLEMENTED` (per-request streaming out of a shared batch
requires token-level continuous batching — see the roadmap).

**Queueing.** Streaming does not flow through the unary scheduler loop (a stream
produces tokens over time and cannot fit its dispatch-then-await shape). Instead,
`queued` mode serializes streams with a dedicated 1-permit **stream gate**,
preserving one-request-at-a-time worker access — required because MLX segfaults on
concurrent native calls. Waiting streams count toward `queue_depth`, and the wait
is reported as `gate_wait_ms` on the final chunk (surfaced as `queue_wait_ms` in
the benchmark) so it is not left hiding inside client TTFT. In `direct` mode there
is no gate and no serialization. If a client disconnects mid-stream, the engine
stops draining the worker so the decode can be abandoned instead of producing
tokens nobody will read.

**Backend support.** MLX streams via `mlx_lm.stream_generate` (validated locally).
The vLLM path streams via `AsyncLLMEngine` (`WORKER_BACKEND=vllm VLLM_ASYNC=1`),
where a single engine serves streaming, unary, and batch requests and vLLM
continuously batches them at the token level — so per-request TTFT/TPOT stay
valid even under concurrency, and a client disconnect calls `engine.abort()` to
free the decode. The vLLM async runtime requires GPU and is validated on RunPod;
the default synchronous vLLM runtime does not stream.

Drive it from the benchmark with `--stream`:

```bash
python bench/benchmark.py --requests 8 --concurrency 4 --max-tokens 64 --mode queued_stream --stream
```

The summary then includes `client_ttft_ms` and `engine_ttft_ms` blocks alongside
the worker-reported `ttft_ms`, so the three layers can be compared directly.

Stream a single request with `grpcurl`:

```bash
grpcurl -plaintext -import-path ./proto -proto omnispan.proto \
  -d '{"tenant_id":"shared-basic","prompt":"Explain transformer attention in 3 sentences.","max_tokens":64}' \
  127.0.0.1:50061 omnispan.Engine/SubmitGenerateStream
```

## Backend capability matrix

Rows are engine scheduling policies (`ENGINE_MODE`); columns are the two request
modes. `SubmitGenerateStream` in `micro_batch` returns `UNIMPLEMENTED` at the
engine for both backends (per-request streaming out of a shared batch needs
token-level continuous batching).

### MLX (Apple Silicon)

| Policy | Non-streaming (`SubmitGenerate`) | Streaming (`SubmitGenerateStream`) |
|---|---|---|
| `direct` | ⚠️ debug only — **unsafe under concurrency** (concurrent native calls segfault) | ⚠️ debug only — single request streams (client/engine/worker TTFT + TPOT); concurrency still unsafe |
| `queued` | ✅ safe, serialized; judged by throughput + latency | ✅ safe, serialized by a 1-permit stream gate; full TTFT/TPOT decomposition |
| `micro_batch` | ✅ static batch (`batch_generate`), best throughput | ❌ `UNIMPLEMENTED` — static batch can't stream per request; MLX has no continuous batching |

For MLX, **TTFT/TPOT are reported only on the streaming column**. The unary path
returns the whole response at once, so a first-token time would describe an
internal decode property rather than a latency the client observes; the unary
path is judged by throughput and total latency. (vLLM's unary path still reports
TTFT/TPOT from vLLM's own per-request metrics — see below.)

### vLLM (Linux + NVIDIA GPU)

Non-streaming uses the synchronous `LLM` (default). Streaming uses the
`AsyncLLMEngine` runtime (`VLLM_ASYNC=1`), validated on a RunPod A40.

| Policy | Non-streaming (sync `LLM`, default) | Streaming (`VLLM_ASYNC=1`, async engine) |
|---|---|---|
| `direct` | ⚠️ single request; the sync `LLM` is not built for concurrent calls | ✅ **ideal** — concurrent streams all reach vLLM's continuous batcher; **250.6 tokens/s, 144 ms client TTFT** (measured) |
| `queued` | ✅ serialized; the safe way to drive the sync `LLM` under load | ⚠️ works but **starves the batcher** — 32.9 tokens/s, 17.4 s client TTFT (measured; 7.6× slower than `direct`) |
| `micro_batch` | ✅ explicit batch; TTFT/TPOT valid (vLLM's native metrics), but **partly redundant** with vLLM's own batching | ❌ `UNIMPLEMENTED` (engine-level) |

**Reading the matrices — the backends invert.** MLX has no internal scheduler, so
it *relies on the engine* to impose order: `direct` under concurrency segfaults,
and throughput comes from the engine's `micro_batch`. vLLM brings its own
continuous batching, so it wants the engine to *get out of the way*: the async
engine is happiest with `direct` + concurrency (requests flow straight into
vLLM's batcher), while engine-level `queued`/`micro_batch` become redundant or
counterproductive. This contrast is much of why production stacks fold
scheduling and batching into the runtime (vLLM/SGLang) rather than an outer
control plane.

## SLO-driven auto-tuning

[`bench/autotune.py`](bench/autotune.py) sweeps a grid of engine scheduling
configurations, drives a short load test against each, and recommends the config
that **maximizes throughput while meeting a latency SLO**. It owns the engine
lifecycle (launch with the right env → wait until serving → load → tear down);
the worker is assumed to already be running.

Throughput / client-latency SLO (unary load, sweeps all configs):

```bash
python bench/autotune.py \
  --engine-bin engine/target/debug/omnispan-engine \
  --worker-endpoint http://127.0.0.1:50071 \
  --requests 12 --concurrency 6 --max-tokens 64 \
  --windows 5,10,20 --batch-sizes 1,2,4 \
  --slo-client-p95-ms 8000
```

TTFT/TPOT SLO — add `--stream` (those metrics only exist on the streaming path
for MLX). Configs that cannot stream (`micro_batch`) are skipped, so on MLX this
effectively evaluates the stream-capable `queued` config:

```bash
python bench/autotune.py --worker-endpoint http://127.0.0.1:50071 \
  --requests 12 --concurrency 6 --max-tokens 64 --batch-sizes 1,4 \
  --stream --slo-ttft-p95-ms 1500 --slo-tpot-p95-ms 40
```

If an SLO targets a metric a config cannot measure (a TTFT/TPOT SLO under
`micro_batch`, or without `--stream` on MLX), the tuner records an SLO violation
rather than passing the config on an unmeasured metric — so it never recommends a
config on the strength of a number it did not actually capture. This also
surfaces a real MLX limitation: the configs that give throughput (batching)
cannot stream, and the config that streams (`queued`) does not batch — you cannot
have both without continuous batching.

## Benchmark results

### 1. Local MLX performance (Apple Silicon)
*Model: `mlx-community/Qwen2.5-7B-Instruct-4bit` | Concurrency: 10 | Requests: 10 | Max Tokens: 150. Artifacts: [`bench/`](bench/) (`queued.json`, `micro_batch_w*_b*.json`).*

| Serving Configuration | Success | Wall Clock | Throughput | Worker Latency (p50) | Queue Wait (p50) | Client Latency (p50) |
|---|---|---|---|---|---|---|
| **Queued** | 10/10 | 22.75s | 71.65 tokens/s | 2,248 ms | 10,337 ms | 12,640 ms |
| **Micro-Batch (w10/b2)** | 10/10 | 14.73s | 110.68 tokens/s | 2,921 ms | 5,895 ms | 8,858 ms |
| **Micro-Batch (w10/b4)** | 10/10 | 14.21s | 114.73 tokens/s | 5,484 ms | 5,511 ms | 11,107 ms |
| **Micro-Batch (w10/b8)** | 10/10 | 14.73s | 110.69 tokens/s | 11,723 ms | 12 ms | 11,745 ms |
| **Micro-Batch (w20/b4)** | 10/10 | 13.91s | **117.22 tokens/s** | 5,426 ms | 5,511 ms | 10,946 ms |

- **`direct` is unsafe under concurrency:** one request is fine (c=1 baseline: 67.9 tokens/s, 2,306 ms), but concurrent `direct` invocation segfaults the MLX runtime, so it is not run at c=10.
- **Queued serializes:** safe and stable, but queue wait dominates (~10.3s p50) under concurrency.
- **Micro-batch ≈1.6× throughput:** batching lifts throughput to ~117 tokens/s (w20/b4) and cuts client latency vs queued. Batch 8 pushes worker latency to ~11.7s (the whole batch decodes together) for no gain over batch 4.

### 2. Streaming TTFT / TPOT decomposition (MLX, queued)
*Model: `mlx-community/Qwen2.5-7B-Instruct-4bit` | Concurrency: 8 | Requests: 8 | Max Tokens: 150 | streaming. Artifact: [`bench/queued_stream.json`](bench/queued_stream.json).*

| Vantage point | TTFT (p50) | TTFT (p95) | TPOT (p50) | TPOT (p95) |
|---|---|---|---|---|
| **Worker** (model prefill / decode) | 208 ms | 298 ms | 13.8 ms | 14.4 ms |
| **Engine** (dispatch → first forward) | 211 ms | 301 ms | — | — |
| **Client** (on the wire) | 8,391 ms | 15,596 ms | 13.9 ms | 14.5 ms |
| *of which: gate wait* (`queue_wait_ms`) | *8,174 ms* | *15,388 ms* | — | — |

- **The latency budget closes:** client TTFT ≈ gate wait + engine TTFT (8,391 ≈ 8,174 + 211). At concurrency 8 with a single worker, **97% of the first-token latency is queueing**, not the model.
- **Queueing lives in TTFT, not TPOT:** client TPOT (13.9 ms) ≈ worker TPOT (13.8 ms) — once a request is streaming, tokens flow at the decode rate regardless of load. If TPOT is healthy but TTFT is not, the bottleneck is admission/queueing, not the model.
- **Engine overhead is negligible:** engine TTFT tracks worker TTFT within ~3 ms; the time is in the model and the queue, not the control plane.
- TTFT/TPOT come from the streaming path; the unary and `micro_batch` paths are judged by throughput and total latency (table 1).

### 3. Scheduling policy vs. continuous batching (vLLM, RunPod A40)
*Model: `Qwen/Qwen3-32B-AWQ` | Concurrency: 8 | Requests: 8 | Max Tokens: 150 | streaming (`VLLM_ASYNC=1`). Artifacts: [`bench/runpod/qwen3_32b_awq_async_*_stream_8x8.json`](bench/runpod/).*

Identical load and identical worker; **only the engine's scheduling policy differs.**

| Metric | `queued` (stream gate serializes) | `direct` (concurrency passes through) |
|---|---|---|
| Throughput | 32.9 tokens/s | **250.6 tokens/s** (7.6×) |
| Wall clock | 39.6 s | **5.2 s** |
| Worker TTFT (p50) | 47 ms | 133 ms |
| **Client TTFT (p50)** | **17,369 ms** | **144 ms** (120× better) |
| Queue wait (p50) | 17,309 ms | 0.05 ms |
| TPOT (p50) | 32.9 ms | 33.9 ms |

- **Serializing in front of a continuous batcher destroys it:** in `queued` the worker still answers in 47 ms, but 99.6% of client-observed latency is engine queue wait, because vLLM only ever sees one request at a time and has nothing to batch.
- **TPOT barely moves (32.9 → 33.9 ms) while throughput grows 7.6×.** That is continuous batching working: vLLM interleaves 8 sequences through shared decode steps, so per-token latency is nearly unchanged while aggregate output multiplies.
- **This is the exact inverse of MLX.** MLX segfaults on concurrent `direct` calls and *needs* the engine to serialize or batch; vLLM needs the engine to get out of the way. The right control-plane policy is a property of the runtime beneath it — which is the argument for folding scheduling into the inference engine (vLLM/SGLang) rather than layering it on top.

### 4. Admission-limit sweep (vLLM, RunPod A40)
*Model: `Qwen/Qwen3-32B-AWQ` | 32 requests at client concurrency 32 | Max Tokens: 150 | streaming. Sweeps `MAX_CONCURRENT_STREAMS`. Artifact: [`bench/runpod/qwen3_32b_awq_concurrency_sweep.json`](bench/runpod/qwen3_32b_awq_concurrency_sweep.json).*

| `MAX_CONCURRENT_STREAMS` | Throughput | vs N=1 | Client TTFT p50 | Client TTFT p95 | TPOT p50 |
|---|---|---|---|---|---|
| 1 | 32.9 tokens/s | 1.0× | 76.8 s | 146.1 s | 33.0 ms |
| 2 | 65.0 tokens/s | 2.0× | 37.7 s | 72.6 s | 33.1 ms |
| 4 | 128.9 tokens/s | 3.9× | 17.8 s | 35.5 s | 33.3 ms |
| 8 | 250.6 tokens/s | 7.6× | 7.9 s | 15.7 s | 34.0 ms |
| 16 | 467.5 tokens/s | 14.2× | 2.9 s | 5.8 s | 36.0 ms |
| unlimited (= 32) | **859.9 tokens/s** | **26.2×** | **0.33 s** | **0.34 s** | 38.2 ms |

A second sweep at higher offered load (512 requests, client concurrency 256) extends the curve past the point where the client stopped being the bottleneck. Artifact: [`bench/runpod/qwen3_32b_awq_concurrency_sweep_hi.json`](bench/runpod/qwen3_32b_awq_concurrency_sweep_hi.json).

| N | Throughput | Marginal gain | TPOT p50 |
|---|---|---|---|
| 32 | 860.2 tokens/s | +84% | 38.7 ms |
| 64 | 1,170.4 tokens/s | +36% | 55.3 ms |
| **128** | **1,258.6 tokens/s** | **+7.5%** | 103.7 ms |
| 256 | 1,260.2 tokens/s | +0.1% | 103.8 ms |
| unlimited | 1,266.4 tokens/s | +0.5% | 102.0 ms |

*N=32 appears in both sweeps at 859.9 and 860.2 tokens/s — an independent reproducibility check on the method.*

**Two inflection points, on two different curves, at different N:**

- **Throughput saturation: N≈128** (~1,260 tokens/s). Beyond it the engine's gate is a no-op — N=256 and unlimited are identical, because vLLM's own limits (KV cache, `max_num_seqs`) bind before ours does.
- **Latency knee: N≈32–64.** TPOT is flat to N=32 (33 → 39 ms), then degrades sharply: 55 ms at 64, 104 ms at 128.

Because those points differ, **throughput keeps improving after per-token latency has already begun to fall apart** — which is why the throughput-optimal and SLO-optimal settings disagree:

| Objective | Best N | Result |
|---|---|---|
| Maximum throughput | 128 | 1,259 tokens/s, TPOT 104 ms |
| TPOT SLO ≤ 50 ms (≈ reading speed) | **32** | 860 tokens/s (68% of peak), TPOT 39 ms |

Which is correct depends on whether the tier is interactive or bulk.

- **At this offered load no N met the 2 s TTFT p95 SLO** (all ≥17 s). When an SLO is unreachable at *every* admission setting, the answer is not tuning — it is more replicas. That is the autoscaling signal, and the point where a single-node control plane runs out of moves.
- **Consequence for the control plane:** treat the admission limit as a *safety valve* (KV memory, tenant quotas, TPOT protection), not a throughput dial — and prefer driving it from measured KV-cache pressure (`gpu_cache_usage_perc`) rather than a static constant, since both inflection points move with workload shape. The opposite holds for MLX, where the gate is mandatory for safety.

> **Workload dependence — read the numbers narrowly.** These sweeps use 13-token
> prompts and 150-token outputs (11.5:1 output-to-input), uniform across requests:
> a **decode-bound workload with negligible prefill**. That is why batching scaled
> so cheaply — adding sequences to a memory-bandwidth-bound decode step is nearly
> free until bandwidth or KV memory runs out. Long prompts would move saturation
> much earlier (prefill is compute-bound and stalls decode for every sequence in
> the batch); longer or mixed-length outputs would too, by consuming KV faster and
> introducing stragglers. N≈128 characterizes *this* workload on *this* GPU, not
> serving in general.

### 5. RunPod A40 performance (vLLM, Automatic Prefix Caching)
*Model: `Qwen/Qwen3-32B-AWQ` | Concurrency: 2 | Requests: 2 | Max Tokens: 64 | Prefix-cache workload (6× repeated policy prefix)*

| Configuration | Wall Clock | Throughput | Worker Latency (p50) | Queue Wait (p50) | Throughput Gain |
|---|---|---|---|---|---|
| **Prefix Caching OFF** | 4.21s | 379.66 tokens/s | 4,149 ms | 11 ms | Baseline |
| **Prefix Caching ON** | 2.87s | **556.71 tokens/s** | **2,820 ms** | 10 ms | **1.47×** |

- **Prefix-cache savings:** vLLM's automatic prefix reuse saves substantial prefill time for prompts with shared prefixes.
- **Worker-side speedup:** queue wait stayed constant (~10–11 ms), confirming the gain is on the worker runtime, not the engine.
- The vLLM path also emits per-request TTFT/TPOT (read from vLLM's native request metrics); those columns will be captured on the next RunPod run.

## Scope and limitations

Omnispan is a **single-node testbed**, and its claims are bounded accordingly:

- One engine, one worker, one GPU/accelerator — no distributed scheduling, no multi-node parallelism.
- Backends cover **NVIDIA (vLLM)** and **Apple Silicon (MLX)** only; no AMD/ROCm.
- TTFT is measured on the streaming path; under batching it is derived/unmeasured as noted above.
- Benchmark samples here are small-N and meant to show tradeoff *shape*, not publishable percentiles.

Cluster-scale concerns (multi-scheduler K8s architectures, topology-aware
placement, fractional GPU allocation, checkpoint/restore, workload isolation)
are intentionally out of scope for the running code and are captured as design
notes instead.

## Design notes and roadmap

See [`docs/`](docs/) for design and planning notes. Planned next steps, roughly
in priority order:

- Prefix-aware / cache-affinity routing (group requests by shared prefix)
- Multi-tenant weighted-fair-queue scheduling with per-tenant fairness metrics
- SGLang worker backend behind the existing runtime interface
- Disaggregated prefill/decode workers with KV handoff (the "prefill leader / decode worker / KV router" pattern)
- Streaming under `micro_batch` via token-level continuous batching (the reason engines like vLLM exist)
- Priority classes in the streaming gate (online preempts batch) — see [docs/batch-tier.md](docs/batch-tier.md)
- Adaptive admission driven by measured KV-cache pressure (`gpu_cache_usage_perc`) instead of a static limit

End-to-end response streaming (`SubmitGenerateStream`) is implemented for
`direct` and `queued` modes, with client/engine/worker TTFT decomposition. The
MLX path is validated locally; the vLLM `AsyncLLMEngine` streaming path
(`VLLM_ASYNC=1`) is validated on a RunPod A40 (see results table 3).

## Regenerate Python gRPC stubs

If `proto/omnispan.proto` changes:

```bash
python -m grpc_tools.protoc \
  -I proto \
  --python_out=worker/generated \
  --grpc_python_out=worker/generated \
  proto/omnispan.proto
```

The Rust stubs regenerate automatically via `engine/build.rs` on `cargo build`.

## Notes

- The worker must run in a Python environment with the selected backend's dependencies installed.
- The engine auto-generates a request ID if the client omits one.
- Treat `direct` mode as debug-only; concurrent direct mode has triggered Python worker segmentation faults in the MLX runtime.
- `BATCH_WINDOW_MS` controls how long the engine waits to gather requests in `micro_batch` mode; `MAX_BATCH_SIZE` caps how many are grouped into one worker batch.
- The worker gRPC contract is unchanged across backends; the backend switch is entirely inside `worker/`.
- Worker startup fails loudly if `WORKER_HOST:WORKER_PORT` is already occupied, which helps catch stale worker processes instead of silently hitting the wrong instance.
- Benchmark artifacts are saved in [`bench/`](bench/) and [`bench/runpod/`](bench/runpod/).
