# Omnispan

Omnispan is a tiny Token Factory perf lab.

Current state:

- `engine/`: Rust direct-path engine
- `worker/`: Python MLX worker
- `proto/`: shared gRPC contract
- `bench/`: benchmark scripts and artifacts
- `docs/`: design and planning notes

## Modes

The current implementation supports direct, queued, and micro-batch modes:

- client -> Rust engine
- engine -> Python worker over gRPC
- worker -> MLX model runtime

Queued mode adds explicit in-engine queue ownership but still executes one request at a time.
Micro-batch mode adds a short batching window and groups pending requests before dispatching them to the worker batch path.

Important:

- Treat `direct` mode as debug-only.
- Under concurrent direct load, the current Python MLX worker has crashed in native code.
- Use `queued` mode for any meaningful load test or benchmark until worker-side parallel safety is proven.

## Prerequisites

- Rust toolchain
- `python`
- Python environment with the worker dependencies installed
- `grpcurl` for manual testing

Install Python dependencies:

```bash
python -m pip install -r worker/requirements.txt
```

## Run

Start the worker:

```bash
python worker/worker.py
```

Start the engine in a second terminal:

```bash
cd engine
ENGINE_MODE=direct WORKER_ENDPOINT=http://127.0.0.1:50071 cargo run --bin omnispan-engine
```

Use direct mode only for single-request debugging.

Run queued mode instead:

```bash
cd engine
ENGINE_MODE=queued WORKER_ENDPOINT=http://127.0.0.1:50071 cargo run --bin omnispan-engine
```

Use queued mode for benchmarks and concurrent tests.

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

## Regenerate Python gRPC Stubs

If `proto/omnispan.proto` changes:

```bash
python -m grpc_tools.protoc \
  -I proto \
  --python_out=worker/generated \
  --grpc_python_out=worker/generated \
  proto/omnispan.proto
```

## Notes

- The worker must run in a Python environment that has `mlx_lm` installed.
- The engine auto-generates a request ID if the client omits one.
- Concurrent direct mode has triggered Python worker segmentation faults in the current MLX runtime path.
- `BATCH_WINDOW_MS` controls how long the engine waits to gather additional requests in `micro_batch` mode.
- `MAX_BATCH_SIZE` controls how many pending requests are grouped into one worker batch.
- Benchmark artifacts from the earlier FastAPI prototype are in `bench/`.
