# Omnispan

Omnispan is a tiny Token Factory perf lab.

Current state:

- `engine/`: Rust direct-path engine
- `worker/`: Python MLX worker
- `proto/`: shared gRPC contract
- `bench/`: benchmark scripts and artifacts
- `docs/`: design and planning notes

## Direct Mode

The current implementation supports direct mode only:

- client -> Rust engine
- engine -> Python worker over gRPC
- worker -> MLX model runtime

No queueing or batching is implemented yet.

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
- Benchmark artifacts from the earlier FastAPI prototype are in `bench/`.
