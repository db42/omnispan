import argparse
import json
import math
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import grpc


DEFAULT_PROMPT = "Explain how a transformer attention mechanism works in 3 sentences."
ROOT_DIR = Path(__file__).resolve().parents[1]
GENERATED_DIR = ROOT_DIR / "worker" / "generated"
if str(GENERATED_DIR) not in sys.path:
    sys.path.insert(0, str(GENERATED_DIR))

import omnispan_pb2  # noqa: E402
import omnispan_pb2_grpc  # noqa: E402


QUEUE_WAIT_RE = re.compile(r"queue_wait_ms=([0-9]+(?:\.[0-9]+)?)")


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * pct
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return values[lower]
    weight = rank - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def parse_queue_wait_ms(status: str) -> float:
    match = QUEUE_WAIT_RE.search(status or "")
    if not match:
        return 0.0
    return float(match.group(1))


def send_request(target: str, tenant_id: str, prompt: str, max_tokens: int, timeout: float) -> dict:
    start = time.perf_counter()
    try:
        with grpc.insecure_channel(target) as channel:
            stub = omnispan_pb2_grpc.EngineStub(channel)
            response = stub.SubmitGenerate(
                omnispan_pb2.GenerateRequest(
                    tenant_id=tenant_id,
                    prompt=prompt,
                    max_tokens=max_tokens,
                ),
                timeout=timeout,
            )

        elapsed_ms = (time.perf_counter() - start) * 1000
        queue_wait_ms = parse_queue_wait_ms(response.status)
        return {
            "ok": response.error_message == "",
            "status": response.status,
            "elapsed_ms": elapsed_ms,
            "worker_latency_ms": float(response.worker_latency_ms),
            "end_to_end_latency_ms": float(response.end_to_end_latency_ms),
            "queue_wait_ms": queue_wait_ms,
            "input_tokens": int(response.input_tokens),
            "output_tokens": int(response.output_tokens),
            "tenant_id": tenant_id,
            "error": response.error_message,
        }
    except grpc.RpcError as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "ok": False,
            "status": exc.code().name if callable(exc.code) else None,
            "elapsed_ms": elapsed_ms,
            "error": exc.details() if callable(exc.details) else str(exc),
            "tenant_id": tenant_id,
        }
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "ok": False,
            "status": None,
            "elapsed_ms": elapsed_ms,
            "error": str(exc),
            "tenant_id": tenant_id,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a concurrent benchmark against the gRPC engine."
    )
    parser.add_argument(
        "--target",
        default="127.0.0.1:50061",
        help="gRPC target for the engine.",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=10,
        help="Total number of requests to send.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Maximum number of in-flight requests.",
    )
    parser.add_argument(
        "--tenant-prefix",
        default="tenant",
        help="Tenant ID prefix. Requests rotate across this prefix plus an index.",
    )
    parser.add_argument(
        "--tenant-count",
        type=int,
        default=2,
        help="Number of tenant IDs to rotate across.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt to send to the model.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=150,
        help="max_tokens field for each request.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--mode",
        default="unknown",
        help="Mode label to include in the output artifact.",
    )
    return parser


def summarize(results: list[dict], started_at: float, finished_at: float, mode: str, target: str) -> dict:
    total_elapsed_s = finished_at - started_at
    success_results = [result for result in results if result["ok"]]
    failure_results = [result for result in results if not result["ok"]]

    client_latencies = sorted(result["elapsed_ms"] for result in success_results)
    worker_latencies = sorted(result["worker_latency_ms"] for result in success_results)
    engine_latencies = sorted(result["end_to_end_latency_ms"] for result in success_results)
    queue_waits = sorted(result.get("queue_wait_ms", 0.0) for result in success_results)
    total_input_tokens = sum(result.get("input_tokens", 0) for result in success_results)
    total_output_tokens = sum(result.get("output_tokens", 0) for result in success_results)
    total_tokens = total_input_tokens + total_output_tokens

    return {
        "target": target,
        "mode": mode,
        "total_requests": len(results),
        "successful_requests": len(success_results),
        "failed_requests": len(failure_results),
        "wall_clock_seconds": round(total_elapsed_s, 2),
        "requests_per_second": round(len(success_results) / total_elapsed_s, 2) if total_elapsed_s else 0.0,
        "tokens_per_second": round(total_tokens / total_elapsed_s, 2) if total_elapsed_s else 0.0,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "client_latency_ms": {
            "p50": round(percentile(client_latencies, 0.50), 2),
            "p95": round(percentile(client_latencies, 0.95), 2),
            "p99": round(percentile(client_latencies, 0.99), 2),
            "mean": round(statistics.fmean(client_latencies), 2) if client_latencies else 0.0,
        },
        "engine_latency_ms": {
            "p50": round(percentile(engine_latencies, 0.50), 2),
            "p95": round(percentile(engine_latencies, 0.95), 2),
            "p99": round(percentile(engine_latencies, 0.99), 2),
            "mean": round(statistics.fmean(engine_latencies), 2) if engine_latencies else 0.0,
        },
        "worker_latency_ms": {
            "p50": round(percentile(worker_latencies, 0.50), 2),
            "p95": round(percentile(worker_latencies, 0.95), 2),
            "p99": round(percentile(worker_latencies, 0.99), 2),
            "mean": round(statistics.fmean(worker_latencies), 2) if worker_latencies else 0.0,
        },
        "queue_wait_ms": {
            "p50": round(percentile(queue_waits, 0.50), 2),
            "p95": round(percentile(queue_waits, 0.95), 2),
            "p99": round(percentile(queue_waits, 0.99), 2),
            "mean": round(statistics.fmean(queue_waits), 2) if queue_waits else 0.0,
        },
        "failures": [
            {
                "status": result["status"],
                "error": result.get("error", ""),
                "tenant_id": result["tenant_id"],
            }
            for result in failure_results
        ],
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.requests <= 0:
        parser.error("--requests must be positive")
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    if args.tenant_count <= 0:
        parser.error("--tenant-count must be positive")

    started_at = time.perf_counter()
    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = []
        for index in range(args.requests):
            tenant_index = index % args.tenant_count
            tenant_id = f"{args.tenant_prefix}-{tenant_index + 1}"
            futures.append(
                executor.submit(
                    send_request,
                    args.target,
                    tenant_id,
                    args.prompt,
                    args.max_tokens,
                    args.timeout,
                )
            )

        for future in as_completed(futures):
            results.append(future.result())

    finished_at = time.perf_counter()
    summary = summarize(results, started_at, finished_at, args.mode, args.target)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
