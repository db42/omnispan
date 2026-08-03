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
DEFAULT_APC_SHARED_BLOCK = (
    "You are an enterprise research assistant serving risk and procurement teams. "
    "Follow these rules exactly. "
    "Always answer in concise bullet points. "
    "Use factual language only. "
    "Do not speculate beyond the provided policy frame. "
    "Highlight operational, security, compliance, commercial, and dependency risks separately. "
    "When evidence is weak, say that evidence is limited. "
    "Do not include narrative introductions or conclusions. "
    "Prefer short phrases over long sentences. "
    "Preserve section order exactly as provided. "
    "Use this schema for every answer: "
    "Operational Risk, Security Risk, Compliance Risk, Commercial Risk, Dependency Risk. "
    "If a category is not applicable, say not material. "
)
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


def build_prompt(
    base_prompt: str,
    request_index: int,
    shared_prefix_repeats: int,
    suffix_template: str,
) -> str:
    if shared_prefix_repeats <= 0:
        return base_prompt

    shared_prefix = (DEFAULT_APC_SHARED_BLOCK + "\n") * shared_prefix_repeats
    suffix = suffix_template.format(index=request_index + 1)
    return f"{shared_prefix}\nQuestion: {suffix}"


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
            "ttft_ms": float(response.ttft_ms),
            "tpot_ms": float(response.tpot_ms),
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


def send_request_stream(
    target: str, tenant_id: str, prompt: str, max_tokens: int, timeout: float
) -> dict:
    """Drive the server-streaming endpoint and measure client-observed timing.

    TTFT is captured at the client the instant the first token chunk lands on
    the wire -- the number the user actually experiences -- and compared against
    the engine- and worker-reported TTFT carried in the final chunk.
    """
    start = time.perf_counter()
    first_chunk_at = None
    last_chunk_at = None
    inter_chunk_ms: list[float] = []
    text_parts: list[str] = []
    final = None
    try:
        with grpc.insecure_channel(target) as channel:
            stub = omnispan_pb2_grpc.EngineStub(channel)
            responses = stub.SubmitGenerateStream(
                omnispan_pb2.GenerateRequest(
                    tenant_id=tenant_id,
                    prompt=prompt,
                    max_tokens=max_tokens,
                ),
                timeout=timeout,
            )
            for chunk in responses:
                now = time.perf_counter()
                if chunk.finished:
                    final = chunk
                    continue
                if first_chunk_at is None:
                    first_chunk_at = now
                else:
                    inter_chunk_ms.append((now - last_chunk_at) * 1000)
                last_chunk_at = now
                text_parts.append(chunk.text_delta)

        elapsed_ms = (time.perf_counter() - start) * 1000
        client_ttft_ms = ((first_chunk_at - start) * 1000) if first_chunk_at is not None else 0.0
        client_tpot_ms = statistics.fmean(inter_chunk_ms) if inter_chunk_ms else 0.0
        ok = final is not None and final.status == "ok" and final.error_message == ""
        return {
            "ok": ok,
            "status": final.status if final is not None else "no_final_chunk",
            "elapsed_ms": elapsed_ms,
            "worker_latency_ms": float(final.worker_latency_ms) if final is not None else 0.0,
            "end_to_end_latency_ms": float(final.end_to_end_latency_ms) if final is not None else elapsed_ms,
            # The stream gate is the streaming path's queue, so its wait is
            # reported as queue_wait_ms to stay comparable with the unary path.
            "queue_wait_ms": float(final.gate_wait_ms) if final is not None else 0.0,
            "ttft_ms": float(final.ttft_ms) if final is not None else 0.0,
            "tpot_ms": float(final.tpot_ms) if final is not None else 0.0,
            "client_ttft_ms": client_ttft_ms,
            "client_tpot_ms": client_tpot_ms,
            "engine_ttft_ms": float(final.engine_ttft_ms) if final is not None else 0.0,
            "input_tokens": int(final.input_tokens) if final is not None else 0,
            "output_tokens": int(final.output_tokens) if final is not None else 0,
            "tenant_id": tenant_id,
            "error": final.error_message if final is not None else "stream ended without final chunk",
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
        "--shared-prefix-repeats",
        type=int,
        default=0,
        help=(
            "Repeat a built-in enterprise policy block this many times before appending "
            "a short varying suffix. Use this to create a prefix-cache-sensitive workload."
        ),
    )
    parser.add_argument(
        "--suffix-template",
        default="Explain the top vendor risks for company {index} in 3 bullets.",
        help=(
            "Suffix appended after the shared prefix when --shared-prefix-repeats is used. "
            "Use {index} to vary requests."
        ),
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
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use the server-streaming endpoint and measure client-observed TTFT.",
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
    # TTFT is only measured on the streaming (non-batch) worker path, which
    # reports 0.0 when it cannot observe first-token timing. Filter those out so
    # the percentiles describe requests where TTFT was actually captured.
    ttfts = sorted(r["ttft_ms"] for r in success_results if r.get("ttft_ms", 0.0) > 0.0)
    tpots = sorted(r["tpot_ms"] for r in success_results if r.get("tpot_ms", 0.0) > 0.0)
    client_ttfts = sorted(r["client_ttft_ms"] for r in success_results if r.get("client_ttft_ms", 0.0) > 0.0)
    client_tpots = sorted(r["client_tpot_ms"] for r in success_results if r.get("client_tpot_ms", 0.0) > 0.0)
    engine_ttfts = sorted(r["engine_ttft_ms"] for r in success_results if r.get("engine_ttft_ms", 0.0) > 0.0)
    total_input_tokens = sum(result.get("input_tokens", 0) for result in success_results)
    total_output_tokens = sum(result.get("output_tokens", 0) for result in success_results)
    total_tokens = total_input_tokens + total_output_tokens

    summary = {
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

    # TTFT/TPOT are only emitted when actually measured (streaming and unary
    # paths). Synchronous batch has no per-request first-token or inter-token
    # timing, so these blocks are simply omitted there — judge that path by
    # throughput and the latency percentiles above, as the earlier bench/*.json
    # artifacts do.
    if ttfts:
        summary["ttft_ms"] = {
            "measured_samples": len(ttfts),
            "p50": round(percentile(ttfts, 0.50), 2),
            "p95": round(percentile(ttfts, 0.95), 2),
            "p99": round(percentile(ttfts, 0.99), 2),
            "mean": round(statistics.fmean(ttfts), 2),
        }
    if tpots:
        summary["tpot_ms"] = {
            "measured_samples": len(tpots),
            "p50": round(percentile(tpots, 0.50), 2),
            "p95": round(percentile(tpots, 0.95), 2),
            "p99": round(percentile(tpots, 0.99), 2),
            "mean": round(statistics.fmean(tpots), 2),
        }

    # Streaming-only: client- and engine-observed TTFT, so the layers can be
    # compared against the worker-reported ttft_ms above.
    if client_ttfts:
        summary["client_ttft_ms"] = {
            "measured_samples": len(client_ttfts),
            "p50": round(percentile(client_ttfts, 0.50), 2),
            "p95": round(percentile(client_ttfts, 0.95), 2),
            "p99": round(percentile(client_ttfts, 0.99), 2),
            "mean": round(statistics.fmean(client_ttfts), 2),
        }
    if client_tpots:
        summary["client_tpot_ms"] = {
            "measured_samples": len(client_tpots),
            "p50": round(percentile(client_tpots, 0.50), 2),
            "p95": round(percentile(client_tpots, 0.95), 2),
            "p99": round(percentile(client_tpots, 0.99), 2),
            "mean": round(statistics.fmean(client_tpots), 2),
        }
    if engine_ttfts:
        summary["engine_ttft_ms"] = {
            "measured_samples": len(engine_ttfts),
            "p50": round(percentile(engine_ttfts, 0.50), 2),
            "p95": round(percentile(engine_ttfts, 0.95), 2),
            "p99": round(percentile(engine_ttfts, 0.99), 2),
            "mean": round(statistics.fmean(engine_ttfts), 2),
        }

    return summary


def run_load(
    target: str,
    requests: int,
    concurrency: int,
    *,
    mode: str = "unknown",
    tenant_prefix: str = "tenant",
    tenant_count: int = 2,
    prompt: str = DEFAULT_PROMPT,
    shared_prefix_repeats: int = 0,
    suffix_template: str = "Explain the top vendor risks for company {index} in 3 bullets.",
    max_tokens: int = 150,
    timeout: float = 120.0,
    stream: bool = False,
) -> dict:
    """Run one concurrent load test and return the summary dict.

    Importable so the auto-tuner can drive the same load generator across a
    sweep of engine configurations without shelling out to this module. When
    ``stream`` is set, requests use the server-streaming endpoint.
    """
    sender = send_request_stream if stream else send_request
    started_at = time.perf_counter()
    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for index in range(requests):
            tenant_index = index % tenant_count
            tenant_id = f"{tenant_prefix}-{tenant_index + 1}"
            request_prompt = build_prompt(
                base_prompt=prompt,
                request_index=index,
                shared_prefix_repeats=shared_prefix_repeats,
                suffix_template=suffix_template,
            )
            futures.append(
                executor.submit(
                    sender,
                    target,
                    tenant_id,
                    request_prompt,
                    max_tokens,
                    timeout,
                )
            )

        for future in as_completed(futures):
            results.append(future.result())

    finished_at = time.perf_counter()
    return summarize(results, started_at, finished_at, mode, target)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.requests <= 0:
        parser.error("--requests must be positive")
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    if args.tenant_count <= 0:
        parser.error("--tenant-count must be positive")

    summary = run_load(
        target=args.target,
        requests=args.requests,
        concurrency=args.concurrency,
        mode=args.mode,
        tenant_prefix=args.tenant_prefix,
        tenant_count=args.tenant_count,
        prompt=args.prompt,
        shared_prefix_repeats=args.shared_prefix_repeats,
        suffix_template=args.suffix_template,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        stream=args.stream,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
