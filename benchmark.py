import argparse
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


DEFAULT_PROMPT = "Explain how a transformer attention mechanism works in 3 sentences."


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


def send_request(url: str, tenant_id: str, prompt: str, max_tokens: int, timeout: float) -> dict:
    payload = {
        "tenant_id": tenant_id,
        "prompt": prompt,
        "max_tokens": max_tokens,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read()
            elapsed_ms = (time.perf_counter() - start) * 1000
            result = json.loads(response_body.decode("utf-8"))
            return {
                "ok": True,
                "status_code": response.status,
                "elapsed_ms": elapsed_ms,
                "server_latency_ms": float(result.get("latency_ms", 0.0)),
                "input_tokens": int(result.get("input_tokens", 0)),
                "output_tokens": int(result.get("output_tokens", 0)),
                "tenant_id": tenant_id,
            }
    except urllib.error.HTTPError as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        details = exc.read().decode("utf-8", errors="replace")
        return {
            "ok": False,
            "status_code": exc.code,
            "elapsed_ms": elapsed_ms,
            "error": details,
            "tenant_id": tenant_id,
        }
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "ok": False,
            "status_code": None,
            "elapsed_ms": elapsed_ms,
            "error": str(exc),
            "tenant_id": tenant_id,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a concurrent benchmark against the /generate endpoint."
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/generate",
        help="Full /generate endpoint URL.",
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
    return parser


def summarize(results: list[dict], started_at: float, finished_at: float) -> dict:
    total_elapsed_s = finished_at - started_at
    success_results = [result for result in results if result["ok"]]
    failure_results = [result for result in results if not result["ok"]]

    client_latencies = sorted(result["elapsed_ms"] for result in success_results)
    server_latencies = sorted(result["server_latency_ms"] for result in success_results)
    total_input_tokens = sum(result.get("input_tokens", 0) for result in success_results)
    total_output_tokens = sum(result.get("output_tokens", 0) for result in success_results)
    total_tokens = total_input_tokens + total_output_tokens

    return {
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
        "server_latency_ms": {
            "p50": round(percentile(server_latencies, 0.50), 2),
            "p95": round(percentile(server_latencies, 0.95), 2),
            "p99": round(percentile(server_latencies, 0.99), 2),
            "mean": round(statistics.fmean(server_latencies), 2) if server_latencies else 0.0,
        },
        "failures": [
            {
                "status_code": result["status_code"],
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
                    args.url,
                    tenant_id,
                    args.prompt,
                    args.max_tokens,
                    args.timeout,
                )
            )

        for future in as_completed(futures):
            results.append(future.result())

    finished_at = time.perf_counter()
    summary = summarize(results, started_at, finished_at)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
