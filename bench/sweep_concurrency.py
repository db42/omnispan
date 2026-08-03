"""Sweep the engine's streaming admission limit (MAX_CONCURRENT_STREAMS).

The admission limit is the outer bound on the batch a continuous-batching
runtime can form: N=1 idles the batcher, N=unlimited lets it saturate. This
sweep measures throughput and TTFT/TPOT across N to find the knee -- the point
where throughput has saturated but p95 TTFT still meets an SLO. That N is what
a real gateway would enforce.

Assumes a worker is already running. Owns the engine lifecycle per trial.

    python bench/sweep_concurrency.py \
      --engine-bin engine/target/release/omnispan-engine \
      --worker-endpoint http://127.0.0.1:50071 \
      --limits 1,2,4,8,16,32 --requests 32 --concurrency 32 --max-tokens 150 \
      --out bench/runpod/concurrency_sweep.json
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import grpc

ROOT_DIR = Path(__file__).resolve().parents[1]
GENERATED_DIR = ROOT_DIR / "worker" / "generated"
if str(GENERATED_DIR) not in sys.path:
    sys.path.insert(0, str(GENERATED_DIR))

import omnispan_pb2  # noqa: E402
import omnispan_pb2_grpc  # noqa: E402

from benchmark import run_load  # noqa: E402


def wait_until_serving(target: str, timeout_s: float) -> bool:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        try:
            with grpc.insecure_channel(target) as channel:
                omnispan_pb2_grpc.EngineStub(channel).GetEngineStats(
                    omnispan_pb2.StatsRequest(), timeout=1.0
                )
                return True
        except grpc.RpcError:
            time.sleep(0.2)
    return False


def start_engine(engine_bin, host, port, worker_endpoint, limit, mode, log_path):
    env = os.environ.copy()
    env.update(
        {
            "ENGINE_MODE": mode,
            "BIND_HOST": host,
            "BIND_PORT": str(port),
            "WORKER_ENDPOINT": worker_endpoint,
            "MAX_CONCURRENT_STREAMS": str(limit),
        }
    )
    return subprocess.Popen(
        [engine_bin],
        env=env,
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def stop_engine(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine-bin", default="engine/target/release/omnispan-engine")
    ap.add_argument("--bind-host", default="127.0.0.1")
    ap.add_argument("--bind-port", type=int, default=50063)
    ap.add_argument("--worker-endpoint", default="http://127.0.0.1:50071")
    ap.add_argument("--mode", default="queued", help="Engine mode hosting the gate.")
    ap.add_argument("--limits", default="1,2,4,8,16,32", help="MAX_CONCURRENT_STREAMS values (0=unlimited).")
    ap.add_argument("--requests", type=int, default=32)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=150)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--slo-ttft-p95-ms", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--log-dir", default="/tmp/omnispan_sweep")
    args = ap.parse_args()

    limits = [int(v) for v in args.limits.split(",") if v.strip()]
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    target = f"{args.bind_host}:{args.bind_port}"
    rows = []

    for limit in limits:
        label = f"N={'unlimited' if limit == 0 else limit}"
        print(f"[sweep] {label} ...", file=sys.stderr)
        proc = start_engine(
            args.engine_bin, args.bind_host, args.bind_port,
            args.worker_endpoint, limit, args.mode,
            Path(args.log_dir) / f"engine_n{limit}.log",
        )
        try:
            if not wait_until_serving(target, 30):
                rows.append({"limit": limit, "error": "engine not ready"})
                continue
            s = run_load(
                target=target, requests=args.requests, concurrency=args.concurrency,
                mode=f"stream_n{limit}", max_tokens=args.max_tokens,
                timeout=args.timeout, stream=True,
            )
            row = {
                "limit": limit,
                "label": label,
                "tokens_per_second": s["tokens_per_second"],
                "wall_clock_seconds": s["wall_clock_seconds"],
                "failed": s["failed_requests"],
                "worker_ttft_p50": s.get("ttft_ms", {}).get("p50"),
                "client_ttft_p50": s.get("client_ttft_ms", {}).get("p50"),
                "client_ttft_p95": s.get("client_ttft_ms", {}).get("p95"),
                "tpot_p50": s.get("tpot_ms", {}).get("p50"),
                "queue_wait_p50": s["queue_wait_ms"]["p50"],
            }
            if args.slo_ttft_p95_ms is not None:
                p95 = row["client_ttft_p95"]
                row["meets_ttft_slo"] = p95 is not None and p95 <= args.slo_ttft_p95_ms
            rows.append(row)
            print(
                f"[sweep] {label}: {row['tokens_per_second']} tok/s, "
                f"client TTFT p50 {row['client_ttft_p50']} ms",
                file=sys.stderr,
            )
        finally:
            stop_engine(proc)
            time.sleep(1.0)

    ok = [r for r in rows if "tokens_per_second" in r]
    best_throughput = max(ok, key=lambda r: r["tokens_per_second"], default=None)
    within_slo = [r for r in ok if r.get("meets_ttft_slo")] if args.slo_ttft_p95_ms else []
    knee = max(within_slo, key=lambda r: r["tokens_per_second"], default=None)

    report = {
        "sweep": "max_concurrent_streams",
        "requests": args.requests,
        "client_concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "slo_ttft_p95_ms": args.slo_ttft_p95_ms,
        "trials": rows,
        "best_throughput": best_throughput,
        "recommended_within_slo": knee,
    }
    out = json.dumps(report, indent=2)
    print(out)
    if args.out:
        Path(args.out).write_text(out)
        print(f"[sweep] wrote {args.out}", file=sys.stderr)

    print("\n  N          tok/s    client TTFT p50    p95      TPOT", file=sys.stderr)
    for r in ok:
        print(
            "  %-10s %-8s %-18s %-8s %s"
            % (r["label"], r["tokens_per_second"], r["client_ttft_p50"],
               r["client_ttft_p95"], r["tpot_p50"]),
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
