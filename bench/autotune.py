"""SLO-driven auto-tuner for the Omnispan serving engine.

Sweeps a grid of engine scheduling configurations (queued vs micro-batch,
batch window, max batch size), runs a short load test against each, and reports
the configuration that maximizes throughput while still meeting a latency SLO
(e.g. TTFT p95 and TPOT p95 targets).

The worker is assumed to already be running. This script owns the engine
lifecycle: for each candidate config it launches the engine binary with the
matching environment, waits until it is serving, drives load through the shared
`benchmark.run_load` generator, then tears the engine down before the next run.

Example:
    python bench/autotune.py \
      --engine-bin engine/target/debug/omnispan-engine \
      --worker-endpoint http://127.0.0.1:50071 \
      --requests 12 --concurrency 6 --max-tokens 64 \
      --windows 5,10,20 --batch-sizes 1,2,4 \
      --slo-ttft-p95-ms 1500 --slo-tpot-p95-ms 40
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


def build_grid(windows: list[int], batch_sizes: list[int]) -> list[dict]:
    """Expand the sweep into concrete engine configs.

    Batch size 1 has no batching to do, so it maps to `queued` mode and ignores
    the window. Batch sizes > 1 map to `micro_batch` mode and are swept across
    every window value.
    """
    configs: list[dict] = []
    seen: set[tuple] = set()
    for batch in batch_sizes:
        if batch <= 1:
            key = ("queued", 0, 1)
            if key not in seen:
                seen.add(key)
                configs.append({"mode": "queued", "batch_window_ms": 0, "max_batch_size": 1})
            continue
        for window in windows:
            key = ("micro_batch", window, batch)
            if key not in seen:
                seen.add(key)
                configs.append(
                    {"mode": "micro_batch", "batch_window_ms": window, "max_batch_size": batch}
                )
    return configs


def config_label(config: dict) -> str:
    if config["mode"] == "queued":
        return "queued"
    return f"micro_batch_w{config['batch_window_ms']}_b{config['max_batch_size']}"


def wait_until_serving(target: str, timeout_s: float) -> bool:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        try:
            with grpc.insecure_channel(target) as channel:
                stub = omnispan_pb2_grpc.EngineStub(channel)
                stub.GetEngineStats(omnispan_pb2.StatsRequest(), timeout=1.0)
                return True
        except grpc.RpcError:
            time.sleep(0.2)
    return False


def start_engine(
    engine_bin: str,
    bind_host: str,
    bind_port: int,
    worker_endpoint: str,
    config: dict,
    log_path: Path,
) -> subprocess.Popen:
    env = os.environ.copy()
    env.update(
        {
            "ENGINE_MODE": config["mode"],
            "BIND_HOST": bind_host,
            "BIND_PORT": str(bind_port),
            "WORKER_ENDPOINT": worker_endpoint,
            "BATCH_WINDOW_MS": str(config["batch_window_ms"]),
            "MAX_BATCH_SIZE": str(config["max_batch_size"]),
        }
    )
    log_file = open(log_path, "w")
    # New process group so we can signal the whole engine tree on teardown.
    return subprocess.Popen(
        [engine_bin],
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def stop_engine(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait(timeout=5)


def meets_slo(summary: dict, slos: dict[str, float]) -> tuple[bool, list[str]]:
    """Return (passed, violations). A config must succeed on every request and
    stay within every provided SLO threshold."""
    violations: list[str] = []
    if summary["failed_requests"] > 0:
        violations.append(f"{summary['failed_requests']} failed requests")

    checks = {
        "ttft_p95_ms": ("ttft_ms", "p95"),
        "tpot_p95_ms": ("tpot_ms", "p95"),
        "client_p95_ms": ("client_latency_ms", "p95"),
    }
    for slo_key, (block, stat) in checks.items():
        threshold = slos.get(slo_key)
        if threshold is None:
            continue
        block_summary = summary.get(block)
        # For metrics that can be unmeasured under some configs (TTFT/TPOT are
        # not observable on the synchronous batch path, where the summary omits
        # the block entirely), an SLO cannot be verified without samples. Treat
        # "unmeasured" as a violation rather than a silent pass, so the tuner
        # never recommends a config on the strength of a metric it could not
        # actually measure.
        if not block_summary or block_summary.get("measured_samples", 0) == 0:
            violations.append(f"{block}.{stat} SLO set but {block} was not measured under this config")
            continue
        actual = block_summary.get(stat, 0.0)
        if actual > threshold:
            violations.append(f"{block}.{stat}={actual} > {threshold}")
    return (len(violations) == 0, violations)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-bin", default="engine/target/debug/omnispan-engine")
    parser.add_argument("--bind-host", default="127.0.0.1")
    parser.add_argument("--bind-port", type=int, default=50061)
    parser.add_argument("--worker-endpoint", default="http://127.0.0.1:50071")
    parser.add_argument("--requests", type=int, default=12)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--tenant-count", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--windows", default="5,10,20", help="Comma-separated BATCH_WINDOW_MS values.")
    parser.add_argument("--batch-sizes", default="1,2,4", help="Comma-separated MAX_BATCH_SIZE values.")
    parser.add_argument("--slo-ttft-p95-ms", type=float, default=None)
    parser.add_argument("--slo-tpot-p95-ms", type=float, default=None)
    parser.add_argument("--slo-client-p95-ms", type=float, default=None)
    parser.add_argument(
        "--stream",
        action="store_true",
        help=(
            "Drive load over the streaming endpoint (required to measure TTFT/TPOT). "
            "Configs that cannot stream (micro_batch) are skipped."
        ),
    )
    parser.add_argument("--startup-timeout-s", type=float, default=30.0)
    parser.add_argument("--log-dir", default=None, help="Directory for per-config engine logs.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    windows = [int(v) for v in args.windows.split(",") if v.strip()]
    batch_sizes = [int(v) for v in args.batch_sizes.split(",") if v.strip()]
    grid = build_grid(windows, batch_sizes)

    if args.stream:
        # micro_batch cannot stream per request (the engine returns UNIMPLEMENTED),
        # so a streaming sweep can only evaluate stream-capable configs.
        streamable = [c for c in grid if c["mode"] != "micro_batch"]
        skipped = len(grid) - len(streamable)
        if skipped:
            print(
                f"[autotune] --stream: skipping {skipped} micro_batch config(s) that cannot stream",
                file=sys.stderr,
            )
        grid = streamable

    slos = {
        "ttft_p95_ms": args.slo_ttft_p95_ms,
        "tpot_p95_ms": args.slo_tpot_p95_ms,
        "client_p95_ms": args.slo_client_p95_ms,
    }
    slos = {k: v for k, v in slos.items() if v is not None}

    log_dir = Path(args.log_dir) if args.log_dir else Path(ROOT_DIR / "bench" / "autotune_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    target = f"{args.bind_host}:{args.bind_port}"
    trials: list[dict] = []

    for config in grid:
        label = config_label(config)
        print(f"[autotune] trying {label} ...", file=sys.stderr)
        proc = start_engine(
            engine_bin=args.engine_bin,
            bind_host=args.bind_host,
            bind_port=args.bind_port,
            worker_endpoint=args.worker_endpoint,
            config=config,
            log_path=log_dir / f"engine_{label}.log",
        )
        try:
            if not wait_until_serving(target, args.startup_timeout_s):
                trials.append({"config": label, "error": "engine did not become ready"})
                continue

            summary = run_load(
                target=target,
                requests=args.requests,
                concurrency=args.concurrency,
                mode=label,
                tenant_count=args.tenant_count,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                stream=args.stream,
            )
            passed, violations = meets_slo(summary, slos)
            trials.append(
                {
                    "config": label,
                    "mode": config["mode"],
                    "batch_window_ms": config["batch_window_ms"],
                    "max_batch_size": config["max_batch_size"],
                    "tokens_per_second": summary["tokens_per_second"],
                    # None when unmeasured (synchronous batch omits these blocks).
                    "ttft_p95_ms": summary.get("ttft_ms", {}).get("p95"),
                    "tpot_p95_ms": summary.get("tpot_ms", {}).get("p95"),
                    "client_p95_ms": summary["client_latency_ms"]["p95"],
                    "failed_requests": summary["failed_requests"],
                    "meets_slo": passed,
                    "slo_violations": violations,
                }
            )
        finally:
            stop_engine(proc)
            # Give the OS a moment to release the port before the next launch.
            time.sleep(0.5)

    passing = [t for t in trials if t.get("meets_slo")]
    winner = max(passing, key=lambda t: t["tokens_per_second"], default=None)

    report = {
        "slos": slos,
        "grid_size": len(grid),
        "trials": sorted(
            [t for t in trials if "tokens_per_second" in t],
            key=lambda t: t["tokens_per_second"],
            reverse=True,
        ),
        "errors": [t for t in trials if "error" in t],
        "recommended_config": winner,
    }
    print(json.dumps(report, indent=2))

    if winner is None:
        print("[autotune] no configuration met the SLO", file=sys.stderr)
    else:
        print(
            f"[autotune] recommended: {winner['config']} "
            f"@ {winner['tokens_per_second']} tok/s",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
