#!/usr/bin/env python
"""Probe the installed vLLM's API surface for the pieces Omnispan depends on.

vLLM's engine internals move between releases (notably the V0 -> V1 transition),
and the breakages are quiet: metrics fields read via getattr(..., None) degrade
to 0.0 instead of raising, so TTFT/TPOT silently become zero. Run this after
installing or upgrading vLLM, before trusting a benchmark run.

    python scripts/probe_vllm_api.py

Exits non-zero if a required attribute is missing.
"""

import dataclasses
import inspect
import sys


def main() -> int:
    try:
        import vllm
        from vllm import AsyncEngineArgs, AsyncLLMEngine, RequestOutput
        from vllm.outputs import CompletionOutput
    except Exception as error:  # noqa: BLE001
        print(f"FAIL: cannot import vllm: {type(error).__name__}: {error}")
        return 1

    print(f"vllm {vllm.__version__}")
    problems: list[str] = []

    # --- AsyncEngineArgs fields we pass in vllm_async_runtime.load() ---
    arg_fields = {f.name for f in dataclasses.fields(AsyncEngineArgs)}
    required_args = [
        "model",
        "tensor_parallel_size",
        "gpu_memory_utilization",
        "trust_remote_code",
        "enable_prefix_caching",
        "max_model_len",
        "enforce_eager",
        "dtype",
        "quantization",
    ]
    print("\nAsyncEngineArgs:")
    for name in required_args:
        ok = name in arg_fields
        print(f"  {'ok ' if ok else 'MISSING'} {name}")
        if not ok:
            problems.append(f"AsyncEngineArgs.{name} missing")

    # Removed in vLLM 0.26 (V1). Passing it raises TypeError at load.
    for name in ["disable_log_requests"]:
        if name in arg_fields:
            print(f"  note: {name} still accepted")
        else:
            print(f"  note: {name} NOT accepted (do not pass it)")

    # --- generate() call shape ---
    gen_params = list(inspect.signature(AsyncLLMEngine.generate).parameters)
    print(f"\nAsyncLLMEngine.generate params: {gen_params[:4]} ...")
    for name in ["prompt", "sampling_params", "request_id"]:
        if name not in gen_params:
            problems.append(f"AsyncLLMEngine.generate missing '{name}'")
            print(f"  MISSING {name}")

    # --- streaming output shape ---
    out_params = set(inspect.signature(RequestOutput.__init__).parameters)
    comp_fields = (
        {f.name for f in dataclasses.fields(CompletionOutput)}
        if dataclasses.is_dataclass(CompletionOutput)
        else set(inspect.signature(CompletionOutput.__init__).parameters)
    )
    print("\nOutput shape:")
    for label, present in [
        ("RequestOutput.prompt_token_ids", "prompt_token_ids" in out_params),
        ("RequestOutput.outputs", "outputs" in out_params),
        ("RequestOutput.metrics", "metrics" in out_params),
        ("CompletionOutput.text", "text" in comp_fields),
        ("CompletionOutput.token_ids", "token_ids" in comp_fields),
    ]:
        print(f"  {'ok ' if present else 'MISSING'} {label}")
        if not present and not label.endswith(".metrics"):
            problems.append(f"{label} missing")

    # --- metrics type: the sync runtime reads these for TTFT/TPOT ---
    print("\nMetrics (used by the sync vLLM runtime for TTFT/TPOT):")
    try:
        from vllm import RequestMetrics  # noqa: F401

        print("  ok  vllm.RequestMetrics importable (V0-style)")
    except Exception:
        print("  note: vllm.RequestMetrics NOT importable (V1)")
        try:
            from vllm.v1.metrics.stats import RequestStateStats

            fields = [a for a in dir(RequestStateStats) if not a.startswith("_")]
            print(f"  V1 RequestStateStats attrs: {fields}")
            for name in ["first_token_time", "first_scheduled_time", "finished_time"]:
                if name not in fields:
                    print(
                        f"  WARN {name} absent -> sync-path TTFT/TPOT will read 0.0"
                    )
        except Exception as error:  # noqa: BLE001
            print(f"  WARN cannot inspect V1 metrics: {type(error).__name__}")

    print()
    if problems:
        print(f"FAIL: {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("OK: required vLLM API surface present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
