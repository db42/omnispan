# Benchmarks

## Runpod A40: Qwen3-32B-AWQ

### Prefix-cache-sensitive workload

Configuration:

- backend: `vllm`
- model: `Qwen/Qwen3-32B-AWQ`
- worker GPU: single A40
- engine mode: `micro_batch`
- requests: `2`
- concurrency: `2`
- max tokens: `64`
- shared prefix repeats: `6`
- suffix template: `Explain the top vendor risks for company {index} in 3 bullets.`

Artifacts:

- APC on:
  - [bench/runpod/qwen3_32b_awq_apc_on_2x2_r6_t64.json](/Users/dushyant.bansal/work/omnispan/bench/runpod/qwen3_32b_awq_apc_on_2x2_r6_t64.json)
- APC off:
  - [bench/runpod/qwen3_32b_awq_apc_off_2x2_r6_t64.json](/Users/dushyant.bansal/work/omnispan/bench/runpod/qwen3_32b_awq_apc_off_2x2_r6_t64.json)

Observed result:

- APC on:
  - wall clock `2.87s`
  - worker latency p50 `2820.85 ms`
  - throughput `556.71 tokens/s`

- APC off:
  - wall clock `4.21s`
  - worker latency p50 `4149.71 ms`
  - throughput `379.66 tokens/s`

Interpretation:

- On this long shared-prefix workload, vLLM automatic prefix caching improved throughput by about `1.47x`.
- Queue wait stayed effectively unchanged at around `11 ms`, so the improvement came from the worker execution path rather than the Rust engine.
- This is a useful proof point for the orchestration-layer thesis: prefix locality can materially matter when the workload shape is favorable.
