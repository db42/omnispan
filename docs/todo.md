First goal
Build a tiny "Token Factory Perf Lab".

Scope it narrowly for learning and showcase value:

One model
One endpoint
One or two synthetic tenants at most
Minimal product surface
Multiple serving modes to compare performance behavior clearly

Primary objective:
Demonstrate cutting-edge inference serving ideas with clean measurements, not broad feature parity.

Current direction

This project is now performance-first, not feature-first.

The working architecture is:

- `engine/`: Rust edge + serving engine
- `worker/`: Python model worker with pluggable backends
- `proto/`: shared gRPC contract
- `bench/`: benchmark harness and result artifacts

The engine owns concurrency and scheduling.
The worker is currently treated as a serialized execution resource unless batching is explicit.

Important learned constraint:

- direct concurrent execution into the current Python MLX worker is not safe
- `direct` mode is debug-only
- `queued` mode is the primary safe path
- `micro_batch` is the first real performance optimization path

Current status

- Phase 1 inference prototype is done
- direct Rust engine -> Python worker path is done
- queued execution path is done
- micro-batching is done
- vLLM worker backend support is added
- current best micro-batch config is:
  - `BATCH_WINDOW_MS=10`
  - `MAX_BATCH_SIZE=4`

Current benchmark takeaway

Queued mode established explicit queue wait versus worker execution time.
Micro-batching improved throughput materially over queued mode on the current MLX setup.

Optimization roadmap

1. Queue ownership and latency decomposition
   Status: done

2. Micro-batching
   Status: done
   Notes:
   - validated that worker-side batching is faster than serial execution locally
   - validated a working engine-side micro-batch path

3. Prefix-aware serving / cache-aware routing
   Status: next
   Notes:
   - if a production inference framework already implements low-level KV reuse, we should not rebuild that primitive
   - the serving-layer value is still real:
     - prompt canonicalization
     - shared-prefix detection
     - request grouping to maximize reuse
     - cache-aware policy and observability

4. Continuous batching
   Status: later
   Notes:
   - move from request-boundary micro-batching toward in-flight scheduling

5. Speculative decoding
   Status: later
   Notes:
   - only after the serving engine is more mature

What you are building
A multi-tenant LLM inference serving system. Three layers:

Inference core. MLX locally or vLLM on GPU serving an open-weight model.
Serving layer. Your code on top. Multi-tenant routing, rate limiting, token counting, per-tenant billing hooks.
Benchmark dashboard. Visual output showing cost per token, latency, throughput. This is your demo artifact.

The third layer is what makes this a business story, not just a weekend hack.

What to use on M1 Max
MLX instead of vLLM. vLLM does not support Metal. MLX is Apple's own ML framework, purpose-built for Apple Silicon, and it is genuinely fast on your chip. Llama 3.2 3B or Mistral 7B will run well within your 32GB. The inference primitives you learn with MLX transfer directly to vLLM on CUDA. Same concepts, different backend.
Llama 3.2 3B as your primary model. Fast enough to make multi-tenant behavior interesting. Later you add a second model to make routing non-trivial.

Phased plan at 8-10 hrs/week
Phase 1: Inference core (Week 1-2)
Get a model running via MLX. Understand what is actually happening:

How tokenization works
What a forward pass costs in memory and time
What throughput looks like under load

Milestone: you can hit your local endpoint with 10 concurrent requests and watch it process them. You understand why it slows down.
Phase 2: Serving layer (Week 3-5)
This is where your systems background kicks in. Build in Python (FastAPI):

Tenant model. Each tenant has an API key, a rate limit (requests per minute, tokens per day), and a tier (free/pro).
Request router. Incoming request. Lookup tenant. Check limits. Route to model. Return response.
Token counter. Count input and output tokens per request. Accumulate per tenant.
Billing hook. Not real billing. A simple ledger: tenant X consumed Y tokens this hour at Z rupees per 1000 tokens. Log it. The math matters, not the payment integration.

Milestone: Two fake tenants hitting the system simultaneously. One exhausts its rate limit and gets a 429. The other keeps running. You can see token consumption accumulating.
Phase 3: Continuous batching (Week 6-8)
This is the technically hard part and the most important thing to understand deeply.
Naive serving handles one request at a time. Continuous batching groups in-flight requests together so the GPU (or in your case, the Neural Engine) is doing useful work on multiple requests simultaneously. This is what separates a toy system from something real. vLLM's core innovation when it launched was doing this well.
You will implement a simplified version. Not production quality. But you will understand why it matters and what the tradeoff between latency and throughput looks like.
Milestone: You can show a graph. Single request mode vs batched mode. Throughput doubles. Latency per request increases slightly. You understand the tradeoff.
Phase 4: Benchmark dashboard (Week 9-10)
A simple web UI. One page. Shows:

Tokens per second per tenant
p50 and p99 latency
Cost per 1000 tokens at current throughput (you set a rupee price, the math runs live)
A comparison row: your system vs OpenAI API pricing for equivalent output
