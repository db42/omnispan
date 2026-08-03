# Design Note: Offline Batch Tier

**Status: design only. Not implemented.**

Omnispan currently implements the *online* serving tier: requests arrive, are
admitted, and are answered under a latency objective. This note describes the
second tier every major provider offers alongside it — an offline batch tier —
and what it would take here. It is written to record the reasoning, not to
justify building it now (see [Why not build it](#why-not-build-this-yet)).

## What the batch tier is

An asynchronous, deadline-bounded product tier: the caller submits a file of
many requests, gets a job ID, and collects results later. Turnaround is measured
in hours, not milliseconds, and it prices at roughly half the online rate
(OpenAI Batch API, Anthropic Message Batches, Bedrock batch inference).

The discount is not a different model or different hardware. It is payment for
accepting **preemption and a deadline** instead of a latency guarantee.

## What stays the same

This is the important part, and it is easy to get wrong: the batch tier is
**not** a different execution mechanism.

- **Same continuous-batching runtime.** vLLM/SGLang interleave sequences across
  decode steps identically.
- **Same admission gate.** The engine still decides how many sequences are
  in flight.
- **Still streams internally.** The runtime emits tokens per step as always;
  nobody is watching, so they are accumulated rather than forwarded.

Static/request-level batching is *not* what makes the batch tier work. That was
an MLX-era workaround (see the `micro_batch` findings in the README) and it does
not reappear here.

## What changes

Only the objective — and deleting the latency constraint unlocks optimizations
that are illegal online.

| Dimension | Online tier | Batch tier |
|---|---|---|
| Objective | max throughput **subject to** TTFT/TPOT SLO | max throughput, unconstrained |
| Admission limit `N` | tuned to protect p95 TTFT | as high as KV memory allows |
| Ordering | roughly FIFO / fair; users notice reordering | free to reorder |
| Length-aware grouping | ✗ | ✅ bucket similar-length sequences to cut straggler and padding waste |
| Preemption | ✗ (a latency spike) | ✅ evict KV to CPU/disk, resume later |
| Time-shifting | ✗ | ✅ run in demand troughs / on spot capacity |
| API shape | request → response (SSE or unary) | submit job → poll → download |
| Failure handling | fail fast, return error | retry; partial results with per-line errors |
| Metrics that matter | TTFT/TPOT p50/p95/p99 | tokens per GPU-hour, cost per 1M tokens, job completion time |

Put simply: **online is constrained optimization; batch is the same optimization
with the constraint removed.** The concurrency sweep that matters online
(`bench/sweep_concurrency.py`, finding the knee where throughput saturates before
p95 TTFT breaches) has a trivial answer in batch: as large as memory permits.

## The interesting part: co-residency

Online and batch usually run on the **same cluster**. Online demand is diurnal
and spiky, so a fleet sized for peak sits underused most of the day. Batch work
backfills those troughs.

That makes it a priority and fairshare scheduling problem, not a capacity
problem:

- Batch jobs are admitted as **low-priority, preemptible** work.
- An arriving online request **evicts** batch sequences (checkpoint their KV,
  requeue them).
- Batch consumes an **over-quota / opportunistic pool**; online holds the
  reserved pool.
- Fairness is enforced over **GPU-time**, not request counts, because sequences
  are long-lived and variable.

This is where "eliminate GPU waste in multi-tenant environments" is actually
earned, and it maps onto the mechanisms Kubernetes GPU schedulers expose today
(reservation pods, preemptible priority classes, time-based fairshare across
over-quota pools).

## What it would take in Omnispan

A new admission policy plus a job API — deliberately *not* a new engine.

1. **Job API** — `SubmitBatch(requests[]) -> job_id`, `GetBatch(job_id) -> status
   | results`. Requires a job store (even SQLite) for durability across engine
   restarts; a batch tier that loses jobs on restart is not a batch tier.
2. **Batch admission policy** — unbounded `N`, length-sorted queue, yields slots
   to online requests.
3. **Preemption** — the hard part. Needs the worker to expose "stop this
   sequence and return its state" and to resume it later. vLLM supports internal
   preemption/swapping, but not under external control through our gRPC
   contract, so this would require either extending the contract or accepting
   restart-from-scratch (wasteful but simple).
4. **Priority classes in the gate** — the existing semaphore becomes a
   two-class scheduler: online acquires ahead of batch and can force batch to
   yield.
5. **Different metrics** — tokens per GPU-hour and job completion time replace
   TTFT/TPOT, which are meaningless here.

A credible first cut is (1), (2), (4): a job API, an unbounded length-sorted
batch queue, and a two-priority gate — with preemption approximated by "batch
sequences are not started while online work is queued," rather than true
mid-flight eviction.

## Why not build this yet

- The online story is the one with measured results behind it, and it is not
  finished (admission-limit knee, multi-tenant fairness).
- Preemption done properly needs worker-runtime control Omnispan's gRPC contract
  does not expose; done improperly it is a toy that does not demonstrate the
  real tradeoff.
- The genuinely interesting content — priority, preemption, fairshare over
  GPU-time — is a *scheduling* problem that would be better demonstrated by
  adding priority classes to the existing online gate than by building a job
  store and async API around it.

If this is picked up later, start with priority classes in the streaming gate.
That produces the online/batch tension on real hardware without any of the job
plumbing.
