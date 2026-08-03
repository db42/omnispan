"""vLLM runtime backed by AsyncLLMEngine.

Unlike the synchronous ``LLM`` path in ``vllm_runtime.py``, this runtime is
async-native: a single AsyncLLMEngine serves streaming, unary, and batch
requests, and vLLM continuously batches them internally at the token level. That
is what lets per-request TTFT/TPOT stay valid even while many requests share the
GPU -- each sequence still streams independently.

Selected with ``WORKER_BACKEND=vllm`` and ``VLLM_ASYNC=1``. Requires GPU + vLLM,
so it is validated on RunPod rather than in local (MLX/Apple Silicon) dev.
"""

import asyncio
import statistics
import time
from typing import Any

from worker_runtime import WorkerRuntime


class VllmAsyncWorkerRuntime(WorkerRuntime):
    is_async = True

    def __init__(
        self,
        model_id: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        trust_remote_code: bool = False,
        enforce_eager: bool = False,
        enable_prefix_caching: bool = False,
        dtype: str | None = None,
        quantization: str | None = None,
        max_num_seqs: int | None = None,
        max_num_batched_tokens: int | None = None,
    ):
        super().__init__(model_id=model_id)
        # vLLM's own continuous-batching limits. The effective batch is
        # min(engine admission limit, max_num_seqs, KV memory available), so
        # these are the inner dial that pairs with the engine's outer
        # MAX_CONCURRENT_STREAMS gate.
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.trust_remote_code = trust_remote_code
        self.enforce_eager = enforce_eager
        self.enable_prefix_caching = enable_prefix_caching
        self.dtype = dtype
        self.quantization = quantization
        self.engine = None

    def load(self) -> None:
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        engine_kwargs: dict[str, Any] = {
            "model": self.model_id,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "trust_remote_code": self.trust_remote_code,
            "enable_prefix_caching": self.enable_prefix_caching,
        }
        if self.max_model_len is not None:
            engine_kwargs["max_model_len"] = self.max_model_len
        if self.enforce_eager:
            engine_kwargs["enforce_eager"] = True
        if self.dtype:
            engine_kwargs["dtype"] = self.dtype
        if self.quantization:
            engine_kwargs["quantization"] = self.quantization
        if self.max_num_seqs is not None:
            engine_kwargs["max_num_seqs"] = self.max_num_seqs
        if self.max_num_batched_tokens is not None:
            engine_kwargs["max_num_batched_tokens"] = self.max_num_batched_tokens

        self.engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**engine_kwargs))

    # --- Sync interface is unused for this runtime; the worker awaits the async
    # methods below because is_async is True. ---
    def generate(self, request_id, tenant_id, prompt, max_tokens) -> dict:
        raise NotImplementedError("VllmAsyncWorkerRuntime is async; use agenerate")

    def generate_batch(self, requests) -> dict:
        raise NotImplementedError("VllmAsyncWorkerRuntime is async; use agenerate_batch")

    async def agenerate_stream(
        self,
        request_id: str,
        tenant_id: str,
        prompt: str,
        max_tokens: int,
    ):
        """Yield one token event per new token, then a terminal summary event.

        vLLM's async ``generate`` yields RequestOutput snapshots whose text and
        token_ids are cumulative, so we diff against what we have already emitted
        to produce per-token deltas. If the consumer stops early (client
        disconnect), the ``finally`` aborts the request so vLLM stops decoding
        for a stream nobody is reading.
        """
        if self.engine is None:
            raise RuntimeError("worker runtime is not loaded")

        from vllm import SamplingParams

        sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)

        start = time.perf_counter()
        first_token_at = None
        last_token_at = None
        inter_token_latencies_ms: list[float] = []
        emitted_text_len = 0
        token_index = 0
        input_tokens = 0
        output_tokens = 0
        aborted = True

        try:
            async for request_output in self.engine.generate(
                prompt, sampling_params, request_id
            ):
                if request_output.prompt_token_ids is not None:
                    input_tokens = len(request_output.prompt_token_ids)
                completion = request_output.outputs[0]
                cumulative_text = completion.text
                new_output_tokens = len(completion.token_ids)

                # Emit one event per newly produced token, timing each arrival.
                while output_tokens < new_output_tokens:
                    now = time.perf_counter()
                    if first_token_at is None:
                        first_token_at = now
                    else:
                        inter_token_latencies_ms.append((now - last_token_at) * 1000)
                    last_token_at = now
                    output_tokens += 1

                delta = cumulative_text[emitted_text_len:]
                if delta:
                    emitted_text_len = len(cumulative_text)
                    yield {
                        "type": "token",
                        "text_delta": delta,
                        "token_index": token_index,
                    }
                    token_index += 1

            aborted = False
            worker_latency_ms = (time.perf_counter() - start) * 1000
            ttft_ms = ((first_token_at - start) * 1000) if first_token_at is not None else 0.0
            tpot_ms = (
                statistics.fmean(inter_token_latencies_ms)
                if inter_token_latencies_ms
                else 0.0
            )
            yield {
                "type": "final",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "worker_latency_ms": round(worker_latency_ms, 2),
                "ttft_ms": round(ttft_ms, 2),
                "tpot_ms": round(tpot_ms, 2),
            }
        finally:
            if aborted:
                await self.engine.abort(request_id)

    async def agenerate(
        self,
        request_id: str,
        tenant_id: str,
        prompt: str,
        max_tokens: int,
    ) -> dict:
        chunks: list[str] = []
        final: dict = {}
        async for event in self.agenerate_stream(request_id, tenant_id, prompt, max_tokens):
            if event["type"] == "token":
                chunks.append(event["text_delta"])
            else:
                final = event

        return {
            "request_id": request_id,
            "tenant_id": tenant_id,
            "response_text": "".join(chunks),
            "input_tokens": final["input_tokens"],
            "output_tokens": final["output_tokens"],
            "worker_latency_ms": final["worker_latency_ms"],
            "ttft_ms": final["ttft_ms"],
            "tpot_ms": final["tpot_ms"],
        }

    async def agenerate_batch(self, requests: list[dict]) -> dict:
        """Run all requests concurrently through the one engine.

        This is not static batching: each request is an independent async
        generation, and vLLM's scheduler continuously batches them at the token
        level. Because each still streams internally, TTFT/TPOT stay valid
        per request -- the opposite of the synchronous MLX batch path.
        """
        started_at = time.perf_counter()
        results = await asyncio.gather(
            *(
                self.agenerate(
                    request["request_id"],
                    request["tenant_id"],
                    request["prompt"],
                    request["max_tokens"],
                )
                for request in requests
            )
        )
        batch_latency_ms = (time.perf_counter() - started_at) * 1000
        return {
            "responses": results,
            "batch_latency_ms": round(batch_latency_ms, 2),
        }
