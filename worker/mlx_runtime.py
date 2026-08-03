import statistics
import time

from worker_runtime import WorkerRuntime


class MlxWorkerRuntime(WorkerRuntime):
    def __init__(self, model_id: str):
        super().__init__(model_id=model_id)
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        import mlx_lm

        self.model, self.tokenizer = mlx_lm.load(self.model_id)

    def generate_stream(
        self,
        request_id: str,
        tenant_id: str,
        prompt: str,
        max_tokens: int,
    ):
        """Yield one event per output token, then a terminal summary event.

        Each token event is {"type": "token", "text_delta", "token_index"}.
        The final event is {"type": "final", ...} carrying the aggregate metrics.
        This is the single source of truth for MLX decoding; the unary
        ``generate`` below simply drains this generator.

        Streaming token-by-token is what makes TTFT (time to first token, the
        prefill cost the client feels) and TPOT (mean inter-token latency during
        decode) observable -- neither is recoverable from a single blocking call.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("worker runtime is not loaded")

        import mlx_lm

        input_tokens = len(self.tokenizer.encode(prompt))

        start = time.perf_counter()
        first_token_at = None
        last_token_at = None
        inter_token_latencies_ms: list[float] = []
        output_tokens = 0
        token_index = 0

        for response in mlx_lm.stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
        ):
            now = time.perf_counter()
            if first_token_at is None:
                first_token_at = now
            else:
                inter_token_latencies_ms.append((now - last_token_at) * 1000)
            last_token_at = now
            output_tokens = response.generation_tokens
            yield {
                "type": "token",
                "text_delta": response.text,
                "token_index": token_index,
            }
            token_index += 1

        worker_latency_ms = (time.perf_counter() - start) * 1000
        ttft_ms = ((first_token_at - start) * 1000) if first_token_at is not None else 0.0
        tpot_ms = statistics.fmean(inter_token_latencies_ms) if inter_token_latencies_ms else 0.0

        yield {
            "type": "final",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "worker_latency_ms": round(worker_latency_ms, 2),
            "ttft_ms": round(ttft_ms, 2),
            "tpot_ms": round(tpot_ms, 2),
        }

    def generate(
        self,
        request_id: str,
        tenant_id: str,
        prompt: str,
        max_tokens: int,
    ) -> dict:
        chunks: list[str] = []
        final: dict = {}
        for event in self.generate_stream(request_id, tenant_id, prompt, max_tokens):
            if event["type"] == "token":
                chunks.append(event["text_delta"])
            else:
                final = event

        # TTFT/TPOT are serving metrics tied to a token stream the client
        # actually receives. The unary path returns the whole response at once,
        # so the client never experiences an early first token -- reporting a
        # first-token time here would describe an internal decode property, not a
        # latency anyone observed. We leave both unmeasured (0.0) and expose
        # TTFT/TPOT only on the streaming path (generate_stream). Judge the unary
        # path by throughput and total latency instead.
        return {
            "request_id": request_id,
            "tenant_id": tenant_id,
            "response_text": "".join(chunks),
            "input_tokens": final["input_tokens"],
            "output_tokens": final["output_tokens"],
            "worker_latency_ms": final["worker_latency_ms"],
            "ttft_ms": 0.0,
            "tpot_ms": 0.0,
        }

    def generate_batch(self, requests: list[dict]) -> dict:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("worker runtime is not loaded")

        import mlx_lm

        prompts = [request["prompt"] for request in requests]
        prompt_token_ids = [self.tokenizer.encode(prompt) for prompt in prompts]
        max_tokens = [request["max_tokens"] for request in requests]

        start = time.perf_counter()
        batch_response = mlx_lm.batch_generate(
            self.model,
            self.tokenizer,
            prompts=prompt_token_ids,
            max_tokens=max_tokens,
            verbose=False,
        )
        worker_latency_ms = (time.perf_counter() - start) * 1000

        results = []
        for request, response_text, prompt_tokens in zip(
            requests, batch_response.texts, prompt_token_ids, strict=True
        ):
            output_tokens = len(self.tokenizer.encode(response_text))
            # TTFT and TPOT are per-request online-serving metrics: they need a
            # first-token event and per-sequence inter-token timing. A synchronous
            # batch (mlx_lm.batch_generate) has neither -- it blocks and returns
            # only final texts. So both are left unmeasured (0.0) here; judge the
            # batch path by aggregate throughput and batch latency instead.
            # (Continuous-batching engines like vLLM restore per-request TTFT/TPOT
            # because each sequence still streams independently.)
            results.append(
                {
                    "request_id": request["request_id"],
                    "tenant_id": request["tenant_id"],
                    "response_text": response_text,
                    "input_tokens": len(prompt_tokens),
                    "output_tokens": output_tokens,
                    "ttft_ms": 0.0,
                    "tpot_ms": 0.0,
                }
            )

        return {
            "responses": results,
            "batch_latency_ms": round(worker_latency_ms, 2),
        }
