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

    def generate(
        self,
        request_id: str,
        tenant_id: str,
        prompt: str,
        max_tokens: int,
    ) -> dict:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("worker runtime is not loaded")

        import mlx_lm

        input_tokens = len(self.tokenizer.encode(prompt))

        start = time.perf_counter()
        response_text = mlx_lm.generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )
        worker_latency_ms = (time.perf_counter() - start) * 1000
        output_tokens = len(self.tokenizer.encode(response_text))

        return {
            "request_id": request_id,
            "tenant_id": tenant_id,
            "response_text": response_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "worker_latency_ms": round(worker_latency_ms, 2),
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
            results.append(
                {
                    "request_id": request["request_id"],
                    "tenant_id": request["tenant_id"],
                    "response_text": response_text,
                    "input_tokens": len(prompt_tokens),
                    "output_tokens": len(self.tokenizer.encode(response_text)),
                }
            )

        return {
            "responses": results,
            "batch_latency_ms": round(worker_latency_ms, 2),
        }
