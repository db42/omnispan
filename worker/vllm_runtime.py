import time
from typing import Any

from worker_runtime import WorkerRuntime


class VllmWorkerRuntime(WorkerRuntime):
    def __init__(
        self,
        model_id: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        trust_remote_code: bool = False,
        enforce_eager: bool = False,
        dtype: str | None = None,
    ):
        super().__init__(model_id=model_id)
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.trust_remote_code = trust_remote_code
        self.enforce_eager = enforce_eager
        self.dtype = dtype
        self.llm = None
        self.tokenizer = None

    def load(self) -> None:
        from vllm import LLM
        from vllm.tokenizers import get_tokenizer

        llm_kwargs: dict[str, Any] = {
            "model": self.model_id,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.max_model_len is not None:
            llm_kwargs["max_model_len"] = self.max_model_len
        if self.enforce_eager:
            llm_kwargs["enforce_eager"] = True
        if self.dtype:
            llm_kwargs["dtype"] = self.dtype

        self.llm = LLM(**llm_kwargs)
        self.tokenizer = get_tokenizer(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
        )

    def generate(
        self,
        request_id: str,
        tenant_id: str,
        prompt: str,
        max_tokens: int,
    ) -> dict:
        outputs = self._generate_outputs([prompt], [max_tokens])
        output = outputs[0]
        completion = self._first_completion(output)

        return {
            "request_id": request_id,
            "tenant_id": tenant_id,
            "response_text": completion.text,
            "input_tokens": self._prompt_token_count(output, prompt),
            "output_tokens": len(completion.token_ids),
            "worker_latency_ms": round(self._worker_latency_ms(output), 2),
        }

    def generate_batch(self, requests: list[dict]) -> dict:
        prompts = [request["prompt"] for request in requests]
        max_tokens = [request["max_tokens"] for request in requests]

        started_at = time.perf_counter()
        outputs = self._generate_outputs(prompts, max_tokens)
        batch_latency_ms = (time.perf_counter() - started_at) * 1000

        results = []
        for request, output in zip(requests, outputs, strict=True):
            completion = self._first_completion(output)
            results.append(
                {
                    "request_id": request["request_id"],
                    "tenant_id": request["tenant_id"],
                    "response_text": completion.text,
                    "input_tokens": self._prompt_token_count(output, request["prompt"]),
                    "output_tokens": len(completion.token_ids),
                }
            )

        return {
            "responses": results,
            "batch_latency_ms": round(batch_latency_ms, 2),
        }

    def _generate_outputs(self, prompts: list[str], max_tokens: list[int]) -> list[Any]:
        if self.llm is None:
            raise RuntimeError("worker runtime is not loaded")

        from vllm import SamplingParams

        sampling_params = [
            SamplingParams(max_tokens=token_count, temperature=0.0)
            for token_count in max_tokens
        ]
        return self.llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)

    def _first_completion(self, output: Any) -> Any:
        if not output.outputs:
            raise RuntimeError("vLLM returned no completion output")
        return output.outputs[0]

    def _prompt_token_count(self, output: Any, prompt: str) -> int:
        prompt_token_ids = getattr(output, "prompt_token_ids", None)
        if prompt_token_ids is not None:
            return len(prompt_token_ids)
        if self.tokenizer is None:
            raise RuntimeError("vLLM tokenizer is not loaded")
        return len(self.tokenizer.encode(prompt))

    def _worker_latency_ms(self, output: Any) -> float:
        metrics = getattr(output, "metrics", None)
        if metrics is None:
            return 0.0

        finished = getattr(metrics, "finished_time", None)
        first_scheduled = getattr(metrics, "first_scheduled_time", None)
        arrival = getattr(metrics, "arrival_time", None)

        if finished is not None and first_scheduled is not None:
            return (finished - first_scheduled) * 1000
        if finished is not None and arrival is not None:
            return (finished - arrival) * 1000
        return 0.0
