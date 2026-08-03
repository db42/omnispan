import os
from abc import ABC, abstractmethod


DEFAULT_MLX_MODEL_ID = "mlx-community/Qwen2.5-7B-Instruct-4bit"
DEFAULT_VLLM_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


class WorkerRuntime(ABC):
    # Async-native runtimes (e.g. the vLLM AsyncLLMEngine path) set this True and
    # implement the ``a*`` coroutine methods; the worker awaits those directly
    # instead of offloading the sync methods to a thread.
    is_async = False

    def __init__(self, model_id: str):
        self.model_id = model_id

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        request_id: str,
        tenant_id: str,
        prompt: str,
        max_tokens: int,
    ) -> dict:
        raise NotImplementedError

    @abstractmethod
    def generate_batch(self, requests: list[dict]) -> dict:
        raise NotImplementedError

    def generate_stream(
        self,
        request_id: str,
        tenant_id: str,
        prompt: str,
        max_tokens: int,
    ):
        """Yield per-token events then a terminal summary event.

        Optional capability. Backends that cannot stream (or whose streaming is
        not yet wired up) raise NotImplementedError; the worker surfaces that as
        an UNIMPLEMENTED gRPC status. The vLLM path will implement this via
        AsyncLLMEngine on GPU hardware.
        """
        raise NotImplementedError("streaming is not supported by this backend")


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_optional_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return int(value)


def create_runtime(backend: str, model_id: str | None = None) -> WorkerRuntime:
    normalized_backend = backend.strip().lower()

    if normalized_backend == "mlx":
        from mlx_runtime import MlxWorkerRuntime

        return MlxWorkerRuntime(model_id or DEFAULT_MLX_MODEL_ID)

    if normalized_backend == "vllm":
        common_kwargs = dict(
            model_id=model_id or DEFAULT_VLLM_MODEL_ID,
            tensor_parallel_size=int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1")),
            gpu_memory_utilization=float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.9")),
            max_model_len=_env_optional_int("VLLM_MAX_MODEL_LEN"),
            trust_remote_code=_env_flag("VLLM_TRUST_REMOTE_CODE", default=False),
            enforce_eager=_env_flag("VLLM_ENFORCE_EAGER", default=False),
            enable_prefix_caching=_env_flag("VLLM_ENABLE_PREFIX_CACHING", default=False),
            dtype=os.getenv("VLLM_DTYPE"),
            quantization=os.getenv("VLLM_QUANTIZATION"),
        )

        # VLLM_ASYNC=1 selects the AsyncLLMEngine-backed runtime, which adds
        # streaming (per-request token output with real TTFT/TPOT) on top of
        # vLLM's native continuous batching. The default keeps the synchronous
        # LLM path, which has no streaming.
        if _env_flag("VLLM_ASYNC", default=False):
            from vllm_async_runtime import VllmAsyncWorkerRuntime

            return VllmAsyncWorkerRuntime(**common_kwargs)

        from vllm_runtime import VllmWorkerRuntime

        return VllmWorkerRuntime(
            debug_batch_logging=_env_flag("WORKER_DEBUG_BATCH_LOGGING", default=False),
            **common_kwargs,
        )

    raise ValueError(
        f"unsupported WORKER_BACKEND={backend!r}; expected one of: mlx, vllm"
    )
