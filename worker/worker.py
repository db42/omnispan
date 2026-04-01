import asyncio
import logging
import os
import sys
from pathlib import Path

import grpc

from worker_runtime import MlxWorkerRuntime


CURRENT_DIR = Path(__file__).resolve().parent
GENERATED_DIR = CURRENT_DIR / "generated"
if str(GENERATED_DIR) not in sys.path:
    sys.path.insert(0, str(GENERATED_DIR))

import omnispan_pb2  # noqa: E402
import omnispan_pb2_grpc  # noqa: E402


class WorkerService(omnispan_pb2_grpc.WorkerServicer):
    def __init__(self, runtime: MlxWorkerRuntime):
        self.runtime = runtime

    async def Generate(self, request, context):
        if not request.prompt.strip():
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "prompt must be non-empty")

        try:
            result = await asyncio.to_thread(
                self.runtime.generate,
                request.request_id,
                request.tenant_id,
                request.prompt,
                request.max_tokens,
            )
        except Exception as error:  # noqa: BLE001
            logging.exception("worker generate failed")
            return omnispan_pb2.WorkerGenerateReply(
                request_id=request.request_id,
                tenant_id=request.tenant_id,
                status="error",
                error_message=str(error),
            )

        return omnispan_pb2.WorkerGenerateReply(
            request_id=result["request_id"],
            tenant_id=result["tenant_id"],
            response_text=result["response_text"],
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            worker_latency_ms=result["worker_latency_ms"],
            status="ok",
            error_message="",
        )


async def serve() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    host = os.getenv("WORKER_HOST", "127.0.0.1")
    port = os.getenv("WORKER_PORT", "50071")
    model_id = os.getenv("MODEL_ID", "mlx-community/Qwen2.5-7B-Instruct-4bit")

    runtime = MlxWorkerRuntime(model_id=model_id)
    runtime.load()

    server = grpc.aio.server()
    omnispan_pb2_grpc.add_WorkerServicer_to_server(WorkerService(runtime), server)
    bind_address = f"{host}:{port}"
    server.add_insecure_port(bind_address)

    logging.info("starting worker on %s with model %s", bind_address, model_id)
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
