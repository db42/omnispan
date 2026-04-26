import asyncio
import logging
import os
import sys
from pathlib import Path

import grpc

from worker_runtime import create_runtime


CURRENT_DIR = Path(__file__).resolve().parent
GENERATED_DIR = CURRENT_DIR / "generated"
if str(GENERATED_DIR) not in sys.path:
    sys.path.insert(0, str(GENERATED_DIR))

import omnispan_pb2  # noqa: E402
import omnispan_pb2_grpc  # noqa: E402


class WorkerService(omnispan_pb2_grpc.WorkerServicer):
    def __init__(self, runtime):
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

    async def GenerateBatch(self, request, context):
        if len(request.requests) == 0:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "requests must be non-empty")

        payloads = []
        for item in request.requests:
            if not item.prompt.strip():
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "prompt must be non-empty")
            payloads.append(
                {
                    "request_id": item.request_id,
                    "tenant_id": item.tenant_id,
                    "prompt": item.prompt,
                    "max_tokens": item.max_tokens,
                }
            )

        try:
            result = await asyncio.to_thread(self.runtime.generate_batch, payloads)
        except Exception as error:  # noqa: BLE001
            logging.exception("worker batch generate failed")
            return omnispan_pb2.WorkerBatchGenerateReply(
                responses=[
                    omnispan_pb2.WorkerGenerateReply(
                        request_id=item["request_id"],
                        tenant_id=item["tenant_id"],
                        status="error",
                        error_message=str(error),
                    )
                    for item in payloads
                ],
                batch_latency_ms=0.0,
            )

        return omnispan_pb2.WorkerBatchGenerateReply(
            responses=[
                omnispan_pb2.WorkerGenerateReply(
                    request_id=item["request_id"],
                    tenant_id=item["tenant_id"],
                    response_text=item["response_text"],
                    input_tokens=item["input_tokens"],
                    output_tokens=item["output_tokens"],
                    worker_latency_ms=result["batch_latency_ms"],
                    status="ok",
                    error_message="",
                )
                for item in result["responses"]
            ],
            batch_latency_ms=result["batch_latency_ms"],
        )


async def serve() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    host = os.getenv("WORKER_HOST", "127.0.0.1")
    port = os.getenv("WORKER_PORT", "50071")
    backend = os.getenv("WORKER_BACKEND", "mlx").strip().lower()
    model_id = os.getenv("MODEL_ID")

    runtime = create_runtime(backend=backend, model_id=model_id)
    runtime.load()

    server = grpc.aio.server()
    omnispan_pb2_grpc.add_WorkerServicer_to_server(WorkerService(runtime), server)
    bind_address = f"{host}:{port}"
    bound_port = server.add_insecure_port(bind_address)
    if bound_port == 0:
        raise RuntimeError(
            f"worker failed to bind {bind_address}; another process is likely already listening"
        )

    logging.info(
        "starting worker pid=%s on %s with backend %s and model %s",
        os.getpid(),
        bind_address,
        backend,
        runtime.model_id,
    )
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
