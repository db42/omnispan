use std::pin::Pin;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use tokio::sync::{mpsc, oneshot, Mutex, Semaphore};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::{Stream, StreamExt};
use tonic::{Request, Response, Status};

use crate::omnispan::engine_server::Engine;
use crate::omnispan::{
    GenerateChunk, GenerateReply, GenerateRequest, StatsReply, StatsRequest, WorkerGenerateReply,
    WorkerGenerateRequest,
};
use crate::queue::QueuedRequest;
use crate::worker_client::DirectWorkerClient;

type GenerateChunkStream = Pin<Box<dyn Stream<Item = Result<GenerateChunk, Status>> + Send>>;

#[derive(Debug, Default)]
pub struct EngineState {
    pub accepted_requests: u64,
    pub queue_depth: u64,
}

#[derive(Clone)]
pub struct EngineService {
    mode: String,
    started_at: Instant,
    pub(crate) state: Arc<Mutex<EngineState>>,
    worker_client: DirectWorkerClient,
    queue_tx: Option<mpsc::Sender<QueuedRequest>>,
    // Serializes streaming requests against the single worker in queued mode
    // (None in direct mode, where concurrency is the caller's responsibility).
    stream_gate: Option<Arc<Semaphore>>,
}

impl EngineService {
    pub fn new(
        mode: String,
        worker_client: DirectWorkerClient,
        queue_tx: Option<mpsc::Sender<QueuedRequest>>,
    ) -> Self {
        // Streaming does not flow through the micro-batch scheduler, so in
        // queued mode we serialize streaming requests with a 1-permit gate to
        // preserve one-request-at-a-time worker access.
        let stream_gate = if mode == "queued" {
            Some(Arc::new(Semaphore::new(1)))
        } else {
            None
        };
        Self {
            mode,
            started_at: Instant::now(),
            state: Arc::new(Mutex::new(EngineState::default())),
            worker_client,
            queue_tx,
            stream_gate,
        }
    }

    pub fn shared_state(&self) -> Arc<Mutex<EngineState>> {
        Arc::clone(&self.state)
    }
}

#[tonic::async_trait]
impl Engine for EngineService {
    async fn submit_generate(
        &self,
        request: Request<GenerateRequest>,
    ) -> Result<Response<GenerateReply>, Status> {
        let started_at = Instant::now();
        let mut inner = request.into_inner();
        if inner.prompt.trim().is_empty() {
            return Err(Status::invalid_argument("prompt must be non-empty"));
        }
        if inner.request_id.trim().is_empty() {
            inner.request_id = new_request_id();
        }

        let mut state = self.state.lock().await;
        state.accepted_requests += 1;
        drop(state);

        match self.mode.as_str() {
            "direct" => {
                // Debug-only path. Concurrent direct requests have triggered native crashes in the
                // Python MLX worker, so the performance lab should treat queued mode as the real
                // execution path until the worker runtime is proven safe for parallel access.
                let reply = execute_with_worker(
                    &self.worker_client,
                    inner.request_id,
                    inner.tenant_id,
                    inner.prompt,
                    inner.max_tokens,
                    started_at,
                    started_at,
                )
                .await;

                Ok(Response::new(reply))
            }
            "queued" | "micro_batch" => {
                let queue_tx = self
                    .queue_tx
                    .as_ref()
                    .ok_or_else(|| Status::failed_precondition("queue sender is not configured"))?;
                let (reply_tx, reply_rx) = oneshot::channel();

                {
                    let mut state = self.state.lock().await;
                    state.queue_depth += 1;
                }

                if queue_tx
                    .send(QueuedRequest {
                        request_id: inner.request_id,
                        tenant_id: inner.tenant_id,
                        prompt: inner.prompt,
                        max_tokens: inner.max_tokens,
                        received_at: started_at,
                        reply_tx,
                    })
                    .await
                    .is_err()
                {
                    let mut state = self.state.lock().await;
                    state.queue_depth = state.queue_depth.saturating_sub(1);
                    return Err(Status::unavailable("queue send failed"));
                }

                let reply = reply_rx
                    .await
                    .map_err(|_| Status::unavailable("queue reply dropped"))?;

                Ok(Response::new(reply))
            }
            _ => Err(Status::failed_precondition(format!(
                "mode {} is not implemented yet",
                self.mode
            ))),
        }
    }

    type SubmitGenerateStreamStream = GenerateChunkStream;

    async fn submit_generate_stream(
        &self,
        request: Request<GenerateRequest>,
    ) -> Result<Response<Self::SubmitGenerateStreamStream>, Status> {
        let received_at = Instant::now();
        let mut inner = request.into_inner();
        if inner.prompt.trim().is_empty() {
            return Err(Status::invalid_argument("prompt must be non-empty"));
        }
        if inner.request_id.trim().is_empty() {
            inner.request_id = new_request_id();
        }
        if self.mode == "micro_batch" {
            return Err(Status::unimplemented(
                "streaming is not supported in micro_batch mode; use direct or queued",
            ));
        }

        {
            let mut state = self.state.lock().await;
            state.accepted_requests += 1;
        }

        let (tx, rx) = mpsc::channel::<Result<GenerateChunk, Status>>(64);
        let worker = self.worker_client.clone();
        let gate = self.stream_gate.clone();
        let request_id = inner.request_id;
        let tenant_id = inner.tenant_id;
        let prompt = inner.prompt;
        let max_tokens = inner.max_tokens;

        let state = Arc::clone(&self.state);
        tokio::spawn(async move {
            // A stream waiting on the gate is real backlog, so it counts toward
            // queue_depth just like a queued unary request.
            if gate.is_some() {
                let mut guard = state.lock().await;
                guard.queue_depth += 1;
            }
            // In queued mode this blocks until the single worker is free.
            let _permit = match &gate {
                Some(semaphore) => Some(semaphore.acquire().await.expect("stream gate closed")),
                None => None,
            };
            if gate.is_some() {
                let mut guard = state.lock().await;
                guard.queue_depth = guard.queue_depth.saturating_sub(1);
            }
            // Gate wait is the streaming analogue of queue wait: time from
            // intake to dispatch. Measured explicitly so it is not left hiding
            // inside the client-observed TTFT.
            let gate_wait_ms = received_at.elapsed().as_secs_f64() * 1000.0;
            let scheduled_at = Instant::now();

            let worker_request = WorkerGenerateRequest {
                request_id: request_id.clone(),
                tenant_id: tenant_id.clone(),
                prompt,
                max_tokens,
                submitted_at_ms: now_unix_ms(),
            };

            match worker.generate_stream(worker_request).await {
                Ok(mut stream) => {
                    let mut engine_ttft_ms = 0.0;
                    let mut seen_token = false;
                    while let Some(item) = stream.next().await {
                        match item {
                            Ok(chunk) if !chunk.finished => {
                                if !seen_token {
                                    // Engine-observed TTFT: time from dispatch to
                                    // the first token forwarded onward.
                                    engine_ttft_ms = scheduled_at.elapsed().as_secs_f64() * 1000.0;
                                    seen_token = true;
                                }
                                let forward = GenerateChunk {
                                    request_id: request_id.clone(),
                                    tenant_id: tenant_id.clone(),
                                    text_delta: chunk.text_delta,
                                    token_index: chunk.token_index,
                                    finished: false,
                                    ..Default::default()
                                };
                                if tx.send(Ok(forward)).await.is_err() {
                                    // Client hung up: stop draining the worker so
                                    // it can abandon the decode instead of
                                    // generating tokens nobody will read.
                                    break;
                                }
                            }
                            Ok(chunk) => {
                                let is_ok = chunk.status == "ok" && chunk.error_message.is_empty();
                                let final_chunk = GenerateChunk {
                                    request_id: request_id.clone(),
                                    tenant_id: tenant_id.clone(),
                                    finished: true,
                                    input_tokens: chunk.input_tokens,
                                    output_tokens: chunk.output_tokens,
                                    worker_latency_ms: chunk.worker_latency_ms,
                                    end_to_end_latency_ms: received_at.elapsed().as_secs_f64()
                                        * 1000.0,
                                    ttft_ms: chunk.ttft_ms,
                                    tpot_ms: chunk.tpot_ms,
                                    engine_ttft_ms,
                                    gate_wait_ms,
                                    status: if is_ok {
                                        "ok".to_string()
                                    } else {
                                        "worker_runtime_error".to_string()
                                    },
                                    error_message: chunk.error_message,
                                    ..Default::default()
                                };
                                let _ = tx.send(Ok(final_chunk)).await;
                                break;
                            }
                            Err(status) => {
                                let _ = tx
                                    .send(Ok(stream_error_chunk(
                                        &request_id,
                                        &tenant_id,
                                        &status,
                                        received_at,
                                        engine_ttft_ms,
                                        gate_wait_ms,
                                    )))
                                    .await;
                                break;
                            }
                        }
                    }
                }
                Err(status) => {
                    let _ = tx
                        .send(Ok(stream_error_chunk(
                            &request_id,
                            &tenant_id,
                            &status,
                            received_at,
                            0.0,
                            gate_wait_ms,
                        )))
                        .await;
                }
            }
        });

        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }

    async fn get_engine_stats(
        &self,
        _request: Request<StatsRequest>,
    ) -> Result<Response<StatsReply>, Status> {
        let state = self.state.lock().await;

        Ok(Response::new(StatsReply {
            uptime_seconds: self.started_at.elapsed().as_secs(),
            accepted_requests: state.accepted_requests,
            queue_depth: state.queue_depth,
            mode: self.mode.clone(),
        }))
    }
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_millis() as u64
}

fn new_request_id() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_nanos();
    format!("req-{now}")
}

pub async fn execute_with_worker(
    worker_client: &DirectWorkerClient,
    request_id: String,
    tenant_id: String,
    prompt: String,
    max_tokens: u32,
    received_at: Instant,
    scheduled_at: Instant,
) -> GenerateReply {
    match worker_client
        .generate(WorkerGenerateRequest {
            request_id: request_id.clone(),
            tenant_id: tenant_id.clone(),
            prompt,
            max_tokens,
            submitted_at_ms: now_unix_ms(),
        })
        .await
    {
        Ok(worker_reply) => build_generate_reply(worker_reply, received_at, scheduled_at),
        Err(error) => build_transport_error_reply(
            request_id,
            tenant_id,
            error,
            received_at,
            scheduled_at,
        ),
    }
}

pub fn build_generate_reply(
    worker_reply: WorkerGenerateReply,
    received_at: Instant,
    scheduled_at: Instant,
) -> GenerateReply {
    let queue_wait_ms = if scheduled_at > received_at {
        (scheduled_at - received_at).as_secs_f64() * 1000.0
    } else {
        0.0
    };

    let is_ok = worker_reply.status == "ok" && worker_reply.error_message.is_empty();
    let normalized_status = if is_ok {
        "ok".to_string()
    } else {
        "worker_runtime_error".to_string()
    };

    GenerateReply {
        request_id: worker_reply.request_id,
        tenant_id: worker_reply.tenant_id,
        response_text: worker_reply.response_text,
        input_tokens: worker_reply.input_tokens,
        output_tokens: worker_reply.output_tokens,
        worker_latency_ms: worker_reply.worker_latency_ms,
        end_to_end_latency_ms: received_at.elapsed().as_secs_f64() * 1000.0,
        ttft_ms: worker_reply.ttft_ms,
        tpot_ms: worker_reply.tpot_ms,
        status: if queue_wait_ms > 0.0 {
            format!("{normalized_status} queue_wait_ms={queue_wait_ms:.2}")
        } else {
            normalized_status
        },
        error_message: worker_reply.error_message,
    }
}

pub fn build_transport_error_reply(
    request_id: String,
    tenant_id: String,
    error: Status,
    received_at: Instant,
    scheduled_at: Instant,
) -> GenerateReply {
    let queue_wait_ms = if scheduled_at > received_at {
        (scheduled_at - received_at).as_secs_f64() * 1000.0
    } else {
        0.0
    };
    let status_prefix = classify_transport_error(&error);

    GenerateReply {
        request_id,
        tenant_id,
        response_text: String::new(),
        input_tokens: 0,
        output_tokens: 0,
        worker_latency_ms: 0.0,
        end_to_end_latency_ms: received_at.elapsed().as_secs_f64() * 1000.0,
        ttft_ms: 0.0,
        tpot_ms: 0.0,
        status: if queue_wait_ms > 0.0 {
            format!("{status_prefix} queue_wait_ms={queue_wait_ms:.2}")
        } else {
            status_prefix.to_string()
        },
        error_message: error.to_string(),
    }
}

fn stream_error_chunk(
    request_id: &str,
    tenant_id: &str,
    error: &Status,
    received_at: Instant,
    engine_ttft_ms: f64,
    gate_wait_ms: f64,
) -> GenerateChunk {
    GenerateChunk {
        request_id: request_id.to_string(),
        tenant_id: tenant_id.to_string(),
        finished: true,
        end_to_end_latency_ms: received_at.elapsed().as_secs_f64() * 1000.0,
        engine_ttft_ms,
        gate_wait_ms,
        status: classify_transport_error(error).to_string(),
        error_message: error.to_string(),
        ..Default::default()
    }
}

fn classify_transport_error(error: &Status) -> &'static str {
    match error.code() {
        tonic::Code::DeadlineExceeded => "worker_timeout",
        tonic::Code::Unavailable => "worker_unavailable",
        tonic::Code::Cancelled => "worker_cancelled",
        tonic::Code::ResourceExhausted => "worker_resource_exhausted",
        tonic::Code::Internal => "worker_internal_error",
        _ => "worker_transport_error",
    }
}
