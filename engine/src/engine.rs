use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use tokio::sync::{mpsc, oneshot, Mutex};
use tonic::{Request, Response, Status};

use crate::omnispan::engine_server::Engine;
use crate::omnispan::{
    GenerateReply, GenerateRequest, StatsReply, StatsRequest, WorkerGenerateReply,
    WorkerGenerateRequest,
};
use crate::queue::QueuedRequest;
use crate::worker_client::DirectWorkerClient;

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
}

impl EngineService {
    pub fn new(
        mode: String,
        worker_client: DirectWorkerClient,
        queue_tx: Option<mpsc::Sender<QueuedRequest>>,
    ) -> Self {
        Self {
            mode,
            started_at: Instant::now(),
            state: Arc::new(Mutex::new(EngineState::default())),
            worker_client,
            queue_tx,
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
                .await?;

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
                    .map_err(|_| Status::unavailable("queue reply dropped"))??;

                Ok(Response::new(reply))
            }
            _ => Err(Status::failed_precondition(format!(
                "mode {} is not implemented yet",
                self.mode
            ))),
        }
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
) -> Result<GenerateReply, Status> {
    let worker_reply = worker_client
        .generate(WorkerGenerateRequest {
            request_id,
            tenant_id,
            prompt,
            max_tokens,
            submitted_at_ms: now_unix_ms(),
        })
        .await?;

    Ok(build_generate_reply(worker_reply, received_at, scheduled_at))
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

    GenerateReply {
        request_id: worker_reply.request_id,
        tenant_id: worker_reply.tenant_id,
        response_text: worker_reply.response_text,
        input_tokens: worker_reply.input_tokens,
        output_tokens: worker_reply.output_tokens,
        worker_latency_ms: worker_reply.worker_latency_ms,
        end_to_end_latency_ms: received_at.elapsed().as_secs_f64() * 1000.0,
        status: if queue_wait_ms > 0.0 {
            format!("{} queue_wait_ms={queue_wait_ms:.2}", worker_reply.status)
        } else {
            worker_reply.status
        },
        error_message: worker_reply.error_message,
    }
}
