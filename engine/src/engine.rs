use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use tokio::sync::Mutex;
use tonic::{Request, Response, Status};

use crate::omnispan::engine_server::Engine;
use crate::omnispan::{
    GenerateReply, GenerateRequest, StatsReply, StatsRequest, WorkerGenerateRequest,
};
use crate::worker_client::DirectWorkerClient;

#[derive(Debug, Default)]
struct EngineState {
    accepted_requests: u64,
    queue_depth: u64,
}

#[derive(Clone)]
pub struct EngineService {
    mode: String,
    started_at: Instant,
    state: Arc<Mutex<EngineState>>,
    worker_client: DirectWorkerClient,
}

impl EngineService {
    pub fn new(mode: String, worker_client: DirectWorkerClient) -> Self {
        Self {
            mode,
            started_at: Instant::now(),
            state: Arc::new(Mutex::new(EngineState::default())),
            worker_client,
        }
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

        if self.mode != "direct" {
            return Err(Status::failed_precondition(format!(
                "mode {} is not implemented yet",
                self.mode
            )));
        }

        let worker_reply = self
            .worker_client
            .generate(WorkerGenerateRequest {
                request_id: inner.request_id.clone(),
                tenant_id: inner.tenant_id.clone(),
                prompt: inner.prompt,
                max_tokens: inner.max_tokens,
                submitted_at_ms: now_unix_ms(),
            })
            .await?;

        Ok(Response::new(GenerateReply {
            request_id: worker_reply.request_id,
            tenant_id: worker_reply.tenant_id,
            response_text: worker_reply.response_text,
            input_tokens: worker_reply.input_tokens,
            output_tokens: worker_reply.output_tokens,
            worker_latency_ms: worker_reply.worker_latency_ms,
            end_to_end_latency_ms: started_at.elapsed().as_secs_f64() * 1000.0,
            status: worker_reply.status,
            error_message: worker_reply.error_message,
        }))
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
