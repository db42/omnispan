use std::sync::Arc;
use std::time::Instant;

use tokio::sync::{mpsc, oneshot, Mutex};
use tonic::Status;

use crate::engine::{execute_with_worker, EngineState};
use crate::omnispan::GenerateReply;
use crate::worker_client::DirectWorkerClient;

pub struct QueuedRequest {
    pub request_id: String,
    pub tenant_id: String,
    pub prompt: String,
    pub max_tokens: u32,
    pub received_at: Instant,
    pub reply_tx: oneshot::Sender<Result<GenerateReply, Status>>,
}

pub async fn run_scheduler_loop(
    mut queue_rx: mpsc::Receiver<QueuedRequest>,
    worker_client: DirectWorkerClient,
    state: Arc<Mutex<EngineState>>,
) {
    while let Some(request) = queue_rx.recv().await {
        let scheduled_at = Instant::now();
        {
            let mut state = state.lock().await;
            state.queue_depth = state.queue_depth.saturating_sub(1);
        }

        let result = execute_with_worker(
            &worker_client,
            request.request_id,
            request.tenant_id,
            request.prompt,
            request.max_tokens,
            request.received_at,
            scheduled_at,
        )
        .await;

        let _ = request.reply_tx.send(result);
    }
}
