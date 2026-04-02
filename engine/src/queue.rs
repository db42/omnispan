use std::sync::Arc;
use std::time::Instant;

use tokio::sync::{mpsc, oneshot, Mutex};
use tokio::time::{sleep, Duration};
use tonic::Status;

use crate::engine::{build_generate_reply, execute_with_worker, EngineState};
use crate::omnispan::WorkerGenerateRequest;
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

#[derive(Clone, Copy)]
pub struct QueueConfig {
    pub micro_batch_enabled: bool,
    pub batch_window_ms: u64,
    pub max_batch_size: usize,
}

pub async fn run_scheduler_loop(
    mut queue_rx: mpsc::Receiver<QueuedRequest>,
    worker_client: DirectWorkerClient,
    state: Arc<Mutex<EngineState>>,
    config: QueueConfig,
) {
    while let Some(request) = queue_rx.recv().await {
        let mut batch = vec![request];
        decrement_queue_depth(&state, 1).await;

        if config.micro_batch_enabled && config.max_batch_size > 1 {
            if config.batch_window_ms > 0 {
                sleep(Duration::from_millis(config.batch_window_ms)).await;
            }

            while batch.len() < config.max_batch_size {
                match queue_rx.try_recv() {
                    Ok(request) => {
                        batch.push(request);
                        decrement_queue_depth(&state, 1).await;
                    }
                    Err(mpsc::error::TryRecvError::Empty) => break,
                    Err(mpsc::error::TryRecvError::Disconnected) => break,
                }
            }
        }

        let scheduled_at = Instant::now();
        if batch.len() == 1 {
            let request = batch.pop().expect("single request batch");
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
            continue;
        }

        let worker_requests: Vec<WorkerGenerateRequest> = batch
            .iter()
            .map(|request| WorkerGenerateRequest {
                request_id: request.request_id.clone(),
                tenant_id: request.tenant_id.clone(),
                prompt: request.prompt.clone(),
                max_tokens: request.max_tokens,
                submitted_at_ms: 0,
            })
            .collect();

        match worker_client.generate_batch(worker_requests).await {
            Ok(batch_reply) => {
                let mut responses = batch_reply.responses.into_iter();
                for request in batch {
                    let reply = responses.next().ok_or_else(|| {
                        Status::internal("worker batch response length mismatch")
                    });
                    let result = reply.map(|worker_reply| {
                        build_generate_reply(worker_reply, request.received_at, scheduled_at)
                    });
                    let _ = request.reply_tx.send(result);
                }
            }
            Err(error) => {
                let message = error.to_string();
                for request in batch {
                    let _ = request.reply_tx.send(Err(Status::unavailable(message.clone())));
                }
            }
        }
    }
}

async fn decrement_queue_depth(state: &Arc<Mutex<EngineState>>, amount: u64) {
    let mut state = state.lock().await;
    state.queue_depth = state.queue_depth.saturating_sub(amount);
}
