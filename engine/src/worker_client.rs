use std::time::Duration;

use tokio::time::timeout;
use tonic::transport::Channel;
use tonic::Status;

use crate::omnispan::worker_client::WorkerClient;
use crate::omnispan::{
    WorkerBatchGenerateReply, WorkerBatchGenerateRequest, WorkerGenerateReply, WorkerGenerateRequest,
};

#[derive(Clone)]
pub struct DirectWorkerClient {
    endpoint: String,
    timeout: Duration,
}

impl DirectWorkerClient {
    pub fn new(endpoint: String, timeout_ms: u64) -> Self {
        Self {
            endpoint,
            timeout: Duration::from_millis(timeout_ms),
        }
    }

    pub async fn generate(
        &self,
        request: WorkerGenerateRequest,
    ) -> Result<WorkerGenerateReply, Status> {
        let mut client = timeout(
            self.timeout,
            WorkerClient::<Channel>::connect(self.endpoint.clone()),
        )
        .await
        .map_err(|_| {
            Status::deadline_exceeded(format!(
                "worker connect timed out after {} ms",
                self.timeout.as_millis()
            ))
        })?
        .map_err(|error| Status::unavailable(format!("worker connect failed: {error}")))?;

        let response = timeout(self.timeout, client.generate(request))
            .await
            .map_err(|_| {
                Status::deadline_exceeded(format!(
                    "worker generate timed out after {} ms",
                    self.timeout.as_millis()
                ))
            })??;
        Ok(response.into_inner())
    }

    pub async fn generate_batch(
        &self,
        requests: Vec<WorkerGenerateRequest>,
    ) -> Result<WorkerBatchGenerateReply, Status> {
        let mut client = timeout(
            self.timeout,
            WorkerClient::<Channel>::connect(self.endpoint.clone()),
        )
        .await
        .map_err(|_| {
            Status::deadline_exceeded(format!(
                "worker connect timed out after {} ms",
                self.timeout.as_millis()
            ))
        })?
        .map_err(|error| Status::unavailable(format!("worker connect failed: {error}")))?;

        let response = timeout(
            self.timeout,
            client.generate_batch(WorkerBatchGenerateRequest { requests }),
        )
        .await
        .map_err(|_| {
            Status::deadline_exceeded(format!(
                "worker batch generate timed out after {} ms",
                self.timeout.as_millis()
            ))
        })??;
        Ok(response.into_inner())
    }
}
