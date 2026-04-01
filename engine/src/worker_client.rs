use tonic::transport::Channel;
use tonic::Status;

use crate::omnispan::worker_client::WorkerClient;
use crate::omnispan::{WorkerGenerateReply, WorkerGenerateRequest};

#[derive(Clone)]
pub struct DirectWorkerClient {
    endpoint: String,
}

impl DirectWorkerClient {
    pub fn new(endpoint: String) -> Self {
        Self { endpoint }
    }

    pub async fn generate(
        &self,
        request: WorkerGenerateRequest,
    ) -> Result<WorkerGenerateReply, Status> {
        let mut client = WorkerClient::<Channel>::connect(self.endpoint.clone())
            .await
            .map_err(|error| Status::unavailable(format!("worker connect failed: {error}")))?;

        let response = client.generate(request).await?;
        Ok(response.into_inner())
    }
}
