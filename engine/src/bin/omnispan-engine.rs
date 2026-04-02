use tokio::sync::mpsc;
use tonic::transport::Server;
use tonic_health::server::health_reporter;

use omnispan::config::{env_or_default, env_u16};
use omnispan::engine::EngineService;
use omnispan::omnispan::engine_server::EngineServer;
use omnispan::queue::{run_scheduler_loop, QueueConfig};
use omnispan::worker_client::DirectWorkerClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bind_host = env_or_default("BIND_HOST", "127.0.0.1");
    let bind_port = env_u16("BIND_PORT", 50061);
    let mode = env_or_default("ENGINE_MODE", "direct");
    let worker_endpoint = env_or_default("WORKER_ENDPOINT", "http://127.0.0.1:50071");
    let queue_capacity = env_u16("QUEUE_CAPACITY", 1024) as usize;
    let batch_window_ms = env_u16("BATCH_WINDOW_MS", 10) as u64;
    let max_batch_size = env_u16("MAX_BATCH_SIZE", 4) as usize;
    let addr = format!("{bind_host}:{bind_port}").parse()?;
    let (queue_tx, queue_rx) = mpsc::channel(queue_capacity);

    let worker_client = DirectWorkerClient::new(worker_endpoint.clone());
    let service = EngineService::new(
        mode.clone(),
        worker_client.clone(),
        if mode == "queued" || mode == "micro_batch" {
            Some(queue_tx)
        } else {
            None
        },
    );
    let shared_state = service.shared_state();

    let (mut reporter, health_service) = health_reporter();
    reporter.set_serving::<EngineServer<EngineService>>().await;

    println!(
        "Starting engine on {addr} in {mode} mode with worker {worker_endpoint} batch_window_ms={batch_window_ms} max_batch_size={max_batch_size}"
    );

    if mode == "queued" || mode == "micro_batch" {
        tokio::spawn(run_scheduler_loop(
            queue_rx,
            worker_client,
            shared_state,
            QueueConfig {
                micro_batch_enabled: mode == "micro_batch",
                batch_window_ms,
                max_batch_size,
            },
        ));
    }

    Server::builder()
        .add_service(health_service)
        .add_service(EngineServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
