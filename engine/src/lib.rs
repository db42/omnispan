pub mod config;
pub mod engine;
pub mod queue;
pub mod worker_client;

pub mod omnispan {
    tonic::include_proto!("omnispan");
}
