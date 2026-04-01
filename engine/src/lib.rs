pub mod config;
pub mod engine;
pub mod worker_client;

pub mod omnispan {
    tonic::include_proto!("omnispan");
}
