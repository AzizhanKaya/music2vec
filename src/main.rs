mod dataset;
mod model;
mod train;

use anyhow::Result;
use burn::backend::wgpu;
use env_logger;
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
struct Opt {
    #[structopt(long, default_value = "1")]
    epochs: usize,

    #[structopt(long, default_value = "64")]
    dim: usize,

    #[structopt(long, default_value = "3")]
    k: usize,

    #[structopt(long, default_value = "0.1")]
    lr: f64,

    #[structopt(long, default_value = "10")]
    batch_size: usize,

    #[structopt(long, default_value = "8")]
    window_size: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    dotenv::load().unwrap();
    let opt = Opt::from_args();

    let device = wgpu::WgpuDevice::default();

    println!("Starting music2vec training...");
    println!(
        "Config: epochs={}, dim={}, lr={}, window_size={},batch_size={}, k={}",
        opt.epochs, opt.dim, opt.lr, opt.window_size, opt.batch_size, opt.k
    );

    let database_url =
        std::env::var("DATABASE_URL").expect("DATABASE_URL environment variable must be set");
    let pool = dataset::create_pool(&database_url).await?;

    println!("Starting training...");
    let _ = train::train_loop(opt, pool, device).await?;

    println!("Training completed successfully!");

    println!("All done.");
    Ok(())
}
