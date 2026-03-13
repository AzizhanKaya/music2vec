use crate::dataset::DataSetStreamer;
use crate::model::SkipGram;
use crate::{Opt, dataset};
use anyhow::Result;
use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::Tensor;
use burn::tensor::activation::log_sigmoid;
use burn::tensor::{Float, Int};
use itertools::MultiUnzip;
use pgvector::Vector;
use sqlx::{Pool, Postgres};

pub type Backend = Autodiff<Wgpu>;

fn neg_sampling_loss(
    scores: Tensor<Backend, 1, Float>,
    labels: Tensor<Backend, 1, Float>,
) -> Tensor<Backend, 1, Float> {
    let ones = Tensor::<Backend, 1, Float>::ones_like(&labels);
    (labels.clone() * -log_sigmoid(scores.clone()) + (ones - labels) * -log_sigmoid(-scores)).mean()
}

pub async fn train_loop(opt: Opt, pool: Pool<Postgres>, device: WgpuDevice) -> Result<()> {
    let mut optimizer = AdamConfig::new().init();
    let vocab_size = dataset::vocab_size(&pool).await as usize;

    let mut lr = opt.lr;
    const MIN_LR: f64 = 0.001;
    let lr_init: f64 = lr;

    let mut model: SkipGram<Backend> = SkipGram::new(vocab_size, opt.dim, &device);

    let mut streamer = DataSetStreamer::new(pool.clone()).await;

    for epoch in 1..=opt.epochs {
        println!("\t Epoch {}/{}", epoch, opt.epochs);
        let mut step_count = 0;

        streamer.reset();

        let mut rx = streamer
            .clone()
            .spawn_streamer(opt.batch_size, opt.window_size, opt.k, 8)
            .await;

        while let Some(batch_data) = rx.recv().await {
            if batch_data.is_empty() {
                continue;
            }

            let (centers_ids, contexts_ids, labels): (Vec<i32>, Vec<i32>, Vec<f32>) =
                batch_data.into_iter().multiunzip();

            let centers_tensor =
                Tensor::<Backend, 1, Int>::from_ints(centers_ids.as_slice(), &device)
                    .reshape([centers_ids.len(), 1]);

            let contexts_tensor: Tensor<Backend, 2, Int> =
                Tensor::<Backend, 1, Int>::from_ints(contexts_ids.as_slice(), &device)
                    .reshape([contexts_ids.len(), 1]);

            let labels_tensor: Tensor<Backend, 1, Float> =
                Tensor::from_floats(labels.as_slice(), &device);

            let (center_emb, context_emb) = model.forward(centers_tensor, contexts_tensor);

            let scores: Tensor<Backend, 1> = (center_emb * context_emb)
                .sum_dim(1)
                .reshape([centers_ids.len()]);

            let loss = neg_sampling_loss(scores, labels_tensor);
            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &model);

            model = optimizer.step(lr, model, grads_params);
            let loss_val = loss.into_data().as_slice::<f32>().unwrap()[0];
            step_count += 1;

            streamer
                .pb
                .println(format!(" Step {} - Loss: {:.4}", step_count, loss_val));
        }

        lr = lr_init * (MIN_LR / lr_init).powf(epoch as f64 / opt.epochs as f64);
    }

    println!("Updating embeddings in database...");

    let embeddings = model
        .get_embeddings()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec();

    let embeddings_vec: Vec<Vec<f32>> = embeddings.chunks(opt.dim).map(|c| c.to_vec()).collect();

    let id_emb: (Vec<i32>, Vec<Vector>) = embeddings_vec
        .into_iter()
        .enumerate()
        .map(|(i, v)| ((i as i32) + 1, Vector::from(v)))
        .collect();

    dataset::update_embeddings(&pool, id_emb).await.unwrap();

    Ok(())
}
