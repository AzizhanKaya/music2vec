use burn::module::{Module, Param};
use burn::nn::{Embedding, EmbeddingConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

#[derive(Module, Debug)]
pub struct SkipGram<B: Backend> {
    pub emb: Embedding<B>,
}

impl<B: Backend> SkipGram<B> {
    pub fn init_from(embeddings: Vec<Vec<f32>>, dim: usize, device: &B::Device) -> Self {
        let num_embeddings = embeddings.len();

        let flattened: Vec<f32> = embeddings.into_iter().flatten().collect();

        let weight: Tensor<B, 2> = Tensor::<B, 1>::from_floats(flattened.as_slice(), device)
            .reshape([num_embeddings, dim]);

        let mut emb = EmbeddingConfig::new(num_embeddings, dim).init(device);
        emb.weight = Param::from_tensor(weight);

        Self { emb }
    }

    pub fn new(n_embedding: usize, dim: usize, device: &B::Device) -> Self {
        let bound = (1.0f64 / (dim as f64).sqrt()) as f64;
        let emb = EmbeddingConfig::new(n_embedding, dim)
            .with_initializer(burn::nn::Initializer::Uniform {
                min: -bound,
                max: bound,
            })
            .init(device);

        Self { emb }
    }

    pub fn forward(
        &self,
        center_tensor: Tensor<B, 2, Int>,
        context_tensor: Tensor<B, 2, Int>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let center_embeddings_3d = self.emb.forward(center_tensor);
        let context_embeddings_3d = self.emb.forward(context_tensor);

        let center_embeddings = center_embeddings_3d.squeeze(1);
        let context_embeddings = context_embeddings_3d.squeeze(1);

        (center_embeddings, context_embeddings)
    }

    pub fn get_embeddings(&self) -> Tensor<B, 2> {
        self.emb.weight.val()
    }
}
