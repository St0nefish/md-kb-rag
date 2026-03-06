use anyhow::Result;
use async_openai::{
    Client,
    config::OpenAIConfig,
    types::{CreateEmbeddingRequestArgs, EmbeddingInput},
};

use crate::config::EmbeddingConfig;

pub struct EmbedClient {
    client: Client<OpenAIConfig>,
    model: String,
    batch_size: usize,
}

impl EmbedClient {
    pub fn new(config: &EmbeddingConfig) -> Self {
        let openai_config = OpenAIConfig::new()
            .with_api_base(&config.base_url)
            .with_api_key("not-needed");

        let client = Client::with_config(openai_config);

        Self {
            client,
            model: config.model.clone(),
            batch_size: config.batch_size,
        }
    }

    pub async fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut all_embeddings: Vec<Vec<f32>> = Vec::with_capacity(texts.len());

        for batch in texts.chunks(self.batch_size) {
            let request = CreateEmbeddingRequestArgs::default()
                .model(&self.model)
                .input(EmbeddingInput::StringArray(batch.to_vec()))
                .build()?;

            let response = self.client.embeddings().create(request).await?;

            let mut data = response.data;
            data.sort_by_key(|e| e.index);

            for embedding in data {
                all_embeddings.push(embedding.embedding);
            }
        }

        Ok(all_embeddings)
    }

    pub async fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        let results = self.embed_texts(&[query.to_string()]).await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No embedding returned for query"))
    }
}
