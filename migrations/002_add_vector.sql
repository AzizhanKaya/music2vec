-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Add embedding column (256 dimensions)
ALTER TABLE musics
ADD COLUMN embedding vector(64);

-- Optional: index for similarity search
CREATE INDEX ON musics
USING hnsw (embedding vector_cosine_ops)
WITH (
    m = 16,          -- graph connectivity
    ef_construction = 200  -- indexing time accuracy
);
