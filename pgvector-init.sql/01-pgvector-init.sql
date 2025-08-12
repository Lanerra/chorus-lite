-- Create extension for pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Create a table to store embeddings with vector support
CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding VECTOR(1536) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create index on the embedding column for efficient similarity search
CREATE INDEX IF NOT EXISTS idx_embeddings_embedding ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Create index on the created_at column for time-based queries
CREATE INDEX IF NOT EXISTS idx_embeddings_created_at ON embeddings (created_at);
