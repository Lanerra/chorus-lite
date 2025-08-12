# src/chorus/config/config.py
"""Configuration system for Chorus."""

from __future__ import annotations

import os

from pydantic import BaseModel, Field


class DatabaseConfig(BaseModel):
    """Database configuration settings."""

    postgres_user: str = Field(default="chorus", env="POSTGRES_USER")
    postgres_password: str = Field(default="chorus_password", env="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="chorus_canon", env="POSTGRES_DB")
    postgres_host: str = Field(default="localhost", env="POSTGRES_HOST")
    postgres_port: str = Field(default="5432", env="POSTGRES_PORT")

    @property
    def postgres_url(self) -> str:
        """Generate PostgreSQL connection URL with psycopg driver."""
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}@"
            f"{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    api_base: str = Field(default="http://localhost:8080/v1", env="OPENAI_API_BASE")
    api_key: str = Field(default="sk-1234", env="OPENAI_API_KEY")
    # Default sampling temperature for text generation. If not set, falls back to 0.7.
    temperature: float | None = Field(default=None, env="TEMPERATURE")


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""

    model: str = Field(default="ollama/nomic-embed-text:latest", env="EMBEDDING_MODEL")
    api_base: str = Field(default="http://localhost:11434", env="EMBEDDING_API_BASE")
    api_key: str = Field(default="", env="EMBEDDING_API_KEY")


class AgentModelConfig(BaseModel):
    """Agent model configuration."""

    story_architect: str = Field(
        default="openai/qwen3-a3b", env="STORY_ARCHITECT_MODEL"
    )
    scene_generator: str = Field(
        default="openai/qwen3-a3b", env="SCENE_GENERATOR_MODEL"
    )
    integration_manager: str = Field(
        default="openai/qwen3-a3b", env="INTEGRATION_MANAGER_MODEL"
    )


class SystemConfig(BaseModel):
    """System configuration settings."""

    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="./chorus_log.txt", env="CHORUS_LOG_FILE")
    surprise_me: bool = Field(default=False, env="SURPRISE_ME")
    port: int = Field(default=8000, env="PORT")
    disable_auth: bool = Field(default=False, env="CHORUS_DISABLE_AUTH")
    web_token: str = Field(default="changeme", env="CHORUS_WEB_TOKEN")


class CacheConfig(BaseModel):
    """Cache configuration settings."""

    cache_ttl_world_gen: int = Field(default=3600, env="CACHE_TTL_WORLD_GEN")
    cache_ttl_expensive: int = Field(default=1800, env="CACHE_TTL_EXPENSIVE")


class RetryConfig(BaseModel):
    """Retry configuration settings."""

    retry_attempts: int = Field(default=3, env="RETRY_ATTEMPTS")
    retry_backoff: float = Field(default=0.5, env="RETRY_BACKOFF")
    retry_max_interval: float = Field(default=60, env="RETRY_MAX_INTERVAL")


class IngestionConfig(BaseModel):
    """Ingestion configuration settings."""

    chunk_size: int = Field(default=10000, env="INGEST_CHUNK_SIZE")
    segment_model: str = Field(
        default="openai/qwen3-a3b",
        env="INGEST_SEGMENT_MODEL",
    )


class WorkerConfig(BaseModel):
    """Worker configuration settings."""

    worker_idle: float = Field(default=0.1, env="WORKER_IDLE")
    worker_id: str = Field(default="", env="WORKER_ID")
    worker_handlers: str = Field(default="", env="WORKER_HANDLERS")
    max_task_retries: int = Field(default=3, env="MAX_TASK_RETRIES")
    handler_retries: int = Field(default=3, env="HANDLER_RETRIES")


class SnapshotConfig(BaseModel):
    """Snapshot configuration settings."""

    interval: int = Field(default=3600, env="SNAPSHOT_INTERVAL")


class ConcurrencyConfig(BaseModel):
    """Concurrency and parallel processing configuration."""

    scene_concurrency: int = Field(default=5, env="SCENE_CONCURRENCY")
    queue_workers: int = Field(default=3, env="QUEUE_WORKERS")
    scene_timeout: int = Field(default=300, env="SCENE_TIMEOUT")
    circuit_breaker_threshold: int = Field(default=5, env="CIRCUIT_BREAKER_THRESHOLD")
    circuit_breaker_timeout: int = Field(default=60, env="CIRCUIT_BREAKER_TIMEOUT")


class CheckpointConfig(BaseModel):
    """Enhanced checkpoint system configuration settings."""

    checkpoint_db_url: str = Field(default="", env="CHECKPOINT_DB_URL")
    checkpoint_compression: bool = Field(default=True, env="CHECKPOINT_COMPRESSION")
    checkpoint_batch_size: int = Field(default=10, env="CHECKPOINT_BATCH_SIZE")
    checkpoint_batch_timeout: float = Field(default=5.0, env="CHECKPOINT_BATCH_TIMEOUT")
    checkpoint_heartbeat_interval: float = Field(
        default=15.0, env="CHECKPOINT_HEARTBEAT_INTERVAL"
    )
    checkpoint_lock_ttl: float = Field(default=300.0, env="CHECKPOINT_LOCK_TTL")
    checkpoint_cleanup_interval: int = Field(
        default=1800, env="CHECKPOINT_CLEANUP_INTERVAL"
    )
    checkpoint_pool_min_connections: int = Field(
        default=5, env="CHECKPOINT_POOL_MIN_CONNECTIONS"
    )
    checkpoint_pool_max_connections: int = Field(
        default=20, env="CHECKPOINT_POOL_MAX_CONNECTIONS"
    )
    checkpoint_deployment_type: str = Field(
        default="production", env="CHECKPOINT_DEPLOYMENT_TYPE"
    )


class MiscellaneousConfig(BaseModel):
    """Miscellaneous configuration settings."""

    initial_idea: str = Field(default="", env="INITIAL_IDEA")
    weaver_mode: str = Field(default="standard", env="WEAVER_MODE")
    memory_limit: int = Field(default=10, env="MEMORY_LIMIT")


class ChorusConfig(BaseModel):
    """Main configuration class."""

    database: DatabaseConfig = DatabaseConfig()
    llm: LLMConfig = LLMConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    agents: AgentModelConfig = AgentModelConfig()
    system: SystemConfig = SystemConfig()
    cache: CacheConfig = CacheConfig()
    retry: RetryConfig = RetryConfig()
    ingestion: IngestionConfig = IngestionConfig()
    worker: WorkerConfig = WorkerConfig()
    snapshot: SnapshotConfig = SnapshotConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    miscellaneous: MiscellaneousConfig = MiscellaneousConfig()
    concurrency: ConcurrencyConfig = ConcurrencyConfig()

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"

    @classmethod
    def load(cls) -> ChorusConfig:
        """Load configuration from environment variables."""
        return cls.parse_obj(os.environ)


# Global configuration instance
config = ChorusConfig.load()
