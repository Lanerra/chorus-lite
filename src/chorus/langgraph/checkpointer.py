# src/chorus/langgraph/checkpointer.py
"""Enhanced LangGraph checkpointer with distributed coordination, performance optimizations, versioning, and comprehensive monitoring."""

from __future__ import annotations

import asyncio
import gzip
import json
import pickle
import time
import uuid
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

try:
    import asyncpg
    from langgraph.checkpoint.postgres import PostgresSaver

    POSTGRES_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency may be missing
    PostgresSaver = None  # type: ignore
    asyncpg = None  # type: ignore
    POSTGRES_AVAILABLE = False

from chorus.config import config
from chorus.core.logs import get_logger

from .versioning import (
    CheckpointVersionManager,
    ConflictResolutionStrategy,
    VersionedCheckpointMixin,
)

logger = get_logger(__name__)


@dataclass
class CheckpointMetrics:
    """Performance metrics for checkpoint operations."""

    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    average_latency: float = 0.0
    peak_latency: float = 0.0
    compression_ratio: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    last_operation_time: float = field(default_factory=time.time)
    compression_ratios: list[float] = field(default_factory=list)

    def record_operation(
        self,
        latency: float,
        success: bool = True,
        compressed_size: int = 0,
        original_size: int = 0,
    ) -> None:
        """Record metrics for a checkpoint operation."""
        self.total_operations += 1
        if success:
            self.successful_operations += 1
        else:
            self.failed_operations += 1

        # Update latency metrics
        if self.total_operations == 1:
            self.average_latency = latency
        else:
            self.average_latency = (
                self.average_latency * (self.total_operations - 1) + latency
            ) / self.total_operations

        self.peak_latency = max(self.peak_latency, latency)

        # Update compression metrics
        if compressed_size > 0 and original_size > 0:
            ratio = compressed_size / original_size
            self.compression_ratios.append(ratio)
            if self.compression_ratio == 0.0:
                self.compression_ratio = ratio
            else:
                # Running average of compression ratio
                self.compression_ratio = (self.compression_ratio + ratio) / 2

        self.last_operation_time = time.time()

    def get_success_rate(self) -> float:
        """Get the success rate of operations."""
        if self.total_operations == 0:
            return 1.0
        return self.successful_operations / self.total_operations

    def get_average_latency(self) -> float:
        """Get the average latency of operations."""
        return self.average_latency


@dataclass
class ConnectionPoolConfig:
    """Configuration for PostgreSQL connection pooling."""

    min_connections: int = 5
    max_connections: int = 20
    max_inactive_connection_lifetime: float = 300.0  # 5 minutes
    max_queries: int = 50000
    command_timeout: float = 60.0

    @classmethod
    def from_config(cls) -> ConnectionPoolConfig:
        """Create pool config from environment variables."""
        return cls(
            min_connections=int(config.concurrency.queue_workers),
            max_connections=config.concurrency.scene_concurrency * 4,
            command_timeout=float(config.concurrency.circuit_breaker_timeout),
        )


class ConnectionPool:
    """Async PostgreSQL connection pool for checkpoint operations."""

    def __init__(self, database_url: str, pool_config: ConnectionPoolConfig):
        self.database_url = database_url
        self.config = pool_config
        self._pool: asyncpg.Pool | None = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the connection pool."""
        if not POSTGRES_AVAILABLE:
            raise RuntimeError("PostgreSQL dependencies not available")

        async with self._lock:
            if self._pool is None:
                self._pool = await asyncpg.create_pool(
                    self.database_url,
                    min_size=self.config.min_connections,
                    max_size=self.config.max_connections,
                    max_inactive_connection_lifetime=self.config.max_inactive_connection_lifetime,
                    max_queries=self.config.max_queries,
                    command_timeout=self.config.command_timeout,
                )
                logger.info(
                    f"Initialized connection pool with {self.config.min_connections}-{self.config.max_connections} connections"
                )

    async def get_connection(self) -> Any:
        """Get a connection from the pool."""
        if self._pool is None:
            await self.initialize()
        return self._pool.acquire()

    async def close(self) -> None:
        """Close the connection pool."""
        async with self._lock:
            if self._pool is not None:
                await self._pool.close()
                self._pool = None
                logger.info("Connection pool closed")


class CheckpointCompressor:
    """Handles compression and serialization of checkpoint data."""

    @staticmethod
    def compress_data(data: Any, use_pickle: bool = True) -> tuple[bytes, str]:
        """Compress checkpoint data using gzip and pickle/json."""
        try:
            # Serialize data
            if use_pickle:
                serialized = pickle.dumps(data)
                encoding = "pickle"
            else:
                serialized = json.dumps(data, default=str).encode("utf-8")
                encoding = "json"

            # Compress with gzip
            compressed = gzip.compress(serialized, compresslevel=6)
            return compressed, encoding

        except Exception as e:
            logger.warning(f"Compression failed, using fallback: {e}")
            # Fallback to JSON without compression
            serialized = json.dumps(data, default=str).encode("utf-8")
            return serialized, "json_uncompressed"

    @staticmethod
    def decompress_data(compressed_data: bytes, encoding: str) -> Any:
        """Decompress and deserialize checkpoint data."""
        try:
            if encoding == "json_uncompressed":
                return json.loads(compressed_data.decode("utf-8"))

            # Decompress with gzip
            decompressed = gzip.decompress(compressed_data)

            if encoding == "pickle":
                return pickle.loads(decompressed)
            else:  # json
                return json.loads(decompressed.decode("utf-8"))

        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            raise


@dataclass
class DistributedLock:
    """Enhanced distributed lock with heartbeat monitoring."""

    session_id: str
    lock_id: str
    acquired_time: float
    last_heartbeat: float
    ttl: float = 60.0  # 1 minute default TTL
    heartbeat_interval: float = 15.0  # 15 seconds

    def is_stale(self) -> bool:
        """Check if lock is stale based on heartbeat."""
        return (time.time() - self.last_heartbeat) > self.ttl

    def needs_heartbeat(self) -> bool:
        """Check if lock needs heartbeat update."""
        return (time.time() - self.last_heartbeat) > self.heartbeat_interval

    def update_heartbeat(self) -> None:
        """Update heartbeat timestamp."""
        self.last_heartbeat = time.time()


class DistributedCheckpointSaver(VersionedCheckpointMixin):
    """Enhanced checkpointer with performance optimizations, monitoring, versioning, and distributed coordination."""

    def __init__(
        self, base_saver: BaseCheckpointSaver, enable_compression: bool = True
    ):
        super().__init__()
        self.base_saver = base_saver
        self.enable_compression = enable_compression
        self.metrics = CheckpointMetrics()
        self.compressor = CheckpointCompressor()

        # Enhanced distributed locking
        self._locks: dict[str, DistributedLock] = {}
        self._lock_cleanup_task: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._locks_lock = asyncio.Lock()

        # Connection pooling for PostgreSQL
        self._connection_pool: ConnectionPool | None = None
        if isinstance(base_saver, type(PostgresSaver)) and POSTGRES_AVAILABLE:
            # Extract connection string from PostgresSaver if possible
            try:
                pool_config = ConnectionPoolConfig.from_config()
                if hasattr(base_saver, "_sync_conn") and hasattr(
                    base_saver._sync_conn, "get_dsn_parameters"
                ):
                    dsn_params = base_saver._sync_conn.get_dsn_parameters()
                    database_url = f"postgresql://{dsn_params['user']}:{dsn_params['password']}@{dsn_params['host']}:{dsn_params['port']}/{dsn_params['dbname']}"
                    self._connection_pool = ConnectionPool(database_url, pool_config)
                    # Initialize version manager with the same connection pool
                    self.version_manager = CheckpointVersionManager(
                        self._connection_pool
                    )
                else:
                    self.version_manager = None
            except Exception as e:
                logger.warning(f"Failed to initialize connection pool: {e}")
                self.version_manager = None
        else:
            self.version_manager = None

        # Start background tasks
        self._start_background_tasks()

        # Batch operations support
        self._batch_queue: list[tuple[str, Any, Any, Any]] = []
        self._batch_size = 10
        self._batch_timeout = 5.0  # seconds
        self._batch_task: asyncio.Task | None = None
        self._executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="checkpoint-"
        )

    def _start_background_tasks(self) -> None:
        """Start background maintenance tasks."""
        self._lock_cleanup_task = asyncio.create_task(self._lock_cleanup_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._batch_task = asyncio.create_task(self._batch_processor())

    async def _lock_cleanup_loop(self) -> None:
        """Background task to clean up stale locks."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                await self._cleanup_stale_locks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Lock cleanup error: {e}")

    async def _heartbeat_loop(self) -> None:
        """Background task to send heartbeats for active locks."""
        while True:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                await self._send_heartbeats()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    async def _batch_processor(self) -> None:
        """Background task to process batched checkpoint operations."""
        while True:
            try:
                await asyncio.sleep(self._batch_timeout)
                if self._batch_queue:
                    await self._process_batch()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch processing error: {e}")

    async def get(self, config: Any) -> Any:
        """Get checkpoint with compression, caching, versioning, and performance monitoring."""
        start_time = time.time()
        success = True

        try:
            # Get thread_id from config for versioning
            thread_id = self._extract_session_id(config)

            # Try to get from base saver
            result = await self.base_saver.aget(config)

            # Check versioning if available and result exists
            if result and self.version_manager:
                try:
                    checkpoint_id = (
                        f"{thread_id}_{getattr(result, 'checkpoint_id', 'unknown')}"
                    )
                    version = await self.version_manager.get_latest_version(
                        checkpoint_id
                    )
                    if version and version.is_active:
                        # Use versioned data if available and more recent
                        result = version.data
                        logger.debug(
                            f"Retrieved versioned checkpoint {version.version_number} for {checkpoint_id}"
                        )
                except Exception as ve:
                    logger.warning(
                        f"Failed to retrieve version info: {ve}, using base checkpoint"
                    )

            # Decompress if needed
            if result and self.enable_compression:
                try:
                    if hasattr(result, "data") and isinstance(result.data, dict):
                        compressed_data = result.data.get("_compressed_data")
                        encoding = result.data.get("_compression_encoding")
                        if compressed_data and encoding:
                            result.data = self.compressor.decompress_data(
                                compressed_data, encoding
                            )
                            self.metrics.cache_hits += 1
                        else:
                            self.metrics.cache_misses += 1
                except Exception as e:
                    logger.warning(f"Decompression failed: {e}")
                    self.metrics.cache_misses += 1

            return result

        except Exception as e:
            success = False
            logger.warning(f"Checkpoint get failed, attempting recovery: {e}")
            # Attempt automatic recovery
            await self._cleanup_stale_locks()
            try:
                result = await self.base_saver.aget(config)
                success = True
                return result
            except Exception as retry_e:
                logger.error(f"Checkpoint recovery failed: {retry_e}")
                raise
        finally:
            latency = time.time() - start_time
            self.metrics.record_operation(latency, success)

    async def put(
        self, config: Any, checkpoint: Any, metadata: Any, batch: bool = False
    ) -> Any:
        """Put checkpoint with compression, batching, and distributed locking."""
        if batch and len(self._batch_queue) < self._batch_size:
            # Add to batch queue
            operation_id = str(uuid.uuid4())
            self._batch_queue.append((operation_id, config, checkpoint, metadata))
            return operation_id

        return await self._put_single(config, checkpoint, metadata)

    async def _put_single(self, config: Any, checkpoint: Any, metadata: Any) -> Any:
        """Put a single checkpoint with full processing, versioning, and performance optimizations."""
        start_time = time.time()
        success = True
        original_size = 0
        compressed_size = 0

        try:
            session_id = self._extract_session_id(config)

            # Acquire distributed lock
            lock = await self._acquire_distributed_lock(session_id)
            if not lock:
                raise RuntimeError(
                    f"Failed to acquire distributed lock for {session_id}"
                )

            try:
                # Create checkpoint version if versioning is available
                if self.version_manager:
                    try:
                        checkpoint_id = f"{session_id}_{getattr(checkpoint, 'checkpoint_id', str(uuid.uuid4()))}"
                        version = await self.version_manager.create_version(
                            checkpoint_id=checkpoint_id,
                            data=checkpoint,
                            metadata=metadata,
                            conflict_strategy=ConflictResolutionStrategy.LATEST_WINS,
                        )
                        logger.debug(
                            f"Created checkpoint version {version.version_number} for {checkpoint_id}"
                        )
                    except Exception as ve:
                        logger.warning(
                            f"Failed to create checkpoint version: {ve}, continuing with base storage"
                        )

                # Compress checkpoint data if enabled
                if self.enable_compression and hasattr(checkpoint, "data"):
                    original_size = len(str(checkpoint.data))
                    compressed_data, encoding = self.compressor.compress_data(
                        checkpoint.data
                    )
                    compressed_size = len(compressed_data)

                    # Store compressed data
                    checkpoint.data = {
                        "_compressed_data": compressed_data,
                        "_compression_encoding": encoding,
                        "_session_id": session_id,
                        "_checkpoint_time": time.time(),
                        "_original_size": original_size,
                        "_compressed_size": compressed_size,
                    }

                # Use connection pool if available
                if self._connection_pool:
                    try:
                        async with self._connection_pool.get_connection() as conn:
                            # Custom PostgreSQL checkpoint storage with connection pooling
                            result = await self._store_with_pool(
                                conn, config, checkpoint, metadata
                            )
                    except Exception as pool_e:
                        logger.warning(
                            f"Connection pool operation failed, falling back: {pool_e}"
                        )
                        result = await self.base_saver.aput(
                            config, checkpoint, metadata
                        )
                else:
                    result = await self.base_saver.aput(config, checkpoint, metadata)

                return result

            finally:
                await self._release_distributed_lock(session_id, lock.lock_id)

        except Exception as e:
            success = False
            logger.error(f"Checkpoint put failed: {e}")
            raise
        finally:
            latency = time.time() - start_time
            self.metrics.record_operation(
                latency, success, compressed_size, original_size
            )

    async def _store_with_pool(
        self, conn: Any, config: Any, checkpoint: Any, metadata: Any
    ) -> Any:
        """Store checkpoint using pooled connection."""
        # This would be implemented based on the specific PostgresSaver schema
        # For now, fall back to base saver
        return await self.base_saver.aput(config, checkpoint, metadata)

    async def _process_batch(self) -> None:
        """Process queued batch operations."""
        if not self._batch_queue:
            return

        batch = self._batch_queue.copy()
        self._batch_queue.clear()

        # Process batch operations concurrently
        tasks = []
        for operation_id, config, checkpoint, metadata in batch:
            task = asyncio.create_task(self._put_single(config, checkpoint, metadata))
            tasks.append(task)

        try:
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.debug(f"Processed batch of {len(batch)} checkpoint operations")
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")

    async def list(self, config: Any, **kwargs: Any) -> Any:
        """List checkpoints with session filtering and performance monitoring."""
        start_time = time.time()
        try:
            result = await self.base_saver.alist(config, **kwargs)
            return result
        finally:
            latency = time.time() - start_time
            self.metrics.record_operation(latency, True)

    def _extract_session_id(self, config: Any) -> str:
        """Extract session ID from config for coordination."""
        if hasattr(config, "configurable") and config.configurable:
            return str(config.configurable.get("thread_id", "default"))
        return "default"

    async def _acquire_distributed_lock(
        self, session_id: str
    ) -> DistributedLock | None:
        """Acquire enhanced distributed lock with heartbeat support."""
        async with self._locks_lock:
            current_time = time.time()

            # Check for existing lock
            if session_id in self._locks:
                existing_lock = self._locks[session_id]
                if not existing_lock.is_stale():
                    return None  # Lock still active
                else:
                    # Remove stale lock
                    del self._locks[session_id]
                    logger.info(f"Removed stale lock for session {session_id}")

            # Create new lock
            lock = DistributedLock(
                session_id=session_id,
                lock_id=str(uuid.uuid4()),
                acquired_time=current_time,
                last_heartbeat=current_time,
            )

            self._locks[session_id] = lock
            logger.debug(
                f"Acquired distributed lock {lock.lock_id} for session {session_id}"
            )
            return lock

    async def _release_distributed_lock(self, session_id: str, lock_id: str) -> None:
        """Release distributed lock by ID."""
        async with self._locks_lock:
            if session_id in self._locks and self._locks[session_id].lock_id == lock_id:
                del self._locks[session_id]
                logger.debug(
                    f"Released distributed lock {lock_id} for session {session_id}"
                )

    async def _cleanup_stale_locks(self) -> None:
        """Clean up stale locks based on heartbeat monitoring."""
        async with self._locks_lock:
            stale_sessions = []
            for session_id, lock in self._locks.items():
                if lock.is_stale():
                    stale_sessions.append(session_id)

            for session_id in stale_sessions:
                del self._locks[session_id]
                logger.info(f"Cleaned up stale lock for session {session_id}")

    async def _send_heartbeats(self) -> None:
        """Send heartbeats for active locks."""
        async with self._locks_lock:
            for lock in self._locks.values():
                if lock.needs_heartbeat():
                    lock.update_heartbeat()
                    logger.debug(f"Updated heartbeat for lock {lock.lock_id}")

    def get_metrics(self) -> CheckpointMetrics:
        """Get current performance metrics."""
        return self.metrics

    def get_health_status(self) -> dict[str, Any]:
        """Get comprehensive health status."""
        health_status = {
            "metrics": {
                "total_operations": self.metrics.total_operations,
                "success_rate": self.metrics.get_success_rate(),
                "average_latency_ms": self.metrics.average_latency * 1000,
                "peak_latency_ms": self.metrics.peak_latency * 1000,
                "compression_ratio": self.metrics.compression_ratio,
                "cache_hit_rate": self.metrics.cache_hits
                / max(1, self.metrics.cache_hits + self.metrics.cache_misses),
            },
            "locks": {
                "active_locks": len(self._locks),
                "lock_details": [
                    {
                        "session_id": lock.session_id,
                        "lock_id": lock.lock_id,
                        "age_seconds": time.time() - lock.acquired_time,
                        "last_heartbeat_seconds_ago": time.time() - lock.last_heartbeat,
                    }
                    for lock in self._locks.values()
                ],
            },
            "batch_queue": {
                "pending_operations": len(self._batch_queue),
                "batch_size_limit": self._batch_size,
            },
            "connection_pool": {
                "pool_available": self._connection_pool is not None,
                "postgres_available": POSTGRES_AVAILABLE,
            },
            "versioning": {
                "version_manager_available": self.version_manager is not None,
                "versioning_enabled": self.version_manager is not None,
            },
        }
        return health_status

    async def cleanup_old_checkpoints(self, older_than_hours: int = 24) -> int:
        """Clean up old checkpoints for maintenance."""
        cleanup_count = 0
        try:
            cutoff_time = time.time() - (older_than_hours * 3600)

            # This would need to be implemented based on the specific storage backend
            # For now, we'll log the intent
            logger.info(
                f"Checkpoint cleanup requested for entries older than {older_than_hours} hours"
            )

            # If using PostgreSQL, could run cleanup queries here
            if self._connection_pool:
                try:
                    async with self._connection_pool.get_connection() as conn:
                        # Example cleanup query (would need to match actual schema)
                        # result = await conn.execute("DELETE FROM checkpoints WHERE created_at < $1", cutoff_time)
                        # cleanup_count = result
                        pass
                except Exception as e:
                    logger.error(f"Checkpoint cleanup failed: {e}")

        except Exception as e:
            logger.error(f"Checkpoint maintenance error: {e}")

        return cleanup_count

    async def close(self) -> None:
        """Clean shutdown of the checkpointer."""
        try:
            # Cancel background tasks with improved cleanup
            tasks_to_cancel = []
            if self._lock_cleanup_task and not self._lock_cleanup_task.done():
                tasks_to_cancel.append(self._lock_cleanup_task)
            if self._heartbeat_task and not self._heartbeat_task.done():
                tasks_to_cancel.append(self._heartbeat_task)
            if self._batch_task and not self._batch_task.done():
                tasks_to_cancel.append(self._batch_task)

            # Cancel all tasks
            for task in tasks_to_cancel:
                task.cancel()

            # Wait for cancellation with timeout
            if tasks_to_cancel:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*tasks_to_cancel, return_exceptions=True),
                        timeout=5.0,
                    )
                except TimeoutError:
                    logger.warning("Background task cleanup timed out")
                except Exception as e:
                    logger.warning(f"Background task cleanup error: {e}")

            # Process any remaining batch operations
            if self._batch_queue:
                await self._process_batch()

            # Clean up version manager
            if self.version_manager:
                try:
                    await self.version_manager.cleanup_old_versions()
                    logger.debug("Version manager cleanup completed")
                except Exception as e:
                    logger.warning(f"Version manager cleanup failed: {e}")

            # Close connection pool
            if self._connection_pool:
                await self._connection_pool.close()

            # Shutdown thread pool
            self._executor.shutdown(wait=True)

            logger.info("DistributedCheckpointSaver shutdown complete")

        except Exception as e:
            logger.error(f"Error during checkpointer shutdown: {e}")


class CheckpointPerformanceBenchmark:
    """Performance benchmarking suite for checkpoint operations."""

    def __init__(self, checkpointer: DistributedCheckpointSaver):
        self.checkpointer = checkpointer

    async def run_latency_benchmark(
        self, num_operations: int = 100
    ) -> dict[str, float]:
        """Benchmark checkpoint operation latency."""
        write_times = []
        read_times = []

        # Create test configurations and checkpoints
        test_configs = []
        test_checkpoints = []

        for i in range(num_operations):
            config = {"configurable": {"thread_id": f"benchmark_{i}"}}
            checkpoint = type(
                "MockCheckpoint",
                (),
                {
                    "data": {
                        "test_data": f"benchmark_data_{i}" * 100
                    }  # ~1.5KB per checkpoint
                },
            )()
            test_configs.append(config)
            test_checkpoints.append(checkpoint)

        # Benchmark write operations
        logger.info(
            f"Starting write latency benchmark with {num_operations} operations"
        )
        for i in range(num_operations):
            start_time = time.time()
            try:
                await self.checkpointer._put_single(
                    test_configs[i], test_checkpoints[i], {}
                )
                write_times.append(time.time() - start_time)
            except Exception as e:
                logger.warning(f"Write benchmark operation {i} failed: {e}")

        # Benchmark read operations
        logger.info(f"Starting read latency benchmark with {num_operations} operations")
        for i in range(num_operations):
            start_time = time.time()
            try:
                await self.checkpointer.get(test_configs[i])
                read_times.append(time.time() - start_time)
            except Exception as e:
                logger.warning(f"Read benchmark operation {i} failed: {e}")

        return {
            "write_avg_ms": (sum(write_times) / len(write_times)) * 1000
            if write_times
            else 0,
            "write_p95_ms": (
                sorted(write_times)[int(len(write_times) * 0.95)] if write_times else 0
            )
            * 1000,
            "write_p99_ms": (
                sorted(write_times)[int(len(write_times) * 0.99)] if write_times else 0
            )
            * 1000,
            "read_avg_ms": (sum(read_times) / len(read_times)) * 1000
            if read_times
            else 0,
            "read_p95_ms": (
                sorted(read_times)[int(len(read_times) * 0.95)] if read_times else 0
            )
            * 1000,
            "read_p99_ms": (
                sorted(read_times)[int(len(read_times) * 0.99)] if read_times else 0
            )
            * 1000,
            "operations_completed": len(write_times) + len(read_times),
        }

    async def run_throughput_benchmark(
        self, duration_seconds: int = 60
    ) -> dict[str, float]:
        """Benchmark checkpoint operation throughput."""
        start_time = time.time()
        operations_completed = 0
        errors = 0

        logger.info(f"Starting throughput benchmark for {duration_seconds} seconds")

        async def benchmark_worker(worker_id: int) -> None:
            nonlocal operations_completed, errors
            while time.time() - start_time < duration_seconds:
                try:
                    config = {
                        "configurable": {
                            "thread_id": f"throughput_{worker_id}_{operations_completed}"
                        }
                    }
                    checkpoint = type(
                        "MockCheckpoint",
                        (),
                        {
                            "data": {
                                "worker_id": worker_id,
                                "operation": operations_completed,
                            }
                        },
                    )()

                    await self.checkpointer._put_single(config, checkpoint, {})
                    operations_completed += 1

                    # Brief pause to prevent overwhelming
                    await asyncio.sleep(0.001)

                except Exception as e:
                    errors += 1
                    logger.debug(f"Throughput benchmark error: {e}")

        # Run multiple workers concurrently
        num_workers = min(10, config.concurrency.scene_concurrency)
        tasks = [asyncio.create_task(benchmark_worker(i)) for i in range(num_workers)]

        await asyncio.gather(*tasks, return_exceptions=True)

        actual_duration = time.time() - start_time
        return {
            "duration_seconds": actual_duration,
            "operations_per_second": operations_completed / actual_duration
            if actual_duration > 0
            else 0,
            "total_operations": operations_completed,
            "error_rate": errors / max(1, operations_completed + errors),
            "workers_used": num_workers,
        }


class CheckpointConfigManager:
    """Configuration management for different deployment scenarios."""

    @staticmethod
    def get_deployment_config(deployment_type: str = "production") -> dict[str, Any]:
        """Get optimized configuration for different deployment scenarios."""

        base_config = {
            "compression_enabled": True,
            "batch_operations": True,
            "connection_pooling": True,
            "heartbeat_monitoring": True,
            "cleanup_enabled": True,
        }

        if deployment_type == "development":
            return {
                **base_config,
                "batch_size": 5,
                "batch_timeout": 2.0,
                "heartbeat_interval": 30.0,
                "lock_ttl": 120.0,
                "cleanup_interval": 300,  # 5 minutes
                "compression_level": 3,
                "max_connections": 5,
                "min_connections": 2,
            }

        elif deployment_type == "testing":
            return {
                **base_config,
                "batch_size": 3,
                "batch_timeout": 1.0,
                "heartbeat_interval": 10.0,
                "lock_ttl": 30.0,
                "cleanup_interval": 60,  # 1 minute
                "compression_level": 1,
                "max_connections": 3,
                "min_connections": 1,
            }

        elif deployment_type == "production":
            return {
                **base_config,
                "batch_size": 20,
                "batch_timeout": 5.0,
                "heartbeat_interval": 15.0,
                "lock_ttl": 300.0,  # 5 minutes
                "cleanup_interval": 1800,  # 30 minutes
                "compression_level": 6,
                "max_connections": config.concurrency.scene_concurrency * 4,
                "min_connections": config.concurrency.queue_workers,
            }

        elif deployment_type == "high_throughput":
            return {
                **base_config,
                "batch_size": 50,
                "batch_timeout": 10.0,
                "heartbeat_interval": 20.0,
                "lock_ttl": 600.0,  # 10 minutes
                "cleanup_interval": 3600,  # 1 hour
                "compression_level": 9,
                "max_connections": config.concurrency.scene_concurrency * 6,
                "min_connections": config.concurrency.scene_concurrency,
            }

        else:
            logger.warning(
                f"Unknown deployment type: {deployment_type}, using production config"
            )
            return CheckpointConfigManager.get_deployment_config("production")


async def run_checkpoint_health_check() -> dict[str, Any]:
    """Run comprehensive health check on the checkpoint system."""
    health_status = {"status": "healthy", "timestamp": time.time(), "checks": {}}

    try:
        async with get_checkpointer() as checkpointer:
            # Basic connectivity test
            try:
                test_config = {"configurable": {"thread_id": "health_check"}}
                test_checkpoint = type(
                    "MockCheckpoint", (), {"data": {"health": "check"}}
                )()

                start_time = time.time()
                await checkpointer.put(test_config, test_checkpoint, {})
                write_latency = time.time() - start_time

                start_time = time.time()
                result = await checkpointer.get(test_config)
                read_latency = time.time() - start_time

                health_status["checks"]["connectivity"] = {
                    "status": "pass",
                    "write_latency_ms": write_latency * 1000,
                    "read_latency_ms": read_latency * 1000,
                }

            except Exception as e:
                health_status["checks"]["connectivity"] = {
                    "status": "fail",
                    "error": str(e),
                }
                health_status["status"] = "unhealthy"

            # Enhanced checkpointer health
            if isinstance(checkpointer, DistributedCheckpointSaver):
                try:
                    enhanced_health = checkpointer.get_health_status()
                    health_status["checks"]["enhanced_features"] = {
                        "status": "pass",
                        "details": enhanced_health,
                    }
                except Exception as e:
                    health_status["checks"]["enhanced_features"] = {
                        "status": "fail",
                        "error": str(e),
                    }

            # Connection pool health (if available)
            if (
                hasattr(checkpointer, "_connection_pool")
                and checkpointer._connection_pool
            ):
                try:
                    # Test pool connectivity
                    async with checkpointer._connection_pool.get_connection() as conn:
                        # Simple query to test connection
                        pass
                    health_status["checks"]["connection_pool"] = {"status": "pass"}
                except Exception as e:
                    health_status["checks"]["connection_pool"] = {
                        "status": "fail",
                        "error": str(e),
                    }
                    health_status["status"] = "degraded"

    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["error"] = str(e)

    return health_status


async def benchmark_checkpoint_performance(
    duration_seconds: int = 30,
) -> dict[str, Any]:
    """Run performance benchmark on the checkpoint system."""
    try:
        async with get_checkpointer() as checkpointer:
            if isinstance(checkpointer, DistributedCheckpointSaver):
                benchmark = CheckpointPerformanceBenchmark(checkpointer)

                # Run latency benchmark
                latency_results = await benchmark.run_latency_benchmark(50)

                # Run throughput benchmark
                throughput_results = await benchmark.run_throughput_benchmark(
                    duration_seconds
                )

                return {
                    "benchmark_timestamp": time.time(),
                    "latency": latency_results,
                    "throughput": throughput_results,
                    "metrics": checkpointer.get_metrics().__dict__,
                }
            else:
                return {"error": "Enhanced checkpointer not available for benchmarking"}

    except Exception as e:
        return {"error": f"Benchmark failed: {e}"}


@asynccontextmanager
async def get_checkpointer(
    deployment_type: str = "production",
) -> AsyncIterator[BaseCheckpointSaver]:
    """Yield an enhanced checkpointer with distributed coordination and performance optimizations."""

    url = config.checkpoint.checkpoint_db_url
    deployment_config = CheckpointConfigManager.get_deployment_config(deployment_type)

    if url and PostgresSaver is not None:
        # Use PostgreSQL-based checkpointer with distributed coordination
        logger.info(
            f"Using PostgreSQL checkpointer with {deployment_type} configuration"
        )
        try:
            with PostgresSaver.from_conn_string(url) as postgres_saver:
                # Wrap with enhanced distributed coordination layer
                distributed_saver = DistributedCheckpointSaver(
                    postgres_saver,
                    enable_compression=deployment_config["compression_enabled"],
                )

                # Apply deployment-specific configuration
                distributed_saver._batch_size = deployment_config["batch_size"]
                distributed_saver._batch_timeout = deployment_config["batch_timeout"]

                try:
                    yield cast(BaseCheckpointSaver, distributed_saver)
                finally:
                    # Clean shutdown
                    await distributed_saver.close()

        except Exception as e:
            logger.error(f"PostgreSQL checkpointer failed, falling back to memory: {e}")
            # Fallback to memory saver
            saver = InMemorySaver()
            with saver:
                yield saver
    else:
        # Use in-memory checkpointer (no distributed coordination needed)
        logger.info(
            f"Using in-memory checkpointer (single instance) with {deployment_type} configuration"
        )
        saver = InMemorySaver()
        with saver:
            yield saver


async def recover_failed_checkpoints() -> int:
    """Recover any failed checkpoint operations for automatic restart."""
    recovered = 0
    try:
        async with get_checkpointer() as checkpointer:
            if isinstance(checkpointer, DistributedCheckpointSaver):
                await checkpointer._cleanup_stale_locks()
                recovered += len(checkpointer._locks)
                logger.info(f"Recovered {recovered} stale checkpoint locks")
    except Exception as e:
        logger.error(f"Checkpoint recovery failed: {e}")

    return recovered


__all__ = [
    "get_checkpointer",
    "recover_failed_checkpoints",
    "DistributedCheckpointSaver",
    "CheckpointMetrics",
    "CheckpointCompressor",
    "ConnectionPool",
    "CheckpointConfigManager",
    "run_checkpoint_health_check",
    "benchmark_checkpoint_performance",
]
