# src/chorus/core/logs.py
"""Enhanced structured event streaming and logging system for real-time communication.

This module implements a comprehensive event system with JSON logging, real-time WebSocket
streaming, token bucket rate limiting, and advanced performance monitoring capabilities.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Prevent circular imports
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

if TYPE_CHECKING:
    from chorus.web.websocket import WebSocketManager


class LogLevel(Enum):
    """Log levels with numeric values for filtering."""

    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class EventType(Enum):
    """Enhanced event types for structured logging."""

    # System events
    SYSTEM = "system"
    SESSION = "session"

    # Workflow events
    WORKFLOW_START = "workflow_start"
    WORKFLOW_COMPLETE = "workflow_complete"
    WORKFLOW_ERROR = "workflow_error"
    WORKFLOW_NODE = "workflow_node"

    # LangGraph specific events
    LANGGRAPH_NODE = "langgraph_node"

    # Scene generation events
    SCENE_GENERATION = "scene_generation"
    SCENE_DRAFT = "scene_draft"
    SCENE_REVIEW = "scene_review"
    SCENE_REVISION = "scene_revision"
    SCENE_FINALIZED = "scene_finalized"

    # LLM interaction events
    LLM_REQUEST = "llm_request"
    
    # Database operation events
    DATABASE_OPERATION = "database_operation"
    
    # Agent operation events
    AGENT_OPERATION = "agent_operation"
    
    # Task processing events
    TASK_PROCESSING = "task_processing"

    # User interaction events
    USER_ACTION = "user_action"
    USER_FEEDBACK = "user_feedback"

    # Performance events
    PERFORMANCE = "performance"
    METRICS = "metrics"

    # Error events
    ERROR = "error"
    WARNING = "warning"

    # Enhanced error handling events
    ERROR_HANDLING_START = "error_handling_start"
    ERROR_RECOVERY_SUCCESS = "error_recovery_success"
    ERROR_RECOVERY_FAILED = "error_recovery_failed"
    ERROR_CLASSIFICATION = "error_classification"
    CIRCUIT_BREAKER_OPEN = "circuit_breaker_open"
    CIRCUIT_BREAKER_CLOSE = "circuit_breaker_close"
    CIRCUIT_BREAKER_HALF_OPEN = "circuit_breaker_half_open"
    FAULT_ISOLATION_ACTIVATED = "fault_isolation_activated"
    FAULT_ISOLATION_DEACTIVATED = "fault_isolation_deactivated"
    RETRY_ATTEMPT = "retry_attempt"
    RETRY_EXHAUSTED = "retry_exhausted"
    ERROR_ESCALATION = "error_escalation"
    MANUAL_INTERVENTION_REQUIRED = "manual_intervention_required"
    ERROR_ROLLBACK = "error_rollback"
    ERROR_SKIP = "error_skip"

    # UI events
    UI_UPDATE = "ui_update"
    UI_DELTA = "ui_delta"


class Priority(Enum):
    """Event priority levels for rate limiting."""

    CRITICAL = 1  # Errors, system failures
    HIGH = 2  # User actions, workflow events
    NORMAL = 3  # Scene generation, content updates
    LOW = 4  # Performance metrics, background sync


@dataclass
class EventMetrics:
    """Performance metrics for event processing."""

    total_events: int = 0
    events_by_type: dict[str, int] = field(default_factory=dict)
    events_by_priority: dict[int, int] = field(default_factory=dict)
    delivery_latencies: deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    failed_deliveries: int = 0
    rate_limited_events: int = 0

    def record_event(self, event_type: str, priority: int) -> None:
        """Record an event for metrics tracking."""
        self.total_events += 1
        self.events_by_type[event_type] = self.events_by_type.get(event_type, 0) + 1
        self.events_by_priority[priority] = self.events_by_priority.get(priority, 0) + 1

    def record_delivery_latency(self, latency: float) -> None:
        """Record delivery latency in milliseconds."""
        self.delivery_latencies.append(latency)

    def record_failed_delivery(self) -> None:
        """Record a failed delivery."""
        self.failed_deliveries += 1

    def record_rate_limited(self) -> None:
        """Record a rate-limited event."""
        self.rate_limited_events += 1

    def get_average_latency(self) -> float:
        """Get average delivery latency in milliseconds."""
        if not self.delivery_latencies:
            return 0.0
        return sum(self.delivery_latencies) / len(self.delivery_latencies)

    def get_p95_latency(self) -> float:
        """Get 95th percentile delivery latency in milliseconds."""
        if not self.delivery_latencies:
            return 0.0
        sorted_latencies = sorted(self.delivery_latencies)
        index = int(0.95 * len(sorted_latencies))
        return sorted_latencies[min(index, len(sorted_latencies) - 1)]


@dataclass
class TokenBucket:
    """Token bucket implementation for rate limiting with priority support."""

    capacity: int = 100
    refill_rate: float = 10.0  # tokens per second
    tokens: float = field(init=False)
    last_refill: float = field(init=False)
    priority_weights: dict[int, float] = field(
        default_factory=lambda: {
            Priority.CRITICAL.value: 0.1,  # Critical events use fewer tokens
            Priority.HIGH.value: 0.5,  # High priority events
            Priority.NORMAL.value: 1.0,  # Normal events use 1 token
            Priority.LOW.value: 2.0,  # Low priority events use more tokens
        }
    )

    def __post_init__(self) -> None:
        self.tokens = float(self.capacity)
        self.last_refill = time.time()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def can_consume(self, priority: Priority) -> bool:
        """Check if tokens are available for the given priority."""
        self._refill()
        tokens_needed = self.priority_weights.get(priority.value, 1.0)
        return self.tokens >= tokens_needed

    def consume(self, priority: Priority) -> bool:
        """Consume tokens for the given priority. Returns True if successful."""
        if not self.can_consume(priority):
            return False

        tokens_needed = self.priority_weights.get(priority.value, 1.0)
        self.tokens -= tokens_needed
        return True


@dataclass
class StructuredLogEvent:
    """Enhanced structured log event with comprehensive metadata."""

    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: float = field(default_factory=time.time)
    event_type: EventType = EventType.SYSTEM
    level: LogLevel = LogLevel.INFO
    priority: Priority = Priority.NORMAL
    message: str = ""
    session_id: str | None = None
    thread_id: str | None = None
    component: str | None = None
    workflow_node: str | None = None
    scene_id: str | None = None
    user_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Performance tracking
    processing_time_ms: float | None = None
    queue_time_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "level": self.level.name,
            "level_value": self.level.value,
            "priority": self.priority.value,
            "priority_name": self.priority.name,
            "message": self.message,
            "session_id": self.session_id,
            "thread_id": self.thread_id,
            "component": self.component,
            "workflow_node": self.workflow_node,
            "scene_id": self.scene_id,
            "user_id": self.user_id,
            "metadata": self.metadata,
            "processing_time_ms": self.processing_time_ms,
            "queue_time_ms": self.queue_time_ms,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class EventFilter:
    """Enhanced filter for structured log events with granular control."""

    def __init__(
        self,
        min_level: LogLevel = LogLevel.INFO,
        max_level: LogLevel | None = None,
        event_types: set[EventType] | None = None,
        priorities: set[Priority] | None = None,
        session_id: str | None = None,
        thread_id: str | None = None,
        components: set[str] | None = None,
        workflow_nodes: set[str] | None = None,
        scene_ids: set[str] | None = None,
        user_ids: set[str] | None = None,
        time_range: tuple[float, float] | None = None,
        metadata_filters: dict[str, Any] | None = None,
    ):
        """Initialize enhanced event filter.

        Args:
            min_level: Minimum log level to include
            max_level: Maximum log level to include
            event_types: Set of event types to include (None = all)
            priorities: Set of priorities to include (None = all)
            session_id: Specific session ID to filter by
            thread_id: Specific thread ID to filter by
            components: Set of components to include (None = all)
            workflow_nodes: Set of workflow nodes to include (None = all)
            scene_ids: Set of scene IDs to include (None = all)
            user_ids: Set of user IDs to include (None = all)
            time_range: Tuple of (start_time, end_time) to filter by
            metadata_filters: Dictionary of metadata key-value pairs to match
        """
        self.min_level = min_level
        self.max_level = max_level
        self.event_types = event_types
        self.priorities = priorities
        self.session_id = session_id
        self.thread_id = thread_id
        self.components = components
        self.workflow_nodes = workflow_nodes
        self.scene_ids = scene_ids
        self.user_ids = user_ids
        self.time_range = time_range
        self.metadata_filters = metadata_filters or {}

    def matches(self, event: StructuredLogEvent) -> bool:
        """Check if event matches this filter."""
        # Level checks
        if event.level.value < self.min_level.value:
            return False
        if self.max_level and event.level.value > self.max_level.value:
            return False

        # Event type check
        if self.event_types is not None and event.event_type not in self.event_types:
            return False

        # Priority check
        if self.priorities is not None and event.priority not in self.priorities:
            return False

        # Session check
        if self.session_id is not None and event.session_id != self.session_id:
            return False

        # Thread check
        if self.thread_id is not None and event.thread_id != self.thread_id:
            return False

        # Component check
        if (
            self.components is not None
            and event.component is not None
            and event.component not in self.components
        ):
            return False

        # Workflow node check
        if (
            self.workflow_nodes is not None
            and event.workflow_node is not None
            and event.workflow_node not in self.workflow_nodes
        ):
            return False

        # Scene ID check
        if (
            self.scene_ids is not None
            and event.scene_id is not None
            and event.scene_id not in self.scene_ids
        ):
            return False

        # User ID check
        if (
            self.user_ids is not None
            and event.user_id is not None
            and event.user_id not in self.user_ids
        ):
            return False

        # Time range check
        if self.time_range:
            start_time, end_time = self.time_range
            if not (start_time <= event.timestamp <= end_time):
                return False

        # Metadata filters check
        for key, expected_value in self.metadata_filters.items():
            if key not in event.metadata or event.metadata[key] != expected_value:
                return False

        return True


class EventRouter:
    """Routes events to appropriate WebSocket connections based on filters."""

    def __init__(self):
        """Initialize event router."""
        self._connection_filters: dict[str, EventFilter] = {}
        self._session_filters: dict[str, EventFilter] = {}

    def register_connection_filter(
        self, connection_id: str, event_filter: EventFilter
    ) -> None:
        """Register a filter for a specific connection."""
        self._connection_filters[connection_id] = event_filter

    def register_session_filter(
        self, session_id: str, event_filter: EventFilter
    ) -> None:
        """Register a filter for all connections in a session."""
        self._session_filters[session_id] = event_filter

    def unregister_connection_filter(self, connection_id: str) -> None:
        """Remove filter for a connection."""
        self._connection_filters.pop(connection_id, None)

    def unregister_session_filter(self, session_id: str) -> None:
        """Remove filter for a session."""
        self._session_filters.pop(session_id, None)

    def should_route_to_connection(
        self, event: StructuredLogEvent, connection_id: str
    ) -> bool:
        """Check if event should be routed to a specific connection."""
        if connection_id in self._connection_filters:
            return self._connection_filters[connection_id].matches(event)
        return True  # Default to allow if no filter

    def should_route_to_session(
        self, event: StructuredLogEvent, session_id: str
    ) -> bool:
        """Check if event should be routed to a session."""
        if session_id in self._session_filters:
            return self._session_filters[session_id].matches(event)
        return True  # Default to allow if no filter


class EventLogger:
    """Enhanced structured logging system with comprehensive event handling.

    This is the main interface for the structured event system, providing:
    - JSON-structured events with rich metadata
    - Real-time WebSocket streaming with <500ms latency
    - Token bucket rate limiting with priority queues
    - Advanced filtering and routing capabilities
    - Performance monitoring and metrics tracking
    """

    def __init__(self, max_events: int = 10000, enable_rate_limiting: bool = True):
        """Initialize enhanced event logger.

        Args:
            max_events: Maximum number of events to store in memory
            enable_rate_limiting: Whether to enable token bucket rate limiting
        """
        self.max_events = max_events
        self.enable_rate_limiting = enable_rate_limiting

        # Event storage
        self._events: deque[StructuredLogEvent] = deque(maxlen=max_events)
        self._events_lock = asyncio.Lock()

        # Rate limiting
        self._token_bucket = TokenBucket() if enable_rate_limiting else None

        # Event routing
        self._event_router = EventRouter()

        # Performance metrics
        self._metrics = EventMetrics()

        # WebSocket manager for real-time streaming
        self._websocket_manager: WebSocketManager | None = None
        self._websocket_manager_initialized = False

        # Background processing
        self._streaming_task: asyncio.Task | None = None
        self._metrics_task: asyncio.Task | None = None
        self._event_queue: asyncio.Queue[StructuredLogEvent] = asyncio.Queue(
            maxsize=1000
        )

        # Traditional logging integration
        self._traditional_logger = logging.getLogger("chorus")
        self._configure_traditional_logger()

        # Background tasks will start lazily
        self._background_tasks_started = False

    def _configure_traditional_logger(self) -> None:
        """Configure traditional logging system with enhanced formatting."""
        # Remove existing handlers
        for handler in list(self._traditional_logger.handlers):
            self._traditional_logger.removeHandler(handler)

        self._traditional_logger.propagate = False
        level = os.getenv("LOG_LEVEL", "INFO").upper()
        self._traditional_logger.setLevel(level)

        # Create enhanced formatter for better visibility
        enhanced_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S"
        )
        
        # Create simple formatter for file output (no emojis in files)
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console handler with enhanced formatting
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(enhanced_formatter)
        self._traditional_logger.addHandler(console_handler)

        # File handler with simple formatting
        root = Path(__file__).resolve().parents[2]
        log_file = Path(os.getenv("CHORUS_LOG_FILE", root / "chorus_log.txt"))
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        self._traditional_logger.addHandler(file_handler)

        # Add custom handler for structured logging
        structured_handler = StructuredLogHandler(self)
        structured_handler.setFormatter(enhanced_formatter)
        self._traditional_logger.addHandler(structured_handler)

    def _get_websocket_manager(self) -> WebSocketManager | None:
        """Lazily get WebSocket manager to avoid circular imports."""
        if not self._websocket_manager_initialized:
            try:
                from chorus.web.websocket import get_websocket_manager

                self._websocket_manager = get_websocket_manager()
            except ImportError:
                # WebSocket not available, continue without real-time streaming
                pass
            self._websocket_manager_initialized = True

        return self._websocket_manager

    def _start_background_tasks(self) -> None:
        """Start background processing tasks."""
        if self._background_tasks_started:
            return

        try:
            # Only start if we have an event loop
            loop = asyncio.get_running_loop()

            if self._streaming_task is None or self._streaming_task.done():
                self._streaming_task = loop.create_task(self._streaming_loop())

            if self._metrics_task is None or self._metrics_task.done():
                self._metrics_task = loop.create_task(self._metrics_loop())

            self._background_tasks_started = True

        except RuntimeError:
            # No event loop running, will start later when needed
            pass

    async def _streaming_loop(self) -> None:
        """Background task for real-time event streaming with latency tracking."""
        while True:
            try:
                # Get events from queue with timeout
                try:
                    event = await asyncio.wait_for(
                        self._event_queue.get(), timeout=0.01
                    )
                except TimeoutError:
                    continue

                start_time = time.time()

                # Calculate queue time
                if event.queue_time_ms is None:
                    event.queue_time_ms = (start_time - event.timestamp) * 1000

                # Stream to WebSocket clients with routing
                websocket_manager = self._get_websocket_manager()
                if websocket_manager:
                    try:
                        await self._route_and_send_event(websocket_manager, event)

                        # Record delivery latency
                        delivery_latency = (time.time() - start_time) * 1000
                        event.processing_time_ms = delivery_latency
                        self._metrics.record_delivery_latency(delivery_latency)

                    except Exception as e:
                        self._metrics.record_failed_delivery()
                        self._traditional_logger.error("Event delivery failed: %s", e)

            except Exception as e:
                # Log to traditional logger to avoid recursion
                self._traditional_logger.error("Error in event streaming: %s", e)
                await asyncio.sleep(0.1)

    async def _route_and_send_event(
        self, websocket_manager: WebSocketManager, event: StructuredLogEvent
    ) -> None:
        """Route and send event based on filters and session."""
        event_data = event.to_dict()

        if event.session_id:
            # Check session-level routing
            if self._event_router.should_route_to_session(event, event.session_id):
                await websocket_manager.send_log_event(
                    event_data, session_id=event.session_id
                )
        else:
            # Broadcast to all connections
            await websocket_manager.send_log_event(event_data)

    async def _metrics_loop(self) -> None:
        """Background task for periodic metrics reporting."""
        while True:
            try:
                await asyncio.sleep(60)  # Report metrics every minute

                # Create metrics event
                metrics_event = StructuredLogEvent(
                    event_type=EventType.METRICS,
                    level=LogLevel.INFO,
                    priority=Priority.LOW,
                    component="event_logger",
                    message="Event system metrics",
                    metadata={
                        "total_events": self._metrics.total_events,
                        "events_by_type": dict(self._metrics.events_by_type),
                        "events_by_priority": dict(self._metrics.events_by_priority),
                        "average_latency_ms": self._metrics.get_average_latency(),
                        "p95_latency_ms": self._metrics.get_p95_latency(),
                        "failed_deliveries": self._metrics.failed_deliveries,
                        "rate_limited_events": self._metrics.rate_limited_events,
                        "queue_size": self._event_queue.qsize(),
                        "memory_events": len(self._events),
                    },
                )

                # Log metrics without going through the queue to avoid recursion
                await self._log_event_internal(metrics_event, bypass_queue=True)

            except Exception as e:
                self._traditional_logger.error("Error in metrics loop: %s", e)
                await asyncio.sleep(5)

    def _handle_task_exception(self, task: asyncio.Task) -> None:
        """Handle exceptions from fire-and-forget tasks."""
        try:
            exception = task.exception()
            if exception is not None:
                # Use basic logging to avoid recursion
                import logging
                fallback_logger = logging.getLogger("chorus.logs.fallback")
                fallback_logger.error(f"Task exception in logging: {exception}")
        except Exception:
            # Ignore errors in error handling to prevent infinite recursion
            pass

    async def log_event(self, event: StructuredLogEvent) -> None:
        """Log a structured event with rate limiting and routing.

        Args:
            event: Structured log event to log
        """
        try:
            await self._log_event_internal(event)
        except Exception as e:
            # Fallback to traditional logging to avoid recursion
            import logging
            fallback_logger = logging.getLogger("chorus.logs.fallback")
            fallback_logger.error(f"Failed to log structured event: {e}")

    async def _log_event_internal(
        self,
        event: StructuredLogEvent,
        bypass_queue: bool = False,
        bypass_traditional: bool = False,
    ) -> None:
        """Internal event logging with optional queue bypass."""
        # Record metrics
        self._metrics.record_event(event.event_type.value, event.priority.value)

        # Apply rate limiting
        if self._token_bucket and not bypass_queue:
            if not self._token_bucket.consume(event.priority):
                self._metrics.record_rate_limited()
                # For rate-limited events, only log critical events
                if event.priority != Priority.CRITICAL:
                    return

        # Store in memory
        async with self._events_lock:
            self._events.append(event)

        # Queue for real-time streaming (unless bypassing)
        if not bypass_queue:
            try:
                # Add queue timestamp
                event.queue_time_ms = 0  # Will be calculated in streaming loop
                self._event_queue.put_nowait(event)
            except asyncio.QueueFull:
                # Drop oldest event if queue is full
                try:
                    self._event_queue.get_nowait()
                    self._event_queue.put_nowait(event)
                except asyncio.QueueEmpty:
                    pass

        # Log to traditional logger for file/console output (unless bypassing)
        if not bypass_traditional:
            try:
                self._log_to_traditional(event)
            except Exception as e:
                # Use basic logging to avoid recursion
                import logging
                fallback_logger = logging.getLogger("chorus.logs.fallback")
                fallback_logger.error(f"Failed to log to traditional logger: {e}")

    def _log_to_traditional(self, event: StructuredLogEvent) -> None:
        """Log event to traditional logging system with enhanced formatting."""
        log_level = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL,
        }.get(event.level, logging.INFO)

        # Enhanced formatting with better visual hierarchy
        event_type_display = self._format_event_type(event.event_type, event.level)
        context_display = self._format_context(event)
        priority_display = self._format_priority(event.priority)
        
        # Performance metrics display
        timing_display = ""
        if event.processing_time_ms or event.queue_time_ms:
            timing_parts = []
            if event.processing_time_ms:
                timing_parts.append(f"proc:{event.processing_time_ms:.1f}ms")
            if event.queue_time_ms:
                timing_parts.append(f"queue:{event.queue_time_ms:.1f}ms")
            timing_display = f" [{'/'.join(timing_parts)}]"
        
        # Enhanced metadata display for key operations
        metadata_display = self._format_key_metadata(event)
        
        # Construct final formatted message with better structure
        message_parts = [event_type_display]
        if priority_display:
            message_parts.append(priority_display)
        if context_display:
            message_parts.append(context_display)
        if metadata_display:
            message_parts.append(metadata_display)
        
        header = " ".join(message_parts)
        formatted_message = f"{header} ➤ {event.message}{timing_display}"
        
        # Special formatting for errors and critical events
        if event.level in (LogLevel.ERROR, LogLevel.CRITICAL):
            formatted_message = f"🚨 {formatted_message}"
        elif event.level == LogLevel.WARNING:
            formatted_message = f"⚠️  {formatted_message}"
        elif event.event_type in (EventType.WORKFLOW_START, EventType.WORKFLOW_COMPLETE):
            formatted_message = f"🔄 {formatted_message}"
        elif event.event_type in (EventType.LLM_REQUEST, EventType.SCENE_GENERATION):
            formatted_message = f"🤖 {formatted_message}"
        elif event.event_type in (EventType.DATABASE_OPERATION,):
            formatted_message = f"💾 {formatted_message}"

        self._traditional_logger.log(log_level, formatted_message)
    
    def _format_event_type(self, event_type: EventType, level: LogLevel) -> str:
        """Format event type with visual indicators."""
        # Handle case where event_type is not an EventType enum (defensive programming)
        if isinstance(event_type, EventType):
            type_name = event_type.value.upper()
        elif isinstance(event_type, int):
            # Handle integer values (likely LogLevel values passed incorrectly)
            try:
                # Try to find EventType by value
                event_type = EventType.SYSTEM  # Default fallback
                type_name = "SYSTEM"
            except (ValueError, AttributeError):
                type_name = f"UNKNOWN_{event_type}"
        elif hasattr(event_type, 'value'):
            # Handle other enum types
            type_name = str(event_type.value).upper()
        else:
            # Last resort: convert to string
            type_name = str(event_type).upper()
        
        # Color coding for different event types (when using Rich)
        if level in (LogLevel.ERROR, LogLevel.CRITICAL):
            return f"[ERROR:{type_name}]"
        elif level == LogLevel.WARNING:
            return f"[WARN:{type_name}]"
        elif isinstance(event_type, EventType) and event_type in (EventType.WORKFLOW_START, EventType.WORKFLOW_COMPLETE, EventType.LANGGRAPH_NODE):
            return f"[FLOW:{type_name}]"
        elif isinstance(event_type, EventType) and event_type == EventType.LLM_REQUEST:
            return f"[LLM:{type_name}]"
        elif isinstance(event_type, EventType) and event_type == EventType.DATABASE_OPERATION:
            return f"[DB:{type_name}]"
        elif isinstance(event_type, EventType) and event_type in (EventType.SCENE_GENERATION, EventType.SCENE_DRAFT):
            return f"[SCENE:{type_name}]"
        else:
            return f"[{type_name}]"
    
    def _format_context(self, event: StructuredLogEvent) -> str:
        """Format context with improved readability."""
        context_parts = []
        
        # Priority order: scene_id, workflow_node, component, session_id, thread_id
        if event.scene_id:
            context_parts.append(f"scene:{event.scene_id}")
        if event.workflow_node:
            context_parts.append(f"node:{event.workflow_node}")
        if event.component:
            # Simplify component names for readability
            component = event.component.replace("chorus.", "").replace("src.", "")
            context_parts.append(f"comp:{component}")
        if event.session_id:
            context_parts.append(f"sess:{event.session_id[:8]}")
        if event.thread_id:
            context_parts.append(f"thread:{event.thread_id[:8]}")
        
        return f"({' | '.join(context_parts)})" if context_parts else ""
    
    def _format_priority(self, priority: Priority) -> str:
        """Format priority with visual indicators."""
        if priority == Priority.CRITICAL:
            return "[🔴CRIT]"
        elif priority == Priority.HIGH:
            return "[🟡HIGH]"
        elif priority == Priority.LOW:
            return "[🔵LOW]"
        else:
            return ""  # Normal priority doesn't need display
    
    def _format_key_metadata(self, event: StructuredLogEvent) -> str:
        """Format key metadata for better visibility."""
        if not event.metadata:
            return ""
        
        key_info = []
        
        # Show key operational data
        if "operation" in event.metadata:
            key_info.append(f"op:{event.metadata['operation']}")
        if "model" in event.metadata:
            key_info.append(f"model:{event.metadata['model']}")
        if "duration" in event.metadata:
            duration = event.metadata["duration"]
            if isinstance(duration, (int, float)):
                key_info.append(f"dur:{duration:.2f}s")
        if "attempt" in event.metadata and "max_attempts" in event.metadata:
            key_info.append(f"attempt:{event.metadata['attempt']}/{event.metadata['max_attempts']}")
        if "success" in event.metadata:
            success_icon = "✅" if event.metadata["success"] else "❌"
            key_info.append(f"{success_icon}")
        
        return f"<{' | '.join(key_info)}>" if key_info else ""

    def log(
        self,
        level: LogLevel,
        message: str,
        event_type: EventType = EventType.SYSTEM,
        priority: Priority = Priority.NORMAL,
        session_id: str | None = None,
        thread_id: str | None = None,
        component: str | None = None,
        workflow_node: str | None = None,
        scene_id: str | None = None,
        user_id: str | None = None,
        **metadata: Any,
    ) -> None:
        """Log a message with structured metadata.

        Args:
            level: Log level
            message: Log message
            event_type: Type of event
            priority: Event priority for rate limiting
            session_id: Optional session ID
            thread_id: Optional thread ID
            component: Optional component name
            workflow_node: Optional workflow node name
            scene_id: Optional scene ID
            user_id: Optional user ID
            **metadata: Additional metadata
        """
        event = StructuredLogEvent(
            level=level,
            event_type=event_type,
            priority=priority,
            message=message,
            session_id=session_id,
            thread_id=thread_id,
            component=component,
            workflow_node=workflow_node,
            scene_id=scene_id,
            user_id=user_id,
            metadata=metadata,
        )

        # Ensure background tasks are started
        self._start_background_tasks()

        # Use fire-and-forget for non-blocking logging with proper error handling
        try:
            task = asyncio.create_task(self.log_event(event))
            # Add error callback to prevent "Task exception was never retrieved"
            task.add_done_callback(self._handle_task_exception)
        except RuntimeError:
            # No event loop, run synchronously
            try:
                asyncio.run(self.log_event(event))
            except Exception as e:
                # Fallback to traditional logging
                import logging
                fallback_logger = logging.getLogger("chorus.logs.fallback")
                fallback_logger.error(f"Failed to log event: {e}")

    # Convenience methods for different log levels
    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self.log(LogLevel.DEBUG, message, priority=Priority.LOW, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self.log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self.log(
            LogLevel.WARNING,
            message,
            event_type=EventType.WARNING,
            priority=Priority.HIGH,
            **kwargs,
        )

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        self.log(
            LogLevel.ERROR,
            message,
            event_type=EventType.ERROR,
            priority=Priority.CRITICAL,
            **kwargs,
        )

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message."""
        self.log(
            LogLevel.CRITICAL,
            message,
            event_type=EventType.ERROR,
            priority=Priority.CRITICAL,
            **kwargs,
        )

    # Workflow-specific convenience methods
    def log_workflow_start(self, workflow_node: str, **kwargs: Any) -> None:
        """Log workflow node start."""
        self.log(
            LogLevel.INFO,
            f"Starting workflow node: {workflow_node}",
            event_type=EventType.WORKFLOW_START,
            priority=Priority.HIGH,
            workflow_node=workflow_node,
            **kwargs,
        )

    def log_workflow_complete(
        self, workflow_node: str, duration_ms: float | None = None, **kwargs: Any
    ) -> None:
        """Log workflow node completion."""
        metadata = kwargs.get("metadata", {})
        if duration_ms is not None:
            metadata["duration_ms"] = duration_ms
        kwargs["metadata"] = metadata

        self.log(
            LogLevel.INFO,
            f"Completed workflow node: {workflow_node}",
            event_type=EventType.WORKFLOW_COMPLETE,
            priority=Priority.HIGH,
            workflow_node=workflow_node,
            **kwargs,
        )

    def log_workflow_error(self, workflow_node: str, error: str, **kwargs: Any) -> None:
        """Log workflow node error."""
        self.log(
            LogLevel.ERROR,
            f"Error in workflow node {workflow_node}: {error}",
            event_type=EventType.WORKFLOW_ERROR,
            priority=Priority.CRITICAL,
            workflow_node=workflow_node,
            **kwargs,
        )

    def log_scene_event(
        self, scene_id: str, event_type: EventType, message: str, **kwargs: Any
    ) -> None:
        """Log scene-related event."""
        self.log(
            LogLevel.INFO,
            message,
            event_type=event_type,
            priority=Priority.NORMAL,
            scene_id=scene_id,
            **kwargs,
        )

    def log_user_action(
        self, action: str, user_id: str | None = None, **kwargs: Any
    ) -> None:
        """Log user action."""
        self.log(
            LogLevel.INFO,
            f"User action: {action}",
            event_type=EventType.USER_ACTION,
            priority=Priority.HIGH,
            user_id=user_id,
            **kwargs,
        )

    # Error handling specific convenience methods
    def log_error_handling_start(
        self, error_type: str, workflow_node: str | None = None, **kwargs: Any
    ) -> None:
        """Log start of error handling process."""
        self.log(
            LogLevel.INFO,
            f"Starting error handling for {error_type}",
            event_type=EventType.ERROR_HANDLING_START,
            priority=Priority.HIGH,
            workflow_node=workflow_node,
            **kwargs,
        )

    def log_error_recovery_success(
        self, recovery_strategy: str, workflow_node: str | None = None, **kwargs: Any
    ) -> None:
        """Log successful error recovery."""
        self.log(
            LogLevel.INFO,
            f"Error recovery successful using strategy: {recovery_strategy}",
            event_type=EventType.ERROR_RECOVERY_SUCCESS,
            priority=Priority.HIGH,
            workflow_node=workflow_node,
            **kwargs,
        )

    def log_error_recovery_failed(
        self,
        recovery_strategy: str,
        reason: str,
        workflow_node: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Log failed error recovery."""
        self.log(
            LogLevel.ERROR,
            f"Error recovery failed using strategy {recovery_strategy}: {reason}",
            event_type=EventType.ERROR_RECOVERY_FAILED,
            priority=Priority.CRITICAL,
            workflow_node=workflow_node,
            **kwargs,
        )

    def log_error_classification(
        self, error_type: str, recovery_strategy: str, confidence: float, **kwargs: Any
    ) -> None:
        """Log error classification results."""
        metadata = kwargs.get("metadata", {})
        metadata.update(
            {
                "error_type": error_type,
                "recovery_strategy": recovery_strategy,
                "confidence": confidence,
            }
        )
        kwargs["metadata"] = metadata

        self.log(
            LogLevel.INFO,
            f"Error classified as {error_type} with strategy {recovery_strategy} (confidence: {confidence:.2f})",
            event_type=EventType.ERROR_CLASSIFICATION,
            priority=Priority.HIGH,
            **kwargs,
        )

    def log_circuit_breaker_state_change(
        self, component: str, old_state: str, new_state: str, **kwargs: Any
    ) -> None:
        """Log circuit breaker state changes."""
        metadata = kwargs.get("metadata", {})
        metadata.update(
            {"component": component, "old_state": old_state, "new_state": new_state}
        )
        kwargs["metadata"] = metadata

        if new_state == "OPEN":
            event_type = EventType.CIRCUIT_BREAKER_OPEN
            level = LogLevel.WARNING
            priority = Priority.HIGH
        elif new_state == "HALF_OPEN":
            event_type = EventType.CIRCUIT_BREAKER_HALF_OPEN
            level = LogLevel.INFO
            priority = Priority.NORMAL
        else:  # CLOSED
            event_type = EventType.CIRCUIT_BREAKER_CLOSE
            level = LogLevel.INFO
            priority = Priority.NORMAL

        self.log(
            level,
            f"Circuit breaker for {component} changed from {old_state} to {new_state}",
            event_type=event_type,
            priority=priority,
            component=component,
            **kwargs,
        )

    def log_fault_isolation(
        self, component: str, activated: bool, reason: str, **kwargs: Any
    ) -> None:
        """Log fault isolation activation/deactivation."""
        metadata = kwargs.get("metadata", {})
        metadata.update(
            {"component": component, "reason": reason, "activated": activated}
        )
        kwargs["metadata"] = metadata

        event_type = (
            EventType.FAULT_ISOLATION_ACTIVATED
            if activated
            else EventType.FAULT_ISOLATION_DEACTIVATED
        )
        action = "activated" if activated else "deactivated"

        self.log(
            LogLevel.WARNING if activated else LogLevel.INFO,
            f"Fault isolation {action} for {component}: {reason}",
            event_type=event_type,
            priority=Priority.HIGH,
            component=component,
            **kwargs,
        )

    def log_retry_attempt(
        self,
        attempt: int,
        max_attempts: int,
        delay_ms: float,
        workflow_node: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Log retry attempts."""
        metadata = kwargs.get("metadata", {})
        metadata.update(
            {"attempt": attempt, "max_attempts": max_attempts, "delay_ms": delay_ms}
        )
        kwargs["metadata"] = metadata

        self.log(
            LogLevel.INFO,
            f"Retry attempt {attempt}/{max_attempts} after {delay_ms:.0f}ms delay",
            event_type=EventType.RETRY_ATTEMPT,
            priority=Priority.NORMAL,
            workflow_node=workflow_node,
            **kwargs,
        )

    def log_retry_exhausted(
        self, total_attempts: int, workflow_node: str | None = None, **kwargs: Any
    ) -> None:
        """Log retry exhaustion."""
        metadata = kwargs.get("metadata", {})
        metadata.update({"total_attempts": total_attempts})
        kwargs["metadata"] = metadata

        self.log(
            LogLevel.ERROR,
            f"Retry attempts exhausted after {total_attempts} tries",
            event_type=EventType.RETRY_EXHAUSTED,
            priority=Priority.CRITICAL,
            workflow_node=workflow_node,
            **kwargs,
        )

    def log_error_escalation(
        self,
        escalation_level: str,
        reason: str,
        workflow_node: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Log error escalation."""
        metadata = kwargs.get("metadata", {})
        metadata.update({"escalation_level": escalation_level, "reason": reason})
        kwargs["metadata"] = metadata

        self.log(
            LogLevel.ERROR,
            f"Error escalated to {escalation_level}: {reason}",
            event_type=EventType.ERROR_ESCALATION,
            priority=Priority.CRITICAL,
            workflow_node=workflow_node,
            **kwargs,
        )

    def log_manual_intervention_required(
        self,
        intervention_type: str,
        instructions: str,
        workflow_node: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Log manual intervention requirement."""
        metadata = kwargs.get("metadata", {})
        metadata.update(
            {"intervention_type": intervention_type, "instructions": instructions}
        )
        kwargs["metadata"] = metadata

        self.log(
            LogLevel.CRITICAL,
            f"Manual intervention required: {intervention_type}",
            event_type=EventType.MANUAL_INTERVENTION_REQUIRED,
            priority=Priority.CRITICAL,
            workflow_node=workflow_node,
            **kwargs,
        )

    def log_error_rollback(
        self, rollback_point: str, workflow_node: str | None = None, **kwargs: Any
    ) -> None:
        """Log error rollback operation."""
        metadata = kwargs.get("metadata", {})
        metadata.update({"rollback_point": rollback_point})
        kwargs["metadata"] = metadata

        self.log(
            LogLevel.WARNING,
            f"Rolling back to {rollback_point}",
            event_type=EventType.ERROR_ROLLBACK,
            priority=Priority.HIGH,
            workflow_node=workflow_node,
            **kwargs,
        )

    def log_error_skip(
        self, skipped_operation: str, workflow_node: str | None = None, **kwargs: Any
    ) -> None:
        """Log error skip operation."""
        metadata = kwargs.get("metadata", {})
        metadata.update({"skipped_operation": skipped_operation})
        kwargs["metadata"] = metadata

        self.log(
            LogLevel.WARNING,
            f"Skipping operation: {skipped_operation}",
            event_type=EventType.ERROR_SKIP,
            priority=Priority.HIGH,
            workflow_node=workflow_node,
            **kwargs,
        )

    async def get_events(
        self, event_filter: EventFilter | None = None, limit: int | None = None
    ) -> list[StructuredLogEvent]:
        """Get stored events with optional filtering.

        Args:
            event_filter: Optional filter to apply
            limit: Maximum number of events to return

        Returns:
            List of matching events
        """
        async with self._events_lock:
            events = list(self._events)

        # Apply filter
        if event_filter:
            events = [event for event in events if event_filter.matches(event)]

        # Apply limit
        if limit:
            events = events[-limit:]

        return events

    async def get_logs(
        self, session_id: str | None = None, limit: int = 100
    ) -> list[str]:
        """Get log messages as strings.

        Args:
            session_id: Optional session ID to filter by
            limit: Maximum number of log messages to return

        Returns:
            List of log message strings
        """
        event_filter = None
        if session_id:
            event_filter = EventFilter(session_id=session_id)

        events = await self.get_events(event_filter, limit)
        return [f"[{event.level.name}] {event.message}" for event in events]

    def get_metrics(self) -> dict[str, Any]:
        """Get current performance metrics."""
        return {
            "total_events": self._metrics.total_events,
            "events_by_type": dict(self._metrics.events_by_type),
            "events_by_priority": dict(self._metrics.events_by_priority),
            "average_latency_ms": self._metrics.get_average_latency(),
            "p95_latency_ms": self._metrics.get_p95_latency(),
            "failed_deliveries": self._metrics.failed_deliveries,
            "rate_limited_events": self._metrics.rate_limited_events,
            "queue_size": self._event_queue.qsize(),
            "memory_events": len(self._events),
            "rate_limiting_enabled": self.enable_rate_limiting,
            "token_bucket_tokens": self._token_bucket.tokens
            if self._token_bucket
            else None,
        }

    def get_event_router(self) -> EventRouter:
        """Get the event router for advanced filtering configuration."""
        return self._event_router

    def clear_logs(self) -> None:
        """Clear all stored log events."""
        try:
            asyncio.create_task(self._clear_logs_async())
        except RuntimeError:
            # No event loop, clear synchronously
            self._events.clear()

    async def _clear_logs_async(self) -> None:
        """Async implementation of log clearing."""
        async with self._events_lock:
            self._events.clear()

    async def shutdown(self) -> None:
        """Shutdown event logger and cleanup resources."""
        # Cancel background tasks
        for task in [self._streaming_task, self._metrics_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass


class StructuredLogHandler(logging.Handler):
    """Custom logging handler that integrates with structured logging."""

    def __init__(self, event_logger: EventLogger):
        """Initialize handler.

        Args:
            event_logger: Event logger instance
        """
        super().__init__()
        self.event_logger = event_logger

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record as a structured event."""
        try:
            # Convert logging level to LogLevel
            level_map = {
                logging.DEBUG: LogLevel.DEBUG,
                logging.INFO: LogLevel.INFO,
                logging.WARNING: LogLevel.WARNING,
                logging.ERROR: LogLevel.ERROR,
                logging.CRITICAL: LogLevel.CRITICAL,
            }
            level = level_map.get(record.levelno, LogLevel.INFO)

            # Determine event type from logger name and content
            event_type = EventType.SYSTEM
            if "session" in record.name:
                event_type = EventType.SESSION
            elif "scene" in record.name or "scene" in record.getMessage().lower():
                event_type = EventType.SCENE_GENERATION
            elif "workflow" in record.name or "workflow" in record.getMessage().lower():
                event_type = EventType.WORKFLOW_NODE
            elif "ui" in record.name or "web" in record.name:
                event_type = EventType.UI_UPDATE
            elif record.levelno >= logging.ERROR:
                event_type = EventType.ERROR
            elif record.levelno >= logging.WARNING:
                event_type = EventType.WARNING

            # Determine priority based on level
            priority = Priority.NORMAL
            if record.levelno >= logging.CRITICAL:
                priority = Priority.CRITICAL
            elif record.levelno >= logging.ERROR:
                priority = Priority.CRITICAL
            elif record.levelno >= logging.WARNING:
                priority = Priority.HIGH
            elif record.levelno <= logging.DEBUG:
                priority = Priority.LOW

            # Create structured event
            event = StructuredLogEvent(
                level=level,
                event_type=event_type,
                priority=priority,
                message=self.format(record),
                component=record.name,
                metadata={
                    "filename": record.filename,
                    "lineno": record.lineno,
                    "funcName": record.funcName,
                    "pathname": record.pathname,
                },
            )

            # Store in memory directly to avoid recursion
            try:
                # Use internal storage to avoid recursive logging
                asyncio.create_task(
                    self.event_logger._log_event_internal(
                        event, bypass_queue=True, bypass_traditional=True
                    )
                )
            except RuntimeError:
                # No event loop, store in memory only using async context
                async def store_event():
                    await self.event_logger._log_event_internal(
                        event, bypass_queue=True, bypass_traditional=True
                    )

                # If an event loop is running, use it to run the task
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(store_event())
                except RuntimeError:
                    # Fallback for when no loop is running: run synchronously
                    asyncio.run(store_event())

        except Exception:
            # Avoid recursion in error handling
            self.handleError(record)


# Global event logger instance
_event_logger: EventLogger | None = None


def get_event_logger() -> EventLogger:
    """Get global event logger instance."""
    global _event_logger
    if _event_logger is None:
        _event_logger = EventLogger()
    return _event_logger


# Backward compatibility aliases
def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under the Chorus root logger.

    Ensures the root logger is configured once and returns a child logger
    to keep existing call sites working (nodes/graph expect get_logger()).
    """
    event_logger = get_event_logger()
    return event_logger._traditional_logger.getChild(name)


def log_message(message: str) -> None:
    """Store message in structured logging system."""
    event_logger = get_event_logger()
    event_logger.info(message, event_type=EventType.SYSTEM)


def log_calls(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorate func to log calls at the DEBUG level."""
    event_logger = get_event_logger()

    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            event_logger.debug(
                f"Entering {func.__qualname__}",
                component=func.__module__,
                metadata={
                    "function": func.__qualname__,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys()),
                },
            )

            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                event_logger.debug(
                    f"Exiting {func.__qualname__} successfully",
                    component=func.__module__,
                    metadata={
                        "function": func.__qualname__,
                        "duration_ms": duration_ms,
                    },
                )
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                event_logger.error(
                    f"Error in {func.__qualname__}: {e}",
                    component=func.__module__,
                    metadata={
                        "function": func.__qualname__,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "duration_ms": duration_ms,
                    },
                )
                raise

        return cast(Callable[..., Any], async_wrapper)

    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        event_logger.debug(
            f"Entering {func.__qualname__}",
            component=func.__module__,
            metadata={
                "function": func.__qualname__,
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys()),
            },
        )

        try:
            result = func(*args, **kwargs)
            duration_ms = (time.time() - start_time) * 1000
            event_logger.debug(
                f"Exiting {func.__qualname__} successfully",
                component=func.__module__,
                metadata={"function": func.__qualname__, "duration_ms": duration_ms},
            )
            return result
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            event_logger.error(
                f"Error in {func.__qualname__}: {e}",
                component=func.__module__,
                metadata={
                    "function": func.__qualname__,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "duration_ms": duration_ms,
                },
            )
            raise

    return cast(Callable[..., Any], sync_wrapper)


async def get_logs(session_id: str | None = None, limit: int = 100) -> list[str]:
    """Return the captured log messages."""
    event_logger = get_event_logger()
    return await event_logger.get_logs(session_id, limit)


def clear_logs() -> None:
    """Remove all stored log messages."""
    event_logger = get_event_logger()
    event_logger.clear_logs()


__all__ = [
    # Main classes
    "EventLogger",
    "StructuredLogEvent",
    "EventFilter",
    "EventRouter",
    "TokenBucket",
    "EventMetrics",
    "StructuredLogHandler",
    # Enums
    "LogLevel",
    "EventType",
    "Priority",
    # Functions
    "get_event_logger",
    "log_message",
    "get_logger",
    "get_logs",
    "clear_logs",
    "log_calls",
]
