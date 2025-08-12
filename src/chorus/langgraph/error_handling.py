# src/chorus/langgraph/error_handling.py
"""Comprehensive error handling and recovery system for LangGraph workflows.

This module provides production-grade fault tolerance including:
- Error classification and categorization
- Circuit breaker patterns with automatic recovery
- Granular recovery strategies per error type
- Fault isolation to prevent cascading failures
- Comprehensive error monitoring and reporting
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4

from chorus.config import config
from chorus.core.logs import (
    EventLogger,
    EventType,
    Priority,
    get_event_logger,
)


class ErrorType(Enum):
    """Classification of error types for different recovery strategies."""

    # Retriable errors - temporary issues that can be retried
    RETRIABLE = "retriable"

    # Skippable errors - non-critical failures that can be bypassed
    SKIPPABLE = "skippable"

    # Rollback errors - require state rollback and recovery
    ROLLBACK = "rollback"

    # Fatal errors - critical system failures requiring immediate attention
    FATAL = "fatal"

    # Manual errors - complex issues requiring human intervention
    MANUAL = "manual"


class RecoveryStrategy(Enum):
    """Recovery strategies for different error scenarios."""

    RETRY = "retry"  # Exponential backoff retry
    SKIP = "skip"  # Skip the failed operation
    ROLLBACK = "rollback"  # Rollback to previous state
    CIRCUIT_BREAK = "circuit_break"  # Open circuit breaker
    MANUAL_INTERVENTION = "manual"  # Require human input
    ESCALATE = "escalate"  # Escalate to higher-level handler


class CircuitBreakerState(Enum):
    """Circuit breaker states for fault tolerance."""

    CLOSED = "closed"  # Normal operation, all requests allowed
    OPEN = "open"  # Failure threshold exceeded, requests fail fast
    HALF_OPEN = "half_open"  # Testing recovery, limited requests allowed


@dataclass
class ErrorClassification:
    """Classification of an error with recovery metadata."""

    error_type: ErrorType
    recovery_strategy: RecoveryStrategy
    max_retries: int = 3
    backoff_multiplier: float = 2.0
    max_backoff: float = 60.0
    skip_allowed: bool = False
    rollback_required: bool = False
    escalation_threshold: int = 5

    @classmethod
    def for_network_error(cls) -> ErrorClassification:
        """Classification for network-related errors."""
        return cls(
            error_type=ErrorType.RETRIABLE,
            recovery_strategy=RecoveryStrategy.RETRY,
            max_retries=5,
            backoff_multiplier=1.5,
            max_backoff=30.0,
        )

    @classmethod
    def for_rate_limit_error(cls) -> ErrorClassification:
        """Classification for rate limiting errors."""
        return cls(
            error_type=ErrorType.RETRIABLE,
            recovery_strategy=RecoveryStrategy.RETRY,
            max_retries=3,
            backoff_multiplier=3.0,
            max_backoff=120.0,
        )

    @classmethod
    def for_validation_error(cls) -> ErrorClassification:
        """Classification for data validation errors."""
        return cls(
            error_type=ErrorType.SKIPPABLE,
            recovery_strategy=RecoveryStrategy.SKIP,
            skip_allowed=True,
        )

    @classmethod
    def for_authentication_error(cls) -> ErrorClassification:
        """Classification for authentication errors."""
        return cls(
            error_type=ErrorType.FATAL,
            recovery_strategy=RecoveryStrategy.ESCALATE,
            max_retries=0,
        )

    @classmethod
    def for_data_corruption_error(cls) -> ErrorClassification:
        """Classification for data corruption errors."""
        return cls(
            error_type=ErrorType.ROLLBACK,
            recovery_strategy=RecoveryStrategy.ROLLBACK,
            rollback_required=True,
        )

    @classmethod
    def for_unknown_error(cls) -> ErrorClassification:
        """Default classification for unknown errors."""
        return cls(
            error_type=ErrorType.MANUAL,
            recovery_strategy=RecoveryStrategy.MANUAL_INTERVENTION,
        )


@dataclass
class ErrorContext:
    """Context information for error handling and recovery."""

    error_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: float = field(default_factory=time.time)
    node_name: str = ""
    session_id: str | None = None
    thread_id: str | None = None
    scene_id: str | None = None
    error_message: str = ""
    error_type_name: str = ""
    traceback_str: str = ""
    classification: ErrorClassification | None = None
    attempt_count: int = 0
    total_attempts: int = 0
    recovery_actions: list[str] = field(default_factory=list)
    escalation_count: int = 0
    circuit_breaker_triggered: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging and monitoring."""
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp,
            "node_name": self.node_name,
            "session_id": self.session_id,
            "thread_id": self.thread_id,
            "scene_id": self.scene_id,
            "error_message": self.error_message,
            "error_type_name": self.error_type_name,
            "traceback": self.traceback_str,
            "classification": {
                "error_type": self.classification.error_type.value,
                "recovery_strategy": self.classification.recovery_strategy.value,
                "max_retries": self.classification.max_retries,
            }
            if self.classification
            else None,
            "attempt_count": self.attempt_count,
            "total_attempts": self.total_attempts,
            "recovery_actions": self.recovery_actions,
            "escalation_count": self.escalation_count,
            "circuit_breaker_triggered": self.circuit_breaker_triggered,
        }


class ErrorClassifier:
    """Classifies errors and determines appropriate recovery strategies."""

    def __init__(self):
        """Initialize error classifier with predefined rules."""
        self._classification_rules: dict[str, ErrorClassification] = {}
        self._type_patterns: dict[type[Exception], ErrorClassification] = {}
        self._message_patterns: dict[str, ErrorClassification] = {}

        self._setup_default_rules()

    def _setup_default_rules(self) -> None:
        """Setup default classification rules for common error patterns."""

        # Network and connection errors
        network_errors = [
            "ConnectionError",
            "TimeoutError",
            "ConnectTimeoutError",
            "ReadTimeoutError",
            "ConnectionResetError",
            "DNSLookupError",
        ]
        for error_name in network_errors:
            self._classification_rules[error_name] = (
                ErrorClassification.for_network_error()
            )

        # Rate limiting errors
        rate_limit_errors = [
            "RateLimitError",
            "TooManyRequestsError",
            "QuotaExceededError",
        ]
        for error_name in rate_limit_errors:
            self._classification_rules[error_name] = (
                ErrorClassification.for_rate_limit_error()
            )

        # Authentication and authorization errors
        auth_errors = ["AuthenticationError", "UnauthorizedError", "PermissionError"]
        for error_name in auth_errors:
            self._classification_rules[error_name] = (
                ErrorClassification.for_authentication_error()
            )

        # Validation errors
        validation_errors = ["ValidationError", "ValueError", "TypeError", "KeyError"]
        for error_name in validation_errors:
            self._classification_rules[error_name] = (
                ErrorClassification.for_validation_error()
            )

        # Data corruption errors
        corruption_errors = ["DataCorruptionError", "IntegrityError", "ChecksumError"]
        for error_name in corruption_errors:
            self._classification_rules[error_name] = (
                ErrorClassification.for_data_corruption_error()
            )

        # Message pattern-based classification
        self._message_patterns.update(
            {
                "rate limit": ErrorClassification.for_rate_limit_error(),
                "timeout": ErrorClassification.for_network_error(),
                "connection": ErrorClassification.for_network_error(),
                "authentication": ErrorClassification.for_authentication_error(),
                "unauthorized": ErrorClassification.for_authentication_error(),
                "validation": ErrorClassification.for_validation_error(),
                "invalid": ErrorClassification.for_validation_error(),
            }
        )

    def classify_error(
        self, error: Exception, context: ErrorContext
    ) -> ErrorClassification:
        """Classify an error and return appropriate recovery strategy.

        Args:
            error: The exception to classify
            context: Error context information

        Returns:
            ErrorClassification with recovery strategy
        """
        error_type_name = type(error).__name__
        error_message = str(error).lower()

        # Check exact type match
        if error_type_name in self._classification_rules:
            return self._classification_rules[error_type_name]

        # Check type inheritance
        for exception_type, classification in self._type_patterns.items():
            if isinstance(error, exception_type):
                return classification

        # Check message patterns
        for pattern, classification in self._message_patterns.items():
            if pattern in error_message:
                return classification

        # Check for specific LangGraph errors
        if "GraphInterrupt" in error_type_name:
            return ErrorClassification(
                error_type=ErrorType.MANUAL,
                recovery_strategy=RecoveryStrategy.MANUAL_INTERVENTION,
            )

        # Default to unknown error classification
        return ErrorClassification.for_unknown_error()

    def register_classification_rule(
        self, error_type_name: str, classification: ErrorClassification
    ) -> None:
        """Register a custom classification rule.

        Args:
            error_type_name: Name of the error type
            classification: Classification to apply
        """
        self._classification_rules[error_type_name] = classification

    def register_type_pattern(
        self, exception_type: type[Exception], classification: ErrorClassification
    ) -> None:
        """Register a classification for an exception type.

        Args:
            exception_type: Exception type to match
            classification: Classification to apply
        """
        self._type_patterns[exception_type] = classification


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker monitoring."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0
    state_changes: int = 0
    time_in_open_state: float = 0.0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0

    @property
    def failure_rate(self) -> float:
        """Calculate current failure rate."""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests

    @property
    def success_rate(self) -> float:
        """Calculate current success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests


class CircuitBreaker:
    """Circuit breaker implementation with automatic recovery and monitoring."""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        half_open_max_calls: int = 3,
        monitoring_window: float = 300.0,
        logger: EventLogger | None = None,
    ):
        """Initialize circuit breaker.

        Args:
            name: Circuit breaker identifier
            failure_threshold: Number of failures to open circuit
            timeout: Time to wait before attempting recovery (seconds)
            half_open_max_calls: Max calls allowed in half-open state
            monitoring_window: Time window for failure rate calculation
            logger: Optional event logger
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.half_open_max_calls = half_open_max_calls
        self.monitoring_window = monitoring_window
        self.logger = logger or get_event_logger()

        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.half_open_call_count = 0
        self.metrics = CircuitBreakerMetrics()

        # Recent failures for time-based monitoring
        self.recent_failures: deque[float] = deque()

        # Lock for thread safety
        self._lock = asyncio.Lock()

    async def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerOpenError: When circuit is open
            Exception: Original function exception
        """
        async with self._lock:
            await self._check_state()

            if self.state == CircuitBreakerState.OPEN:
                self.metrics.rejected_requests += 1
                self.logger.error(
                    f"Circuit breaker {self.name} is OPEN - rejecting call",
                    component="circuit_breaker",
                    metadata={
                        "circuit_breaker": self.name,
                        "state": self.state.value,
                        "failure_count": self.failure_count,
                        "failure_rate": self.metrics.failure_rate,
                    },
                )
                raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")

            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.half_open_call_count >= self.half_open_max_calls:
                    self.metrics.rejected_requests += 1
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker {self.name} half-open limit exceeded"
                    )

                self.half_open_call_count += 1

        # Execute function outside of lock to avoid blocking
        try:
            self.metrics.total_requests += 1
            start_time = time.time()

            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Record success
            async with self._lock:
                await self._record_success()

            duration = time.time() - start_time
            self.logger.debug(
                f"Circuit breaker {self.name} - successful call",
                component="circuit_breaker",
                metadata={
                    "circuit_breaker": self.name,
                    "duration_ms": duration * 1000,
                    "state": self.state.value,
                },
            )

            return result

        except Exception as e:
            async with self._lock:
                await self._record_failure()

            self.logger.error(
                f"Circuit breaker {self.name} - call failed: {e}",
                component="circuit_breaker",
                metadata={
                    "circuit_breaker": self.name,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "state": self.state.value,
                    "failure_count": self.failure_count,
                },
            )

            raise

    async def _check_state(self) -> None:
        """Check and update circuit breaker state."""
        current_time = time.time()

        # Clean up old failures outside monitoring window
        while (
            self.recent_failures
            and (current_time - self.recent_failures[0]) > self.monitoring_window
        ):
            self.recent_failures.popleft()

        if self.state == CircuitBreakerState.OPEN:
            # Check if timeout has elapsed
            if current_time - self.last_failure_time >= self.timeout:
                await self._transition_to_half_open()

        elif self.state == CircuitBreakerState.CLOSED:
            # Check if failure threshold exceeded
            if (
                len(self.recent_failures) >= self.failure_threshold
                or self.failure_count >= self.failure_threshold
            ):
                await self._transition_to_open()

    async def _record_success(self) -> None:
        """Record successful operation."""
        self.metrics.successful_requests += 1
        self.metrics.last_success_time = time.time()

        if self.state == CircuitBreakerState.HALF_OPEN:
            # Successful call in half-open state - transition to closed
            await self._transition_to_closed()

        # Gradually reduce failure count on success
        self.failure_count = max(0, self.failure_count - 1)

    async def _record_failure(self) -> None:
        """Record failed operation."""
        current_time = time.time()

        self.metrics.failed_requests += 1
        self.metrics.last_failure_time = current_time
        self.last_failure_time = current_time
        self.failure_count += 1
        self.recent_failures.append(current_time)

        if self.state == CircuitBreakerState.HALF_OPEN:
            # Failure in half-open state - return to open
            await self._transition_to_open()
        elif self.state == CircuitBreakerState.CLOSED:
            # Check if we should open the circuit
            if (
                len(self.recent_failures) >= self.failure_threshold
                or self.failure_count >= self.failure_threshold
            ):
                await self._transition_to_open()

    async def _transition_to_open(self) -> None:
        """Transition circuit breaker to OPEN state."""
        if self.state != CircuitBreakerState.OPEN:
            previous_state = self.state
            self.state = CircuitBreakerState.OPEN
            self.metrics.state_changes += 1

            self.logger.warning(
                f"Circuit breaker {self.name} transitioned to OPEN",
                component="circuit_breaker",
                event_type=EventType.WARNING,
                priority=Priority.HIGH,
                metadata={
                    "circuit_breaker": self.name,
                    "previous_state": previous_state.value,
                    "new_state": self.state.value,
                    "failure_count": self.failure_count,
                    "recent_failures": len(self.recent_failures),
                    "failure_threshold": self.failure_threshold,
                },
            )

    async def _transition_to_half_open(self) -> None:
        """Transition circuit breaker to HALF_OPEN state."""
        previous_state = self.state
        self.state = CircuitBreakerState.HALF_OPEN
        self.half_open_call_count = 0
        self.metrics.state_changes += 1

        self.logger.info(
            f"Circuit breaker {self.name} transitioned to HALF_OPEN",
            component="circuit_breaker",
            metadata={
                "circuit_breaker": self.name,
                "previous_state": previous_state.value,
                "new_state": self.state.value,
                "timeout_elapsed": time.time() - self.last_failure_time,
            },
        )

    async def _transition_to_closed(self) -> None:
        """Transition circuit breaker to CLOSED state."""
        previous_state = self.state
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.half_open_call_count = 0
        self.recent_failures.clear()
        self.metrics.state_changes += 1

        self.logger.info(
            f"Circuit breaker {self.name} transitioned to CLOSED",
            component="circuit_breaker",
            metadata={
                "circuit_breaker": self.name,
                "previous_state": previous_state.value,
                "new_state": self.state.value,
                "recovery_successful": True,
            },
        )

    def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self.state

    def get_metrics(self) -> CircuitBreakerMetrics:
        """Get circuit breaker metrics."""
        return self.metrics

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.half_open_call_count = 0
        self.recent_failures.clear()
        self.metrics = CircuitBreakerMetrics()

        self.logger.info(
            f"Circuit breaker {self.name} reset",
            component="circuit_breaker",
            metadata={"circuit_breaker": self.name},
        )


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""

    pass


class RetryManager:
    """Manages retry logic with exponential backoff and jitter."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 0.5,
        max_delay: float = 60.0,
        backoff_multiplier: float = 2.0,
        jitter: bool = True,
        logger: EventLogger | None = None,
    ):
        """Initialize retry manager.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay between retries (seconds)
            max_delay: Maximum delay between retries (seconds)
            backoff_multiplier: Multiplier for exponential backoff
            jitter: Whether to add random jitter to delays
            logger: Optional event logger
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        self.jitter = jitter
        self.logger = logger or get_event_logger()

    async def retry_with_backoff(
        self,
        func: Callable[..., Any],
        error_context: ErrorContext,
        classification: ErrorClassification,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute function with retry and exponential backoff.

        Args:
            func: Function to execute
            error_context: Error context for tracking
            classification: Error classification with retry settings
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Exception: Last exception if all retries failed
        """
        max_attempts = classification.max_retries
        base_delay = self.base_delay
        max_delay = min(self.max_delay, classification.max_backoff)
        multiplier = classification.backoff_multiplier

        last_exception = None

        for attempt in range(max_attempts + 1):
            error_context.attempt_count = attempt
            error_context.total_attempts = max_attempts + 1

            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                if attempt > 0:
                    self.logger.info(
                        f"Retry successful after {attempt} attempts",
                        component="retry_manager",
                        metadata={
                            "function": getattr(func, "__name__", "unknown"),
                            "attempts": attempt + 1,
                            "total_duration": time.time() - error_context.timestamp,
                        },
                    )

                return result

            except Exception as e:
                last_exception = e

                if attempt < max_attempts:
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (multiplier**attempt), max_delay)

                    # Add jitter if enabled
                    if self.jitter:
                        import random

                        delay *= 0.5 + random.random() * 0.5

                    self.logger.warning(
                        f"Retry attempt {attempt + 1}/{max_attempts + 1} failed, retrying in {delay:.2f}s",
                        component="retry_manager",
                        metadata={
                            "function": getattr(func, "__name__", "unknown"),
                            "attempt": attempt + 1,
                            "max_attempts": max_attempts + 1,
                            "delay_seconds": delay,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        },
                    )

                    await asyncio.sleep(delay)
                else:
                    self.logger.error(
                        f"All {max_attempts + 1} retry attempts failed",
                        component="retry_manager",
                        metadata={
                            "function": getattr(func, "__name__", "unknown"),
                            "total_attempts": max_attempts + 1,
                            "final_error": str(e),
                            "error_type": type(e).__name__,
                        },
                    )

        # All retries failed
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError("Retry failed without exception")


@dataclass
class FaultIsolationMetrics:
    """Metrics for fault isolation monitoring."""

    isolated_components: set[str] = field(default_factory=set)
    isolation_events: int = 0
    recovery_events: int = 0
    cascade_prevention_count: int = 0
    total_isolation_time: float = 0.0


class FaultIsolation:
    """Fault isolation system to prevent cascading failures."""

    def __init__(self, logger: EventLogger | None = None):
        """Initialize fault isolation system.

        Args:
            logger: Optional event logger
        """
        self.logger = logger or get_event_logger()
        self.metrics = FaultIsolationMetrics()

        # Component isolation tracking
        self._isolated_components: dict[
            str, float
        ] = {}  # component -> isolation_timestamp
        self._component_circuit_breakers: dict[str, CircuitBreaker] = {}
        self._failure_counts: dict[str, int] = defaultdict(int)

        # Configuration
        self.isolation_threshold = config.concurrency.circuit_breaker_threshold
        self.isolation_timeout = config.concurrency.circuit_breaker_timeout

    async def isolate_component(self, component_name: str, reason: str = "") -> None:
        """Isolate a component to prevent cascading failures.

        Args:
            component_name: Name of component to isolate
            reason: Reason for isolation
        """
        if component_name not in self._isolated_components:
            self._isolated_components[component_name] = time.time()
            self.metrics.isolated_components.add(component_name)
            self.metrics.isolation_events += 1

            self.logger.warning(
                f"Component {component_name} isolated due to failures",
                component="fault_isolation",
                event_type=EventType.WARNING,
                priority=Priority.HIGH,
                metadata={
                    "component": component_name,
                    "reason": reason,
                    "isolation_timestamp": time.time(),
                    "failure_count": self._failure_counts[component_name],
                },
            )

    async def recover_component(self, component_name: str) -> None:
        """Recover an isolated component.

        Args:
            component_name: Name of component to recover
        """
        if component_name in self._isolated_components:
            isolation_start = self._isolated_components.pop(component_name)
            self.metrics.isolated_components.discard(component_name)
            self.metrics.recovery_events += 1
            self.metrics.total_isolation_time += time.time() - isolation_start

            # Reset failure count
            self._failure_counts[component_name] = 0

            self.logger.info(
                f"Component {component_name} recovered from isolation",
                component="fault_isolation",
                metadata={
                    "component": component_name,
                    "isolation_duration": time.time() - isolation_start,
                    "recovery_timestamp": time.time(),
                },
            )

    async def check_isolation_status(self, component_name: str) -> bool:
        """Check if a component is currently isolated.

        Args:
            component_name: Name of component to check

        Returns:
            True if component is isolated
        """
        if component_name not in self._isolated_components:
            return False

        # Check if isolation timeout has elapsed
        isolation_start = self._isolated_components[component_name]
        if time.time() - isolation_start >= self.isolation_timeout:
            await self.recover_component(component_name)
            return False

        return True

    async def record_component_failure(
        self, component_name: str, error: Exception
    ) -> bool:
        """Record a component failure and check if isolation is needed.

        Args:
            component_name: Name of failed component
            error: Exception that occurred

        Returns:
            True if component was isolated
        """
        self._failure_counts[component_name] += 1

        if self._failure_counts[component_name] >= self.isolation_threshold:
            await self.isolate_component(
                component_name,
                f"Failure threshold exceeded: {self._failure_counts[component_name]} failures",
            )
            self.metrics.cascade_prevention_count += 1
            return True

        return False

    async def get_circuit_breaker(self, component_name: str) -> CircuitBreaker:
        """Get or create a circuit breaker for a component.

        Args:
            component_name: Name of component

        Returns:
            CircuitBreaker instance for the component
        """
        if component_name not in self._component_circuit_breakers:
            self._component_circuit_breakers[component_name] = CircuitBreaker(
                name=f"{component_name}_circuit_breaker",
                failure_threshold=self.isolation_threshold,
                timeout=self.isolation_timeout,
                logger=self.logger,
            )

        return self._component_circuit_breakers[component_name]

    def get_metrics(self) -> FaultIsolationMetrics:
        """Get fault isolation metrics."""
        return self.metrics


__all__ = [
    "ErrorType",
    "RecoveryStrategy",
    "CircuitBreakerState",
    "ErrorClassification",
    "ErrorContext",
    "ErrorClassifier",
    "CircuitBreaker",
    "CircuitBreakerOpenError",
    "RetryManager",
    "FaultIsolation",
    "CircuitBreakerMetrics",
    "FaultIsolationMetrics",
]
