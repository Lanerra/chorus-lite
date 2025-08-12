# src/chorus/langgraph/error_recovery.py
"""ErrorRecovery node and comprehensive error handling integration for LangGraph workflows.

This module provides the ErrorRecovery node that serves as the central error handling
component in the workflow, orchestrating various recovery strategies and maintaining
fault tolerance across the entire system.
"""

from __future__ import annotations

import time
import traceback
from typing import Any

from chorus.config import config
from chorus.core.logs import (
    EventLogger,
    get_event_logger,
)

from .error_handling import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    ErrorClassification,
    ErrorClassifier,
    ErrorContext,
    FaultIsolation,
    RecoveryStrategy,
    RetryManager,
)
from .state import StoryState


class ErrorRecoveryManager:
    """Central manager for error recovery orchestration in LangGraph workflows."""

    def __init__(self, logger: EventLogger | None = None):
        """Initialize error recovery manager.

        Args:
            logger: Optional event logger
        """
        self.logger = logger or get_event_logger()

        # Core components
        self.error_classifier = ErrorClassifier()
        self.retry_manager = RetryManager(
            max_retries=config.retry.retry_attempts,
            base_delay=config.retry.retry_backoff,
            max_delay=config.retry.retry_max_interval,
            logger=self.logger,
        )
        self.fault_isolation = FaultIsolation(logger=self.logger)

        # Circuit breakers per component
        self._circuit_breakers: dict[str, CircuitBreaker] = {}

        # Error statistics
        self._error_stats = {
            "total_errors": 0,
            "errors_by_type": {},
            "errors_by_node": {},
            "recovery_success_rate": 0.0,
            "circuit_breaker_activations": 0,
            "manual_interventions": 0,
        }

    async def handle_node_error(
        self,
        error: Exception,
        node_name: str,
        state: StoryState,
        original_function: callable,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Handle an error from a workflow node with comprehensive recovery.

        Args:
            error: The exception that occurred
            node_name: Name of the node that failed
            state: Current workflow state
            original_function: The original function that failed
            *args: Original function arguments
            **kwargs: Original function keyword arguments

        Returns:
            Recovery result with updated state
        """
        # Create error context
        error_context = ErrorContext(
            node_name=node_name,
            session_id=state.get("session_id"),
            thread_id=state.get("thread_id"),
            scene_id=state.get("current_scene_id"),
            error_message=str(error),
            error_type_name=type(error).__name__,
            traceback_str=traceback.format_exc(),
        )

        # Update statistics
        self._error_stats["total_errors"] += 1
        self._error_stats["errors_by_node"][node_name] = (
            self._error_stats["errors_by_node"].get(node_name, 0) + 1
        )
        self._error_stats["errors_by_type"][type(error).__name__] = (
            self._error_stats["errors_by_type"].get(type(error).__name__, 0) + 1
        )

        # Log the start of error handling
        self.logger.log_error_handling_start(
            error_type=type(error).__name__,
            workflow_node=node_name,
            session_id=error_context.session_id,
            thread_id=error_context.thread_id,
            scene_id=error_context.scene_id,
            metadata=error_context.to_dict(),
        )

        # Classify the error
        classification = self.error_classifier.classify_error(error, error_context)
        error_context.classification = classification

        # Log error classification with structured logging
        self.logger.log_error_classification(
            error_type=classification.error_type.value,
            recovery_strategy=classification.recovery_strategy.value,
            confidence=1.0,  # High confidence for rule-based classification
            workflow_node=node_name,
            session_id=error_context.session_id,
            thread_id=error_context.thread_id,
            scene_id=error_context.scene_id,
            metadata={
                "max_retries": classification.max_retries,
                "skip_allowed": classification.skip_allowed,
                "rollback_required": classification.rollback_required,
                "backoff_multiplier": classification.backoff_multiplier,
                "max_backoff": classification.max_backoff,
            },
            # Legacy status checks removed in Phase 4 cleanup
            # These were replaced by simplified APPROVED status handling
        )

        # Check if component should be isolated
        component_isolated = await self.fault_isolation.record_component_failure(
            node_name, error
        )
        if component_isolated:
            error_context.circuit_breaker_triggered = True

        # Execute recovery strategy
        try:
            result = await self._execute_recovery_strategy(
                classification, error_context, state, original_function, *args, **kwargs
            )

            # Log successful error recovery
            self.logger.log_error_recovery_success(
                recovery_strategy=classification.recovery_strategy.value,
                workflow_node=node_name,
                session_id=error_context.session_id,
                thread_id=error_context.thread_id,
                scene_id=error_context.scene_id,
                metadata={
                    "attempts": error_context.total_attempts,
                    "recovery_actions": error_context.recovery_actions,
                    "error_id": error_context.error_id,
                    "recovery_duration": time.time() - error_context.timestamp,
                },
            )

            return result

        except Exception as recovery_error:
            self.logger.error(
                f"Error recovery failed for node {node_name}: {recovery_error}",
                component="error_recovery",
                workflow_node=node_name,
                metadata={
                    "original_error": str(error),
                    "recovery_error": str(recovery_error),
                    "recovery_strategy": classification.recovery_strategy.value,
                },
            )

            # Escalate to manual intervention
            return await self._handle_recovery_failure(
                error_context, state, recovery_error
            )

    async def _execute_recovery_strategy(
        self,
        classification: ErrorClassification,
        error_context: ErrorContext,
        state: StoryState,
        original_function: callable,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute the appropriate recovery strategy.

        Args:
            classification: Error classification with recovery strategy
            error_context: Error context information
            state: Current workflow state
            original_function: Original function that failed
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Recovery result
        """
        strategy = classification.recovery_strategy
        error_context.recovery_actions.append(f"executing_{strategy.value}_strategy")

        if strategy == RecoveryStrategy.RETRY:
            return await self._handle_retry_strategy(
                classification, error_context, state, original_function, *args, **kwargs
            )

        elif strategy == RecoveryStrategy.SKIP:
            return await self._handle_skip_strategy(error_context, state)

        elif strategy == RecoveryStrategy.ROLLBACK:
            return await self._handle_rollback_strategy(error_context, state)

        elif strategy == RecoveryStrategy.CIRCUIT_BREAK:
            return await self._handle_circuit_break_strategy(error_context, state)

        elif strategy == RecoveryStrategy.MANUAL_INTERVENTION:
            return await self._handle_manual_intervention_strategy(error_context, state)

        elif strategy == RecoveryStrategy.ESCALATE:
            return await self._handle_escalation_strategy(error_context, state)

        else:
            # Default to manual intervention for unknown strategies
            return await self._handle_manual_intervention_strategy(error_context, state)

    async def _handle_retry_strategy(
        self,
        classification: ErrorClassification,
        error_context: ErrorContext,
        state: StoryState,
        original_function: callable,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Handle retry recovery strategy with exponential backoff.

        Args:
            classification: Error classification
            error_context: Error context
            state: Current state
            original_function: Function to retry
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Retry result
        """
        error_context.recovery_actions.append("retry_with_exponential_backoff")

        try:
            # Get circuit breaker for this node
            circuit_breaker = await self.fault_isolation.get_circuit_breaker(
                error_context.node_name
            )

            # Execute with circuit breaker protection
            result = await circuit_breaker.call(
                self.retry_manager.retry_with_backoff,
                original_function,
                error_context,
                classification,
                *args,
                **kwargs,
            )

            # If successful, return the result as a proper node response
            if isinstance(result, dict):
                return result
            else:
                return {"retry_result": result, "retry_successful": True}

        except CircuitBreakerOpenError:
            error_context.circuit_breaker_triggered = True
            self._error_stats["circuit_breaker_activations"] += 1
            return await self._handle_circuit_break_strategy(error_context, state)

        except Exception as retry_error:
            # Retry failed, escalate to next strategy
            error_context.recovery_actions.append(
                f"retry_failed_{type(retry_error).__name__}"
            )

            # Try skip if allowed, otherwise manual intervention
            if classification.skip_allowed:
                return await self._handle_skip_strategy(error_context, state)
            else:
                return await self._handle_manual_intervention_strategy(
                    error_context, state
                )

    async def _handle_skip_strategy(
        self, error_context: ErrorContext, state: StoryState
    ) -> dict[str, Any]:
        """Handle skip recovery strategy.

        Args:
            error_context: Error context
            state: Current state

        Returns:
            Skip result
        """
        error_context.recovery_actions.append("skip_failed_operation")

        self.logger.warning(
            f"Skipping failed operation in node {error_context.node_name}",
            component="error_recovery",
            workflow_node=error_context.node_name,
            metadata={"skip_reason": "non_critical_failure_skip_allowed"},
        )

        # Return empty result to continue workflow
        return {
            "error_recovery_action": "skip",
            "skipped_node": error_context.node_name,
            "skip_reason": error_context.error_message,
            "workflow_continues": True,
        }

    async def _handle_rollback_strategy(
        self, error_context: ErrorContext, state: StoryState
    ) -> dict[str, Any]:
        """Handle rollback recovery strategy.

        Args:
            error_context: Error context
            state: Current state

        Returns:
            Rollback result
        """
        error_context.recovery_actions.append("rollback_to_previous_state")

        self.logger.warning(
            f"Rolling back state due to error in node {error_context.node_name}",
            component="error_recovery",
            workflow_node=error_context.node_name,
            metadata={"rollback_reason": "data_corruption_or_invalid_state"},
        )

        # Clear error-prone state fields
        rollback_state = {
            "error_recovery_action": "rollback",
            "rollback_node": error_context.node_name,
            "rollback_reason": error_context.error_message,
            # Clear current scene state if it's corrupted
            "current_scene_id": None,
            # Reset error fields
            "error_messages": [],
            "error_nodes": [],
            # Reset draft if it's corrupted
            "needs_revision": True,
            "revision_count": state.get("revision_count", 0) + 1,
        }

        # If we have a scene ID, clear its state
        if error_context.scene_id:
            scene_states = state.get("scene_states", {})
            if error_context.scene_id in scene_states:
                scene_states[error_context.scene_id] = {
                    "needs_revision": True,
                    "draft": None,
                    "review_status": [],
                    "feedback": [
                        f"Rolled back due to error: {error_context.error_message}"
                    ],
                }
                rollback_state["scene_states"] = scene_states

        return rollback_state

    async def _handle_circuit_break_strategy(
        self, error_context: ErrorContext, state: StoryState
    ) -> dict[str, Any]:
        """Handle circuit breaker strategy.

        Args:
            error_context: Error context
            state: Current state

        Returns:
            Circuit break result
        """
        error_context.recovery_actions.append("circuit_breaker_activated")

        # Isolate the component
        await self.fault_isolation.isolate_component(
            error_context.node_name,
            f"Circuit breaker activated: {error_context.error_message}",
        )

        self.logger.error(
            f"Circuit breaker activated for node {error_context.node_name}",
            component="error_recovery",
            workflow_node=error_context.node_name,
            metadata={
                "isolation_reason": "circuit_breaker_threshold_exceeded",
                "failure_count": self._error_stats["errors_by_node"].get(
                    error_context.node_name, 0
                ),
            },
        )

        # Try to continue workflow with degraded functionality
        return {
            "error_recovery_action": "circuit_break",
            "isolated_component": error_context.node_name,
            "circuit_break_reason": error_context.error_message,
            "degraded_mode": True,
            # Skip this node and continue if possible
            "workflow_continues": True,
            "skip_isolated_nodes": [error_context.node_name],
        }

    async def _handle_manual_intervention_strategy(
        self, error_context: ErrorContext, state: StoryState
    ) -> dict[str, Any]:
        """Handle manual intervention strategy.

        Args:
            error_context: Error context
            state: Current state

        Returns:
            Manual intervention result
        """
        error_context.recovery_actions.append("manual_intervention_required")
        self._error_stats["manual_interventions"] += 1

        self.logger.critical(
            f"Manual intervention required for node {error_context.node_name}",
            component="error_recovery",
            workflow_node=error_context.node_name,
            metadata={
                "intervention_reason": "complex_error_requires_human_review",
                "error_context": error_context.to_dict(),
            },
        )

        # Removed interactive handling - use deterministic fallbacks
        # For non-interactive execution, always retry first before other strategies

        error_context.recovery_actions.append("automatic_retry_requested")
        return {"manual_action": "retry", "retry_requested": True}

    async def _handle_escalation_strategy(
        self, error_context: ErrorContext, state: StoryState
    ) -> dict[str, Any]:
        """Handle escalation strategy.

        Args:
            error_context: Error context
            state: Current state

        Returns:
            Escalation result
        """
        error_context.recovery_actions.append("escalated_to_system_admin")
        error_context.escalation_count += 1

        self.logger.critical(
            f"Error escalated for node {error_context.node_name}",
            component="error_recovery",
            workflow_node=error_context.node_name,
            metadata={
                "escalation_level": error_context.escalation_count,
                "escalation_reason": "critical_system_error",
                "requires_immediate_attention": True,
            },
        )

        # For critical errors like authentication failures, stop the workflow
        return {
            "error_recovery_action": "escalate",
            "escalation_level": error_context.escalation_count,
            "escalation_reason": error_context.error_message,
            "requires_immediate_attention": True,
            "workflow_terminated": True,
            "system_admin_notified": True,
        }

    async def _handle_recovery_failure(
        self, error_context: ErrorContext, state: StoryState, recovery_error: Exception
    ) -> dict[str, Any]:
        """Handle failure of recovery strategy itself.

        Args:
            error_context: Original error context
            state: Current state
            recovery_error: Error during recovery

        Returns:
            Recovery failure result
        """
        error_context.recovery_actions.append(
            f"recovery_failed_{type(recovery_error).__name__}"
        )

        self.logger.critical(
            f"Recovery strategy failed for node {error_context.node_name}: {recovery_error}",
            component="error_recovery",
            workflow_node=error_context.node_name,
            metadata={
                "original_error": error_context.error_message,
                "recovery_error": str(recovery_error),
                "recovery_actions_attempted": error_context.recovery_actions,
            },
        )

        # Last resort - manual intervention
        return await self._handle_manual_intervention_strategy(error_context, state)

    def get_error_statistics(self) -> dict[str, Any]:
        """Get comprehensive error statistics.

        Returns:
            Error statistics dictionary
        """
        total_errors = self._error_stats["total_errors"]
        total_recoveries = sum(
            len(self._error_stats.get("recovery_actions", {}).get(node, []))
            for node in self._error_stats["errors_by_node"]
        )

        return {
            **self._error_stats,
            "recovery_success_rate": (total_recoveries / total_errors)
            if total_errors > 0
            else 0.0,
            "fault_isolation_metrics": self.fault_isolation.get_metrics(),
            "circuit_breakers": {
                name: cb.get_metrics() for name, cb in self._circuit_breakers.items()
            },
        }


# Global error recovery manager instance
_error_recovery_manager: ErrorRecoveryManager | None = None


def get_error_recovery_manager() -> ErrorRecoveryManager:
    """Get global error recovery manager instance."""
    global _error_recovery_manager
    if _error_recovery_manager is None:
        _error_recovery_manager = ErrorRecoveryManager()
    return _error_recovery_manager


async def error_recovery_node(state: StoryState) -> dict[str, Any]:
    """LangGraph node for comprehensive error recovery.

    This node serves as the central error handling component in the workflow,
    providing comprehensive error recovery, fault isolation, and monitoring.

    Args:
        state: Current workflow state

    Returns:
        Recovery result with updated state
    """
    recovery_manager = get_error_recovery_manager()

    # Extract error information from state
    error_messages = state.get("error_messages", [])
    error_nodes = state.get("error_nodes", [])

    if not error_messages or not error_nodes:
        # No errors to recover from
        return {"error_recovery_action": "no_errors", "workflow_continues": True}

    # Get the most recent error
    last_error_message = error_messages[-1] if error_messages else "Unknown error"
    last_error_node = error_nodes[-1] if error_nodes else "unknown_node"

    # Create a synthetic exception for recovery processing
    class WorkflowError(Exception):
        pass

    synthetic_error = WorkflowError(last_error_message)

    # Process the error through recovery manager
    recovery_result = await recovery_manager.handle_node_error(
        error=synthetic_error,
        node_name=last_error_node,
        state=state,
        original_function=lambda: None,  # Placeholder for synthetic error
    )

    # Clear handled errors from state
    recovery_result.update(
        {
            "error_messages": [],
            "error_nodes": [],
            "error_recovery_completed": True,
            "recovery_timestamp": time.time(),
        }
    )

    return recovery_result


__all__ = [
    "ErrorRecoveryManager",
    "get_error_recovery_manager",
    "error_recovery_node",
]
