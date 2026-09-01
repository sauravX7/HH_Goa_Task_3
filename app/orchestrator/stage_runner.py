"""
app/orchestrator/stage_runner.py - Stage execution runner with exponential backoff and error recovery.
Provides structured timing, retries for transient network/RPC failures, and clean domain abort handling.
"""

from datetime import datetime, timezone
import logging
import time
from typing import Any, Callable, Dict, Optional, Tuple, Type

from app.orchestrator.context import PipelineContext

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Custom Pipeline Exceptions
# -------------------------------------------------------------------------

class PipelineStageError(Exception):
    """Base exception for pipeline execution errors."""
    pass


class NoFaceDetectedError(PipelineStageError):
    """Raised when no face is found in input image."""
    pass


class ZeroSearchResultsError(PipelineStageError):
    """Raised when reverse image search returns 0 candidates across all providers."""
    pass


class SimilarityBelowThresholdError(PipelineStageError):
    """Raised when the highest candidate similarity is below the required threshold."""
    pass


class BlockchainExecutionError(PipelineStageError):
    """Raised when on-chain transaction or query reverts or fails."""
    pass


class StageRunner:
    """
    Executes a single pipeline stage with timing, logging, and retry logic.
    """

    def __init__(
        self,
        max_retries: int = 2,
        initial_backoff_seconds: float = 0.5,
        backoff_multiplier: float = 2.0,
    ):
        self.max_retries = max_retries
        self.initial_backoff_seconds = initial_backoff_seconds
        self.backoff_multiplier = backoff_multiplier

    def execute_stage(
        self,
        ctx: PipelineContext,
        stage_number: int,
        stage_name: str,
        stage_func: Callable[[], Any],
        is_abort_on_error: bool = True,
        retryable_exceptions: Tuple[Type[Exception], ...] = (
            ConnectionError,
            TimeoutError,
            ConnectionRefusedError,
            OSError,
        ),
    ) -> Tuple[bool, Any]:
        """
        Executes stage_func within a timed, monitored context.
        Applies exponential backoff for transient failures.
        Returns (success_bool, stage_result).
        """
        start_time = datetime.now(timezone.utc)
        ctx.add_diagnostic(
            message=f"Starting Stage {stage_number}: {stage_name}",
            level="INFO",
            stage=stage_name,
        )

        attempts = 0
        backoff = self.initial_backoff_seconds
        last_exception: Optional[Exception] = None

        while attempts <= self.max_retries:
            attempts += 1
            try:
                result = stage_func()
                end_time = datetime.now(timezone.utc)
                ctx.log_stage(
                    stage_number=stage_number,
                    stage_name=stage_name,
                    start_time=start_time,
                    end_time=end_time,
                    status="SUCCESS",
                    details={"attempts": attempts},
                )
                ctx.add_diagnostic(
                    message=f"Completed Stage {stage_number}: {stage_name} successfully",
                    level="INFO",
                    stage=stage_name,
                    details={"attempts": attempts},
                )
                return True, result

            except (NoFaceDetectedError, ZeroSearchResultsError, SimilarityBelowThresholdError) as e:
                # Domain abort conditions — do not retry
                end_time = datetime.now(timezone.utc)
                error_msg = str(e)
                ctx.log_stage(
                    stage_number=stage_number,
                    stage_name=stage_name,
                    start_time=start_time,
                    end_time=end_time,
                    status="ABORTED",
                    details={"error": error_msg, "domain_abort": True},
                    error_message=error_msg,
                )
                ctx.abort(reason=error_msg, stage_number=stage_number, stage_name=stage_name)
                return False, None

            except retryable_exceptions as e:
                last_exception = e
                logger.warning(
                    f"Stage {stage_number} ({stage_name}) attempt {attempts} failed with {type(e).__name__}: {e}."
                )
                if attempts <= self.max_retries:
                    logger.info(f"Retrying Stage {stage_number} in {backoff:.2f}s...")
                    time.sleep(backoff)
                    backoff *= self.backoff_multiplier

            except Exception as e:
                last_exception = e
                logger.error(f"Stage {stage_number} ({stage_name}) failed with unexpected error: {e}", exc_info=True)
                break

        # If loop exited due to repeated failure
        end_time = datetime.now(timezone.utc)
        error_msg = str(last_exception) if last_exception else "Unknown failure"
        ctx.log_stage(
            stage_number=stage_number,
            stage_name=stage_name,
            start_time=start_time,
            end_time=end_time,
            status="ERROR",
            details={"error": error_msg, "attempts": attempts},
            error_message=error_msg,
        )
        if is_abort_on_error:
            ctx.abort(reason=error_msg, stage_number=stage_number, stage_name=stage_name)
        return False, None
