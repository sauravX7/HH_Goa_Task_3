"""
app/orchestrator/logger.py - Diagnostic and Execution Summary Logger.
Formats and persists artifacts/pipeline_log.json containing stage durations,
timings, status codes, error details, and artifact paths.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from app.config import config
from app.models import PipelineExecutionSummary
from app.orchestrator.context import PipelineContext

logger = logging.getLogger(__name__)


class PipelineLogger:
    """
    Structured execution logger for pipeline runs.
    Generates and persists artifacts/pipeline_log.json.
    """

    def __init__(self, log_file_path: Optional[Path] = None):
        self.log_file_path = log_file_path or config.paths.pipeline_log_file

    def write_execution_log(self, ctx: PipelineContext) -> PipelineExecutionSummary:
        """
        Builds the PipelineExecutionSummary from context and writes it to pipeline_log.json.
        """
        now_utc = datetime.now(timezone.utc)
        if ctx.end_time is None:
            ctx.end_time = now_utc

        overall_status = "ABORTED" if ctx.is_aborted else (
            "SUCCESS" if all(s.status == "SUCCESS" for s in ctx.stage_logs) else "FAILED"
        )

        artifact_map = ctx.get_artifact_map()
        # Filter to only existing artifacts
        existing_artifacts = {k: v for k, v in artifact_map.items() if Path(v).exists()}

        summary = PipelineExecutionSummary(
            pipeline_id=ctx.pipeline_id,
            start_time=ctx.start_time,
            end_time=ctx.end_time,
            total_duration_seconds=round(ctx.total_duration_seconds, 4),
            status=overall_status,
            stages=ctx.stage_logs,
            artifact_paths=existing_artifacts,
            pipeline_log_path=self.log_file_path,
            diagnostic_logs=ctx.diagnostic_logs,
        )

        ctx.execution_summary = summary

        # Ensure directory exists and write JSON
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)

        summary_dict = summary.model_dump(mode="json")
        with open(self.log_file_path, "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, indent=2)

        logger.info(
            f"Saved pipeline diagnostic execution log to {self.log_file_path} "
            f"(Status: {overall_status}, Total Duration: {summary.total_duration_seconds:.3f}s)"
        )
        return summary
