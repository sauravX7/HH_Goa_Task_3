"""
app/orchestrator - Pipeline orchestration, state management, and diagnostic logging.
"""

from app.orchestrator.context import PipelineContext
from app.orchestrator.logger import PipelineLogger
from app.orchestrator.pipeline import FullPipelineOrchestrator, PipelineOrchestrator
from app.orchestrator.stage_runner import (
    BlockchainExecutionError,
    NoFaceDetectedError,
    PipelineStageError,
    SimilarityBelowThresholdError,
    StageRunner,
    ZeroSearchResultsError,
)

__all__ = [
    "PipelineContext",
    "PipelineLogger",
    "PipelineOrchestrator",
    "FullPipelineOrchestrator",
    "StageRunner",
    "PipelineStageError",
    "NoFaceDetectedError",
    "ZeroSearchResultsError",
    "SimilarityBelowThresholdError",
    "BlockchainExecutionError",
]
