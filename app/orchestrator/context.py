"""
app/orchestrator/context.py - Strongly-typed state management for all 10 pipeline stages.
Maintains input parameters, stage outputs, execution timings, diagnostic logs, and artifact locations.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from app.config import config
from app.models import (
    BlockchainRegistrationResult,
    BlockchainVerificationResult,
    CanonicalMetadataResult,
    CryptographicDigestResult,
    DiagnosticLogEntry,
    EvidencePackage,
    FaceDetectionResult,
    MatchValidationResult,
    PipelineExecutionSummary,
    SearchProvenanceResult,
    StageExecutionLog,
    TamperDetectionResult,
)


class PipelineContext:
    """
    State container passed across all 10 execution stages.
    Maintains typed stage results, timestamps, logs, and paths.
    """

    def __init__(
        self,
        image_path: Path,
        artifacts_dir: Optional[Path] = None,
        is_demo: bool = False,
        similarity_threshold: Optional[float] = None,
        distance_threshold: Optional[float] = None,
        network: Optional[str] = None,
        contract_address: Optional[str] = None,
        pipeline_id: Optional[str] = None,
    ):
        self.pipeline_id = pipeline_id or f"pipe_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.image_path = Path(image_path).resolve()
        self.artifacts_dir = Path(artifacts_dir).resolve() if artifacts_dir else config.paths.artifacts_dir
        self.demo_dir = self.artifacts_dir / "demo"
        self.is_demo = is_demo
        self.similarity_threshold = (
            similarity_threshold if similarity_threshold is not None else config.effective_similarity_threshold
        )
        self.distance_threshold = (
            distance_threshold if distance_threshold is not None else config.effective_distance_threshold
        )
        self.network = network or config.effective_network
        self.contract_address = contract_address or config.effective_contract_address

        self.start_time: datetime = datetime.now(timezone.utc)
        self.end_time: Optional[datetime] = None
        self.is_aborted: bool = False
        self.abort_reason: Optional[str] = None

        # Stage outputs
        self.face_detection: Optional[FaceDetectionResult] = None
        self.search_provenance: Optional[SearchProvenanceResult] = None
        self.validation_result: Optional[MatchValidationResult] = None
        self.evidence_package: Optional[EvidencePackage] = None
        self.canonical_metadata: Optional[CanonicalMetadataResult] = None
        self.crypto_digests: Optional[CryptographicDigestResult] = None
        self.blockchain_registration: Optional[BlockchainRegistrationResult] = None
        self.blockchain_verification: Optional[BlockchainVerificationResult] = None
        self.tamper_result: Optional[TamperDetectionResult] = None
        self.execution_summary: Optional[PipelineExecutionSummary] = None

        # Logs
        self.stage_logs: List[StageExecutionLog] = []
        self.diagnostic_logs: List[DiagnosticLogEntry] = []

    def ensure_directories(self) -> None:
        """Create artifacts and demo output directories."""
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        if self.is_demo:
            self.demo_dir.mkdir(parents=True, exist_ok=True)

    def log_stage(
        self,
        stage_number: int,
        stage_name: str,
        start_time: datetime,
        end_time: datetime,
        status: str,
        details: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> StageExecutionLog:
        """Record a completed or aborted stage log entry."""
        duration = max(0.0, (end_time - start_time).total_seconds())
        entry = StageExecutionLog(
            stage_number=stage_number,
            stage_name=stage_name,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            status=status,
            details=details or {},
            error_message=error_message,
        )
        self.stage_logs.append(entry)
        return entry

    def add_diagnostic(
        self,
        message: str,
        level: str = "INFO",
        stage: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> DiagnosticLogEntry:
        """Record diagnostic log entry."""
        entry = DiagnosticLogEntry(
            timestamp=datetime.now(timezone.utc),
            level=level,
            stage=stage,
            message=message,
            details=details or {},
        )
        self.diagnostic_logs.append(entry)
        return entry

    def abort(self, reason: str, stage_number: int, stage_name: str) -> None:
        """Mark the pipeline execution as aborted due to an unrecoverable condition or check failure."""
        self.is_aborted = True
        self.abort_reason = reason
        self.add_diagnostic(
            message=f"Pipeline execution aborted at Stage {stage_number} ({stage_name}): {reason}",
            level="WARNING",
            stage=stage_name,
            details={"stage_number": stage_number, "reason": reason},
        )

    def get_artifact_map(self) -> Dict[str, str]:
        """Returns map of expected artifact file names to their local paths."""
        return {
            "face_crop": str(self.artifacts_dir / "face_crop.jpg"),
            "search_result": str(self.artifacts_dir / "search_result.png"),
            "metadata": str(self.artifacts_dir / "metadata.json"),
            "canonical_post": str(self.artifacts_dir / "canonical_post.json"),
            "sha256": str(self.artifacts_dir / "sha256.txt"),
            "keccak256": str(self.artifacts_dir / "keccak256.txt"),
            "tx_receipt": str(self.artifacts_dir / "tx_receipt.json"),
            "verification_report": str(self.artifacts_dir / "verification_report.json"),
            "pipeline_log": str(self.artifacts_dir / "pipeline_log.json"),
        }

    @property
    def total_duration_seconds(self) -> float:
        """Sum of all recorded stage durations."""
        return sum(s.duration_seconds for s in self.stage_logs)
