"""Pydantic v2 Data Models and Type Contracts for All 10 Pipeline Stages."""

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, ConfigDict, Field


class ValidationStatus(str, Enum):
    MATCH_CONFIRMED = "MATCH_CONFIRMED"
    SIMILARITY_BELOW_THRESHOLD = "SIMILARITY_BELOW_THRESHOLD"
    NO_FACE_IN_CANDIDATE = "NO_FACE_IN_CANDIDATE"
    REJECTED = "REJECTED"


class VerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    TAMPER_DETECTED = "TAMPER_DETECTED"
    NOT_FOUND_ON_CHAIN = "NOT_FOUND_ON_CHAIN"


# -------------------------------------------------------------------------
# Stage 1: Face Identification & Feature Processing Models
# -------------------------------------------------------------------------

class BoundingBox(BaseModel):
    model_config = ConfigDict(frozen=True)
    top: int
    right: int
    bottom: int
    left: int

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.top, self.right, self.bottom, self.left)


class FaceDetectionResult(BaseModel):
    face_detected: bool
    bounding_box: Optional[Tuple[int, int, int, int]] = None  # (top, right, bottom, left)
    embedding: List[float] = Field(default_factory=list)      # 128-d or 512-d normalized vector
    face_crop_path: Optional[Path] = None
    confidence: float = 0.0
    detector_backend: str = "face_recognition"
    faces_count: int = 0
    all_bounding_boxes: List[Tuple[int, int, int, int]] = Field(default_factory=list)
    image_shape: Optional[Tuple[int, int, int]] = None       # (height, width, channels)
    processing_time_ms: float = 0.0


# -------------------------------------------------------------------------
# Stage 2: Genuine Search Provenance Engine Models
# -------------------------------------------------------------------------

class SearchCandidate(BaseModel):
    rank: int
    title: str
    source_url: str
    thumbnail_url: Optional[str] = None
    image_url: Optional[str] = None
    snippet: Optional[str] = None
    author: Optional[str] = None
    post_date: Optional[str] = None
    provider_confidence: Optional[float] = None
    raw_payload: Dict[str, Any] = Field(default_factory=dict)


class ProviderAttemptLog(BaseModel):
    provider_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    success: bool
    latency_seconds: float
    error_message: Optional[str] = None
    candidates_found: int = 0


class SearchProvenanceResult(BaseModel):
    provider_used: str
    query_image_hash: str
    query_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    query_id: str
    candidates: List[SearchCandidate] = Field(default_factory=list)
    fallback_history: List[ProviderAttemptLog] = Field(default_factory=list)
    total_results_found: int = 0


# -------------------------------------------------------------------------
# Stage 3: Match Validation Engine Models
# -------------------------------------------------------------------------

class RejectedCandidateLog(BaseModel):
    rank: int
    source_url: str
    thumbnail_url: Optional[str] = None
    similarity_score: Optional[float] = None
    distance_score: Optional[float] = None
    rejection_reason: str


class ValidatedCandidate(BaseModel):
    selected_rank: int
    candidate: SearchCandidate
    similarity_score: float
    distance_score: float
    candidate_embedding: List[float] = Field(default_factory=list)


class MatchValidationResult(BaseModel):
    validation_status: ValidationStatus
    selected_candidate: Optional[ValidatedCandidate] = None
    similarity_score: Optional[float] = None
    distance_score: Optional[float] = None
    provider_confidence: Optional[float] = None
    selected_rank: Optional[int] = None
    threshold_used: float = 0.60
    rejected_candidates: List[RejectedCandidateLog] = Field(default_factory=list)


# -------------------------------------------------------------------------
# Stage 4: Evidence Package & Raw Metadata
# -------------------------------------------------------------------------

class EvidencePackage(BaseModel):
    face_crop_path: Path
    screenshot_path: Optional[Path] = None
    raw_metadata_path: Path
    evidence_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    raw_metadata: Dict[str, Any] = Field(default_factory=dict)


# -------------------------------------------------------------------------
# Stage 5: Canonical Metadata Models
# -------------------------------------------------------------------------

class CanonicalMetadata(BaseModel):
    author: str
    caption: str
    media_sha256: str
    post_id: str
    post_timestamp: str           # ISO-8601 UTC
    schema_version: str = "1.0.0"
    search_provider: str
    similarity_score: Union[str, float]
    source_url: str
    validated_at: Optional[str] = None  # ISO-8601 UTC


class CanonicalMetadataResult(BaseModel):
    canonical_obj: CanonicalMetadata
    canonical_dict: Dict[str, Any]
    canonical_json_bytes: bytes
    canonical_json_str: str
    canonical_file_path: Path


# -------------------------------------------------------------------------
# Stage 6: Cryptographic Hashes
# -------------------------------------------------------------------------

class CryptographicDigestResult(BaseModel):
    sha256_hash: str              # 64 char hex
    keccak256_hash: str          # 66 char hex with 0x prefix (bytes32 format)
    sha256_path: Path
    keccak256_path: Path


# -------------------------------------------------------------------------
# Stage 7: Blockchain Registration Models
# -------------------------------------------------------------------------

class DecodedEvent(BaseModel):
    event_name: str
    contract_address: str
    block_number: int
    transaction_hash: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class BlockchainRegistrationResult(BaseModel):
    tx_hash: str
    contract_address: str
    block_number: int
    block_hash: str
    gas_used: int
    network_name: str
    chain_id: int
    stored_hash: str
    decoded_events: List[Dict[str, Any]] = Field(default_factory=list)
    receipt_path: Path
    status: int = 1
    from_address: Optional[str] = None


# -------------------------------------------------------------------------
# Stage 8: Blockchain Verification Models
# -------------------------------------------------------------------------

class BlockchainVerificationResult(BaseModel):
    verification_status: VerificationStatus
    is_verified: bool
    on_chain_exists: bool
    on_chain_content_hash: str
    computed_content_hash: str
    hashes_match: bool
    on_chain_metadata: Dict[str, Any] = Field(default_factory=dict)
    event_verification: Dict[str, Any] = Field(default_factory=dict)
    block_number: int = 0
    block_timestamp: int = 0
    verification_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    rationale: str = ""


# -------------------------------------------------------------------------
# Stage 9: Tamper Detection Models
# -------------------------------------------------------------------------

class FieldDiff(BaseModel):
    field_name: str
    original_value: Any
    tampered_value: Any
    impact_description: str


class TamperScenarioOutcome(BaseModel):
    scenario_id: str              # e.g., "SCENARIO_1_MODIFIED_CAPTION"
    scenario_name: str
    description: str
    status: str                   # "TAMPER_DETECTED" or "VERIFIED"
    original_hash: str
    tampered_hash: str
    hashes_differ: bool
    on_chain_query_result: str    # "MISMATCH" or "NOT_FOUND"
    diffs: List[FieldDiff] = Field(default_factory=list)
    detected: bool = True


class TamperDetectionResult(BaseModel):
    baseline_status: VerificationStatus
    total_scenarios: int = 5
    detected_tamper_count: int = 5
    all_tampered_detected: bool = True
    scenarios: List[TamperScenarioOutcome] = Field(default_factory=list)
    report_path: Path


# -------------------------------------------------------------------------
# Stage 10: Orchestration & Diagnostic Logging Models
# -------------------------------------------------------------------------

class StageExecutionLog(BaseModel):
    stage_number: int
    stage_name: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    status: str                   # "SUCCESS", "FAILED", "SKIPPED"
    details: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None


class DiagnosticLogEntry(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    level: str = "INFO"
    stage: Optional[str] = None
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)


class PipelineExecutionSummary(BaseModel):
    pipeline_id: str
    start_time: datetime
    end_time: datetime
    total_duration_seconds: float
    status: str                   # "SUCCESS" or "FAILED"
    stages: List[StageExecutionLog] = Field(default_factory=list)
    artifact_paths: Dict[str, str] = Field(default_factory=dict)
    pipeline_log_path: Path
    diagnostic_logs: List[DiagnosticLogEntry] = Field(default_factory=list)
