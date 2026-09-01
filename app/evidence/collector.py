"""
app/evidence/collector.py - Evidence bundle assembler compiling artifacts/metadata.json.

Implements Requirement R4:
- Bundles face detection query parameters, search provenance, candidate rankings, match validation metrics, and raw metadata.
- Persists structured metadata to artifacts/metadata.json.
- Returns validated EvidencePackage instance.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from app.config import config
from app.models import (
    EvidencePackage,
    FaceDetectionResult,
    MatchValidationResult,
    SearchProvenanceResult,
)


class EvidenceCollector:
    """
    Assembles evidence bundle and outputs artifacts/metadata.json.
    """

    def __init__(
        self,
        metadata_file: Optional[Path] = None,
        face_crop_file: Optional[Path] = None,
        screenshot_file: Optional[Path] = None,
    ):
        self.metadata_file = metadata_file or config.paths.metadata_file
        self.face_crop_file = face_crop_file or config.paths.face_crop_file
        self.screenshot_file = screenshot_file or config.paths.screenshot_file

    def assemble(
        self,
        face_detection: Union[FaceDetectionResult, Dict[str, Any]],
        search_provenance: Union[SearchProvenanceResult, Dict[str, Any]],
        match_validation: Union[MatchValidationResult, Dict[str, Any]],
        face_crop_path: Optional[Path] = None,
        screenshot_path: Optional[Path] = None,
        output_file: Optional[Path] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> EvidencePackage:
        """
        Compiles all stage results into structured metadata dictionary and writes to metadata.json.
        """
        target_metadata_file = output_file or self.metadata_file
        target_face_crop = face_crop_path or self.face_crop_file
        target_screenshot = screenshot_path or self.screenshot_file

        # Serialize input models if passed as Pydantic models
        if isinstance(face_detection, FaceDetectionResult):
            query_dict = {
                "face_detected": face_detection.face_detected,
                "confidence": face_detection.confidence,
                "detector_backend": face_detection.detector_backend,
                "bounding_box": face_detection.bounding_box,
                "faces_count": face_detection.faces_count,
                "face_crop_path": str(face_detection.face_crop_path) if face_detection.face_crop_path else None,
                "processing_time_ms": face_detection.processing_time_ms,
            }
        else:
            query_dict = dict(face_detection)

        if isinstance(search_provenance, SearchProvenanceResult):
            prov_dict = {
                "provider_used": search_provenance.provider_used,
                "query_image_hash": search_provenance.query_image_hash,
                "query_timestamp": search_provenance.query_timestamp.isoformat() if isinstance(search_provenance.query_timestamp, datetime) else str(search_provenance.query_timestamp),
                "query_id": search_provenance.query_id,
                "total_results_found": search_provenance.total_results_found,
                "candidates": [c.model_dump(mode="json") if hasattr(c, "model_dump") else dict(c) for c in search_provenance.candidates],
                "fallback_history": [f.model_dump(mode="json") if hasattr(f, "model_dump") else dict(f) for f in search_provenance.fallback_history],
            }
        else:
            prov_dict = dict(search_provenance)

        if isinstance(match_validation, MatchValidationResult):
            val_dict = {
                "validation_status": match_validation.validation_status.value if hasattr(match_validation.validation_status, "value") else str(match_validation.validation_status),
                "similarity_score": match_validation.similarity_score,
                "distance_score": match_validation.distance_score,
                "provider_confidence": match_validation.provider_confidence,
                "selected_rank": match_validation.selected_rank,
                "threshold_used": match_validation.threshold_used,
                "selected_candidate": match_validation.selected_candidate.model_dump(mode="json") if match_validation.selected_candidate and hasattr(match_validation.selected_candidate, "model_dump") else match_validation.selected_candidate,
                "rejected_candidates": [r.model_dump(mode="json") if hasattr(r, "model_dump") else dict(r) for r in match_validation.rejected_candidates],
            }
        else:
            val_dict = dict(match_validation)

        now_utc = datetime.now(timezone.utc)

        raw_metadata_payload: Dict[str, Any] = {
            "version": "1.0.0",
            "evidence_collected_at": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "query": query_dict,
            "search_provenance": prov_dict,
            "match_validation": val_dict,
            "artifacts": {
                "face_crop": str(target_face_crop),
                "screenshot": str(target_screenshot) if target_screenshot else None,
                "metadata": str(target_metadata_file),
            },
        }

        if extra_metadata:
            raw_metadata_payload["extra"] = extra_metadata

        # Save metadata.json
        target_metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with open(target_metadata_file, "w", encoding="utf-8") as f:
            json.dump(raw_metadata_payload, f, indent=2, ensure_ascii=False, default=str)

        return EvidencePackage(
            face_crop_path=target_face_crop,
            screenshot_path=target_screenshot if target_screenshot and target_screenshot.exists() else None,
            raw_metadata_path=target_metadata_file,
            evidence_timestamp=now_utc,
            raw_metadata=raw_metadata_payload,
        )


def assemble_evidence_package(
    face_detection: Union[FaceDetectionResult, Dict[str, Any]],
    search_provenance: Union[SearchProvenanceResult, Dict[str, Any]],
    match_validation: Union[MatchValidationResult, Dict[str, Any]],
    face_crop_path: Optional[Path] = None,
    screenshot_path: Optional[Path] = None,
    output_file: Optional[Path] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> EvidencePackage:
    """
    Convenience function to assemble evidence package and save metadata.json.
    """
    collector = EvidenceCollector(
        metadata_file=output_file,
        face_crop_file=face_crop_path,
        screenshot_file=screenshot_path,
    )
    return collector.assemble(
        face_detection=face_detection,
        search_provenance=search_provenance,
        match_validation=match_validation,
        face_crop_path=face_crop_path,
        screenshot_path=screenshot_path,
        output_file=output_file,
        extra_metadata=extra_metadata,
    )
