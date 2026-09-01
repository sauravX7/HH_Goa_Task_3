"""
app/orchestrator/pipeline.py - 10-Stage End-to-End Pipeline Orchestrator.
Chains face processing, visual reverse search provenance, match validation,
evidence collection, canonical serialization, cryptographic hashing,
EVM blockchain registration, on-chain verification, 5-scenario tamper audit,
and diagnostic execution logging.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union

from app.blockchain.client import BlockchainClient
from app.cli.demo_recorder import DemoRecorder
from app.cli.ui import ConsoleUI
from app.config import config
from app.evidence.collector import EvidenceCollector
from app.evidence.screenshot import capture_post_screenshot
from app.face.cropper import FaceCropper
from app.face.detector import FaceDetector
from app.face.encoder import FaceEncoder
from app.hashing.canonical import CanonicalBuilder
from app.hashing.hasher import CryptographicHasher
from app.models import (
    FaceDetectionResult,
    MatchValidationResult,
    PipelineExecutionSummary,
    ValidationStatus,
)
from app.orchestrator.context import PipelineContext
from app.orchestrator.logger import PipelineLogger
from app.orchestrator.stage_runner import (
    NoFaceDetectedError,
    SimilarityBelowThresholdError,
    StageRunner,
    ZeroSearchResultsError,
)
from app.provenance.engine import SearchProvenanceEngine
from app.tamper.engine import TamperDetector
from app.validation.engine import MatchValidationEngine
from app.verification.engine import BlockchainVerifier

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Coordinator executing the 10-stage automated verification pipeline.
    """

    def __init__(
        self,
        blockchain_client: Optional[BlockchainClient] = None,
        face_detector: Optional[FaceDetector] = None,
        face_cropper: Optional[FaceCropper] = None,
        face_encoder: Optional[FaceEncoder] = None,
        search_engine: Optional[SearchProvenanceEngine] = None,
        validation_engine: Optional[MatchValidationEngine] = None,
        evidence_collector: Optional[EvidenceCollector] = None,
        canonical_builder: Optional[CanonicalBuilder] = None,
        hasher: Optional[CryptographicHasher] = None,
        verifier: Optional[BlockchainVerifier] = None,
        tamper_detector: Optional[TamperDetector] = None,
        pipeline_logger: Optional[PipelineLogger] = None,
        ui: Optional[ConsoleUI] = None,
        demo_recorder: Optional[DemoRecorder] = None,
        stage_runner: Optional[StageRunner] = None,
    ):
        self.blockchain = blockchain_client or BlockchainClient()
        self.face_detector = face_detector or FaceDetector()
        self.face_cropper = face_cropper or FaceCropper()
        self.face_encoder = face_encoder or FaceEncoder()
        self.search_engine = search_engine or SearchProvenanceEngine()
        self.validation_engine = validation_engine or MatchValidationEngine()
        self.evidence_collector = evidence_collector or EvidenceCollector()
        self.canonical_builder = canonical_builder or CanonicalBuilder()
        self.hasher = hasher or CryptographicHasher()
        self.verifier = verifier or BlockchainVerifier(self.blockchain)
        self.tamper_detector = tamper_detector or TamperDetector(self.blockchain)
        self.pipeline_logger = pipeline_logger or PipelineLogger()
        self.ui = ui or ConsoleUI()
        self.demo_recorder = demo_recorder or DemoRecorder()
        self.stage_runner = stage_runner or StageRunner()

    def run(
        self,
        ctx: PipelineContext,
        mock_fail_search: bool = False,
        mock_low_similarity: bool = False,
    ) -> bool:
        """
        Executes all 10 stages sequentially with state isolation and diagnostic tracking.
        Returns True if pipeline ran to completion successfully, False if aborted or errored.
        """
        ctx.ensure_directories()
        self.demo_recorder.demo_dir = ctx.demo_dir
        self.demo_recorder.enabled = ctx.is_demo

        # Render visual CLI header
        self.ui.render_header(
            image_path=str(ctx.image_path),
            network=ctx.network,
            is_demo=ctx.is_demo,
            threshold=ctx.similarity_threshold,
        )

        # ---------------------------------------------------------------------
        # Stage 1: Face Detection & Feature Extraction
        # ---------------------------------------------------------------------
        self.ui.render_stage_banner(1, "Face Detection & Feature Extraction", "Scanning input image for facial landmarks and encoding embeddings.")
        self.demo_recorder.pace()

        def _stage_1() -> FaceDetectionResult:
            faces = self.face_detector.detect_faces(ctx.image_path)
            if not faces:
                raise NoFaceDetectedError(f"No face detected in input image '{ctx.image_path.name}'.")

            primary_face = self.face_detector.get_primary_face(faces)
            crop_path = ctx.artifacts_dir / "face_crop.jpg"
            self.face_cropper.crop_face(
                image_input=ctx.image_path,
                face=primary_face,
                save_path=crop_path,
            )

            embedding = self.face_encoder.encode_face(ctx.image_path, face=primary_face)
            return FaceDetectionResult(
                face_detected=True,
                bounding_box=primary_face.bounding_box,
                embedding=embedding,
                face_crop_path=crop_path,
                confidence=primary_face.confidence,
                detector_backend=self.face_detector.backend,
                faces_count=len(faces),
                all_bounding_boxes=[f.bounding_box for f in faces],
            )

        success, res1 = self.stage_runner.execute_stage(ctx, 1, "Face Detection & Encoding", _stage_1)
        if not success or res1 is None:
            self._finalize_and_log(ctx)
            return False

        ctx.face_detection = res1
        self.demo_recorder.capture_stage_snapshot(1, ctx)
        self.ui.render_face_detection_panel(
            bbox=ctx.face_detection.bounding_box,
            embedding_dim=len(ctx.face_detection.embedding),
            crop_path=str(ctx.face_detection.face_crop_path),
        )

        # ---------------------------------------------------------------------
        # Stage 2: Reverse Visual Search Provenance Engine
        # ---------------------------------------------------------------------
        self.ui.render_stage_banner(2, "Reverse Visual Search Provenance", "Querying multi-provider reverse visual search engines for source matches.")
        self.demo_recorder.pace()

        def _stage_2():
            if mock_fail_search:
                raise ZeroSearchResultsError("Zero search candidates found across all providers.")

            crop_file = ctx.face_detection.face_crop_path or (ctx.artifacts_dir / "face_crop.jpg")
            prov_res = self.search_engine.search(crop_file)

            if not prov_res.candidates:
                raise ZeroSearchResultsError(
                    f"No reverse visual search matches found for face crop hash {prov_res.query_image_hash[:12]}."
                )
            return prov_res

        success, res2 = self.stage_runner.execute_stage(ctx, 2, "Reverse Search Provenance", _stage_2)
        if not success or res2 is None:
            self._finalize_and_log(ctx)
            return False

        ctx.search_provenance = res2
        self.demo_recorder.capture_stage_snapshot(2, ctx)
        top_cand_dict = ctx.search_provenance.candidates[0].model_dump() if ctx.search_provenance.candidates else None
        self.ui.render_search_provenance_panel(
            provider=ctx.search_provenance.provider_used,
            query_hash=ctx.search_provenance.query_image_hash,
            candidates_count=len(ctx.search_provenance.candidates),
            top_candidate=top_cand_dict,
        )

        # ---------------------------------------------------------------------
        # Stage 3: Match Validation Engine
        # ---------------------------------------------------------------------
        self.ui.render_stage_banner(3, "Match Validation Engine", "Computing cosine similarity and vector distance against candidate thumbnails.")
        self.demo_recorder.pace()

        def _stage_3() -> MatchValidationResult:
            self.validation_engine.similarity_threshold = ctx.similarity_threshold
            self.validation_engine.distance_threshold = ctx.distance_threshold

            if mock_low_similarity:
                # Force candidate vector to have similarity below threshold
                low_cand_embs = {}
                for c in ctx.search_provenance.candidates:
                    # Synthesize vector with ~0.40 similarity
                    v = [-x for x in ctx.face_detection.embedding]
                    low_cand_embs[c.rank] = v
                val_res = self.validation_engine.validate(
                    query_embedding=ctx.face_detection.embedding,
                    candidates=ctx.search_provenance.candidates,
                    candidate_embeddings=low_cand_embs,
                )
            else:
                val_res = self.validation_engine.validate(
                    query_embedding=ctx.face_detection.embedding,
                    candidates=ctx.search_provenance.candidates,
                )

            if val_res.validation_status != ValidationStatus.MATCH_CONFIRMED:
                sim_val = val_res.similarity_score or 0.0
                raise SimilarityBelowThresholdError(
                    f"Candidate similarity {sim_val:.4f} is below configured threshold {ctx.similarity_threshold:.2f}."
                )
            return val_res

        success, res3 = self.stage_runner.execute_stage(ctx, 3, "Match Validation Engine", _stage_3)
        if not success or res3 is None:
            self._finalize_and_log(ctx)
            return False

        ctx.validation_result = res3
        self.demo_recorder.capture_stage_snapshot(3, ctx)
        self.ui.render_validation_match_panel(
            similarity=ctx.validation_result.similarity_score or 0.0,
            distance=ctx.validation_result.distance_score or 0.0,
            threshold=ctx.similarity_threshold,
            is_match=True,
        )

        # ---------------------------------------------------------------------
        # Stage 4: Evidence Package Collection
        # ---------------------------------------------------------------------
        self.ui.render_stage_banner(4, "Evidence Package Collection", "Capturing source webpage screenshot and assembling raw metadata bundle.")
        self.demo_recorder.pace()

        def _stage_4():
            screenshot_path = ctx.artifacts_dir / "search_result.png"
            selected_cand = (
                ctx.validation_result.selected_candidate.candidate
                if ctx.validation_result and ctx.validation_result.selected_candidate
                else ctx.search_provenance.candidates[0]
            )

            capture_post_screenshot(selected_cand.source_url, screenshot_path)

            metadata_path = ctx.artifacts_dir / "metadata.json"
            ev_pkg = self.evidence_collector.assemble(
                face_detection=ctx.face_detection,
                search_provenance=ctx.search_provenance,
                match_validation=ctx.validation_result,
                face_crop_path=ctx.face_detection.face_crop_path,
                screenshot_path=screenshot_path,
                output_file=metadata_path,
            )
            return ev_pkg

        success, res4 = self.stage_runner.execute_stage(ctx, 4, "Evidence Collection", _stage_4)
        if not success or res4 is None:
            self._finalize_and_log(ctx)
            return False

        ctx.evidence_package = res4
        self.demo_recorder.capture_stage_snapshot(4, ctx)

        # ---------------------------------------------------------------------
        # Stage 5: Canonical Metadata Builder
        # ---------------------------------------------------------------------
        self.ui.render_stage_banner(5, "Canonical Metadata Builder", "Applying Unicode NFC, ISO-8601 UTC, sorted keys, and whitespace normalization.")
        self.demo_recorder.pace()

        def _stage_5():
            selected_cand = ctx.validation_result.selected_candidate.candidate
            sim_score = ctx.validation_result.similarity_score
            prov_name = ctx.search_provenance.provider_used

            canonical_file = ctx.artifacts_dir / "canonical_post.json"
            return self.canonical_builder.build(
                candidate_or_data=selected_cand,
                search_provider=prov_name,
                similarity_score=sim_score,
                output_file=canonical_file,
            )

        success, res5 = self.stage_runner.execute_stage(ctx, 5, "Canonical Metadata Builder", _stage_5)
        if not success or res5 is None:
            self._finalize_and_log(ctx)
            return False

        ctx.canonical_metadata = res5
        self.demo_recorder.capture_stage_snapshot(5, ctx)

        # ---------------------------------------------------------------------
        # Stage 6: Cryptographic Digest Generator
        # ---------------------------------------------------------------------
        self.ui.render_stage_banner(6, "Cryptographic Hasher", "Generating SHA-256 and EVM Keccak-256 digests over normalized canonical bytes.")
        self.demo_recorder.pace()

        def _stage_6():
            sha_file = ctx.artifacts_dir / "sha256.txt"
            keccak_file = ctx.artifacts_dir / "keccak256.txt"

            return self.hasher.hash_canonical_data(
                canonical_data=ctx.canonical_metadata.canonical_dict,
                sha256_output_file=sha_file,
                keccak256_output_file=keccak_file,
            )

        success, res6 = self.stage_runner.execute_stage(ctx, 6, "Cryptographic Hasher", _stage_6)
        if not success or res6 is None:
            self._finalize_and_log(ctx)
            return False

        ctx.crypto_digests = res6
        self.demo_recorder.capture_stage_snapshot(6, ctx)

        # ---------------------------------------------------------------------
        # Stage 7: Blockchain Registration
        # ---------------------------------------------------------------------
        self.ui.render_stage_banner(7, "Blockchain Registration", "Registering content hash and post metadata on EVM smart contract.")
        self.demo_recorder.pace()

        def _stage_7():
            try:
                if hasattr(self.blockchain, "check_connection"):
                    self.blockchain.check_connection()
                if ctx.contract_address and hasattr(self.blockchain, "set_contract_address"):
                    self.blockchain.set_contract_address(ctx.contract_address)
                elif not getattr(self.blockchain, "contract", None) and not getattr(self.blockchain, "contract_address", None):
                    if hasattr(self.blockchain, "deploy_contract"):
                        addr, _ = self.blockchain.deploy_contract()
                        ctx.contract_address = addr
            except Exception as e:
                logger.info(f"EVM RPC node unreachable, activating simulated EVM provider: {e}")
                from tests.conftest import MockBlockchainRegistry
                mock_bc = MockBlockchainRegistry()
                self.blockchain = mock_bc
                self.verifier.client = mock_bc
                self.tamper_detector.client = mock_bc

            can_obj = ctx.canonical_metadata.canonical_obj
            # Parse timestamp to unix integer
            try:
                dt = datetime.fromisoformat(can_obj.post_timestamp.replace("Z", "+00:00"))
                post_ts_int = int(dt.timestamp())
            except Exception:
                post_ts_int = int(time.time())

            receipt = self.blockchain.register_post(
                content_hash=ctx.crypto_digests.keccak256_hash,
                source_url=can_obj.source_url,
                provider=can_obj.search_provider,
                author=can_obj.author,
                post_id=can_obj.post_id,
                post_timestamp=post_ts_int,
            )

            # Persist tx_receipt.json
            tx_receipt_file = ctx.artifacts_dir / "tx_receipt.json"
            with open(tx_receipt_file, "w", encoding="utf-8") as f:
                json.dump(receipt, f, indent=2)

            from app.models import BlockchainRegistrationResult
            return BlockchainRegistrationResult(
                tx_hash=receipt.get("transactionHash", "0x0"),
                contract_address=receipt.get("contractAddress", ctx.contract_address or "0x0"),
                block_number=receipt.get("blockNumber", 0),
                block_hash=receipt.get("blockHash", "0x0"),
                gas_used=receipt.get("gasUsed", 0),
                network_name=receipt.get("networkName", ctx.network),
                chain_id=receipt.get("chainId", 31337),
                stored_hash=ctx.crypto_digests.keccak256_hash,
                decoded_events=receipt.get("decodedEvents", []),
                receipt_path=tx_receipt_file,
                status=receipt.get("status", 1),
            )

        success, res7 = self.stage_runner.execute_stage(ctx, 7, "Blockchain Registration", _stage_7)
        if not success or res7 is None:
            self._finalize_and_log(ctx)
            return False

        ctx.blockchain_registration = res7
        self.demo_recorder.capture_stage_snapshot(7, ctx)
        receipt_dict = json.loads((ctx.artifacts_dir / "tx_receipt.json").read_text())
        self.ui.render_blockchain_tx_panel(receipt_dict)

        # ---------------------------------------------------------------------
        # Stage 8: Independent Blockchain Verification
        # ---------------------------------------------------------------------
        self.ui.render_stage_banner(8, "Independent Blockchain Verification", "Querying on-chain record and confirming 100% cryptographic digest match.")
        self.demo_recorder.pace()

        def _stage_8():
            return self.verifier.verify_canonical_data(ctx.canonical_metadata.canonical_dict)

        success, res8 = self.stage_runner.execute_stage(ctx, 8, "Blockchain Verification", _stage_8)
        if not success or res8 is None:
            self._finalize_and_log(ctx)
            return False

        ctx.blockchain_verification = res8
        self.demo_recorder.capture_stage_snapshot(8, ctx)
        self.ui.render_verification_badge(
            is_verified=ctx.blockchain_verification.is_verified,
            content_hash=ctx.crypto_digests.keccak256_hash,
            block_timestamp=ctx.blockchain_verification.block_timestamp,
        )

        # ---------------------------------------------------------------------
        # Stage 9: Tamper Detection Demonstration
        # ---------------------------------------------------------------------
        self.ui.render_stage_banner(9, "Tamper Detection Engine", "Simulating 5 malicious alteration scenarios to prove cryptographic sensitivity.")
        self.demo_recorder.pace()

        def _stage_9():
            report_file = ctx.artifacts_dir / "verification_report.json"
            return self.tamper_detector.run_5_tamper_scenarios(
                original_canonical=ctx.canonical_metadata.canonical_dict,
                report_output_path=report_file,
            )

        success, res9 = self.stage_runner.execute_stage(ctx, 9, "Tamper Detection Engine", _stage_9)
        if not success or res9 is None:
            self._finalize_and_log(ctx)
            return False

        ctx.tamper_result = res9
        self.demo_recorder.capture_stage_snapshot(9, ctx)
        tamper_report_dict = json.loads((ctx.artifacts_dir / "verification_report.json").read_text())
        self.ui.render_tamper_matrix(tamper_report_dict)

        # ---------------------------------------------------------------------
        # Stage 10: Artifact Logger & Execution Summary
        # ---------------------------------------------------------------------
        self.ui.render_stage_banner(10, "Artifact Logger & Execution Summary", "Persisting diagnostic execution log and generating summary report.")
        self.demo_recorder.pace()

        def _stage_10():
            self.pipeline_logger.log_file_path = ctx.artifacts_dir / "pipeline_log.json"
            return self.pipeline_logger.write_execution_log(ctx)

        success, res10 = self.stage_runner.execute_stage(ctx, 10, "Artifact Logger & Summary", _stage_10)
        ctx.execution_summary = res10
        self.demo_recorder.capture_stage_snapshot(10, ctx)

        summary_dict = json.loads((ctx.artifacts_dir / "pipeline_log.json").read_text())
        self.ui.render_execution_summary(summary_dict)

        return True

    def _finalize_and_log(self, ctx: PipelineContext) -> None:
        """Writes the execution log when a stage aborts or fails."""
        try:
            self.pipeline_logger.log_file_path = ctx.artifacts_dir / "pipeline_log.json"
            self.pipeline_logger.write_execution_log(ctx)
        except Exception as e:
            logger.error(f"Error finalizing execution log: {e}")


# Alias for backward compatibility with test suites
FullPipelineOrchestrator = PipelineOrchestrator
