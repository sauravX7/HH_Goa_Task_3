"""
app/cli/demo_recorder.py - Screen recording mode coordinator and demo snapshot generator.
Handles paced stage transitions and captures visual summary cards into artifacts/demo/.
"""

import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from app.config import config
from app.orchestrator.context import PipelineContext

logger = logging.getLogger(__name__)


class DemoRecorder:
    """
    Coordinates demo presentation pacing and automatic visual artifact generation.
    """

    def __init__(
        self,
        demo_dir: Optional[Path] = None,
        step_delay_seconds: float = 1.0,
        enabled: bool = True,
    ):
        self.demo_dir = Path(demo_dir) if demo_dir else config.paths.demo_dir
        self.step_delay = step_delay_seconds
        self.enabled = enabled
        if self.enabled:
            self.demo_dir.mkdir(parents=True, exist_ok=True)

    def pace(self, delay: Optional[float] = None) -> None:
        """Pauses execution briefly to allow human observation during demo recording."""
        if not self.enabled:
            return
        pause_duration = delay if delay is not None else self.step_delay
        if pause_duration > 0:
            time.sleep(pause_duration)

    def _create_snapshot_card(
        self,
        title: str,
        subtitle: str,
        stage_num: int,
        key_values: Dict[str, str],
        status: str = "SUCCESS",
        badge_color: Tuple[int, int, int] = (16, 185, 129), # Emerald green
    ) -> Image.Image:
        """Generates a modern, high-contrast dark UI card as a snapshot image."""
        width, height = 1000, 600
        bg_color = (15, 23, 42) # Slate-900
        card_bg = (30, 41, 59) # Slate-800
        border_color = (51, 65, 85) # Slate-700
        text_primary = (248, 250, 252)
        text_secondary = (148, 163, 184)
        accent_cyan = (56, 189, 248)

        img = Image.new("RGB", (width, height), bg_color)
        draw = ImageDraw.Draw(img)

        # Draw card container
        draw.rectangle([(40, 40), (width - 40, height - 40)], fill=card_bg, outline=border_color, width=2)

        # Top banner bar
        draw.rectangle([(40, 40), (width - 40, 110)], fill=(15, 23, 42))
        draw.line([(40, 110), (width - 40, 110)], fill=border_color, width=2)

        # Stage number pill
        stage_label = f"STAGE {stage_num:02d} / 10"
        draw.rectangle([(60, 60), (220, 92)], fill=(3, 105, 161), outline=accent_cyan, width=1)
        draw.text((75, 68), stage_label, fill=(255, 255, 255))

        # Title & Subtitle
        draw.text((240, 65), title, fill=text_primary)
        draw.text((60, 125), subtitle, fill=text_secondary)

        # Status badge
        badge_label = f"● {status}"
        draw.rectangle([(width - 200, 60), (width - 60, 92)], fill=badge_color)
        draw.text((width - 180, 68), badge_label, fill=(255, 255, 255))

        # Key-value rows
        y = 175
        for k, v in key_values.items():
            # Field label
            draw.text((65, y), str(k).upper(), fill=accent_cyan)
            # Field value (truncate if too long)
            val_str = str(v)
            if len(val_str) > 85:
                val_str = val_str[:82] + "..."
            draw.text((65, y + 22), val_str, fill=text_primary)
            draw.line([(60, y + 55), (width - 60, y + 55)], fill=(45, 55, 72), width=1)
            y += 68
            if y > height - 80:
                break

        # Footer timestamp & brand
        draw.text((60, height - 70), "🛡️ Verified Face Provenance & Blockchain Integrity Pipeline", fill=(100, 116, 139))
        return img

    def save_snapshot(self, filename: str, img: Image.Image, alt_filenames: Optional[list] = None) -> Path:
        """Saves generated image to demo directory and creates any aliases."""
        target_path = self.demo_dir / filename
        img.save(target_path, format="PNG")
        if alt_filenames:
            for alt in alt_filenames:
                alt_path = self.demo_dir / alt
                img.save(alt_path, format="PNG")
        return target_path

    def capture_stage_snapshot(self, stage_num: int, ctx: PipelineContext) -> Optional[Path]:
        """Captures specific milestone snapshots depending on the completed stage."""
        if not self.enabled:
            return None

        self.demo_dir.mkdir(parents=True, exist_ok=True)

        if stage_num == 1 and ctx.face_detection:
            bbox_str = str(ctx.face_detection.bounding_box) if ctx.face_detection.bounding_box else "N/A"
            card = self._create_snapshot_card(
                title="Face Identification & Feature Extraction",
                subtitle="High-dimensional facial feature extraction & normalized face crop isolation.",
                stage_num=1,
                key_values={
                    "Detection Status": "Face Detected & Bounding Box Isolated",
                    "Bounding Box": bbox_str,
                    "Embedding Dimension": f"{len(ctx.face_detection.embedding)}-dimensional unit vector",
                    "Face Crop File": "artifacts/face_crop.jpg",
                },
            )
            self.save_snapshot("01_face_detection.png", card, ["demo_01_face_crop.png", "demo_01_face_detection.png"])

        elif stage_num == 2 and ctx.search_provenance:
            top_cand = ctx.search_provenance.candidates[0] if ctx.search_provenance.candidates else None
            card = self._create_snapshot_card(
                title="Reverse Visual Search Provenance",
                subtitle="Live multi-engine visual search candidate discovery with audit logging.",
                stage_num=2,
                key_values={
                    "Active Provider": ctx.search_provenance.provider_used,
                    "Query Face Hash": ctx.search_provenance.query_image_hash,
                    "Candidates Discovered": f"{len(ctx.search_provenance.candidates)} candidates",
                    "Top Source Post URL": top_cand.source_url if top_cand else "N/A",
                },
            )
            self.save_snapshot("02_reverse_search.png", card, ["demo_02_search_results.png", "demo_02_reverse_search.png"])

        elif stage_num == 3 and ctx.validation_result:
            is_match = (ctx.validation_result.validation_status.value == "MATCH_CONFIRMED")
            card = self._create_snapshot_card(
                title="Match Validation Engine",
                subtitle="Cosine similarity ranking & threshold comparison against candidate vectors.",
                stage_num=3,
                key_values={
                    "Validation Status": str(ctx.validation_result.validation_status.value),
                    "Cosine Similarity": f"{ctx.validation_result.similarity_score:.4f}" if ctx.validation_result.similarity_score else "N/A",
                    "Similarity Threshold": f"{ctx.validation_result.threshold_used:.2f}",
                    "Rank Selected": str(ctx.validation_result.selected_rank or 1),
                },
                status="MATCH_CONFIRMED" if is_match else "REJECTED",
                badge_color=(16, 185, 129) if is_match else (239, 68, 68),
            )
            self.save_snapshot("03_match_validation.png", card, ["demo_03_validation_match.png", "demo_03_match_validation.png"])

        elif stage_num == 4:
            card = self._create_snapshot_card(
                title="Evidence Package Collection",
                subtitle="Cryptographic screenshot capture & raw provenance metadata assembly.",
                stage_num=4,
                key_values={
                    "Screenshot Captured": "artifacts/search_result.png",
                    "Raw Metadata Saved": "artifacts/metadata.json",
                    "Normalized Face Crop": "artifacts/face_crop.jpg",
                },
            )
            self.save_snapshot("04_evidence_captured.png", card, ["demo_04_evidence_collection.png"])

        elif stage_num == 5 and ctx.canonical_metadata:
            card = self._create_snapshot_card(
                title="Canonical Metadata Builder",
                subtitle="Deterministic RFC-8785 JSON formatting with Unicode NFC and ISO-8601 UTC.",
                stage_num=5,
                key_values={
                    "Canonical Output File": "artifacts/canonical_post.json",
                    "Author": ctx.canonical_metadata.canonical_obj.author,
                    "Post Timestamp": ctx.canonical_metadata.canonical_obj.post_timestamp,
                    "Canonical Byte Size": f"{len(ctx.canonical_metadata.canonical_json_bytes)} bytes",
                },
            )
            self.save_snapshot("05_canonical_metadata.png", card, ["demo_05_canonical_metadata.png"])

        elif stage_num == 6 and ctx.crypto_digests:
            card = self._create_snapshot_card(
                title="Cryptographic Hasher",
                subtitle="FIPS 180-4 SHA-256 and Ethereum Keccak-256 digests over canonical bytes.",
                stage_num=6,
                key_values={
                    "SHA-256 Digest": ctx.crypto_digests.sha256_hash,
                    "Keccak-256 (bytes32)": ctx.crypto_digests.keccak256_hash,
                    "Persisted Hashes": "artifacts/sha256.txt, artifacts/keccak256.txt",
                },
            )
            self.save_snapshot("06_crypto_digests.png", card, ["demo_06_crypto_digests.png"])

        elif stage_num == 7 and ctx.blockchain_registration:
            card = self._create_snapshot_card(
                title="Blockchain Registration",
                subtitle="EVM smart contract registration with Solidity event emission.",
                stage_num=7,
                key_values={
                    "Network Name": ctx.blockchain_registration.network_name,
                    "Contract Address": ctx.blockchain_registration.contract_address,
                    "Transaction Hash": ctx.blockchain_registration.tx_hash,
                    "Block Number": str(ctx.blockchain_registration.block_number),
                    "Gas Used": f"{ctx.blockchain_registration.gas_used:,} units",
                },
            )
            self.save_snapshot("07_blockchain_tx.png", card, ["demo_04_blockchain_tx.png", "demo_07_blockchain_tx.png"])

        elif stage_num == 8 and ctx.blockchain_verification:
            is_v = ctx.blockchain_verification.is_verified
            card = self._create_snapshot_card(
                title="Independent Blockchain Verification",
                subtitle="On-chain state retrieval, hash recomputation & 100% cryptographic equality check.",
                stage_num=8,
                key_values={
                    "Verification Status": str(ctx.blockchain_verification.verification_status.value),
                    "Computed Hash": ctx.blockchain_verification.computed_content_hash,
                    "On-Chain Hash": ctx.blockchain_verification.on_chain_content_hash,
                    "Hashes Match": "TRUE (100% Cryptographic Equality)" if is_v else "FALSE",
                },
                status="VERIFIED" if is_v else "FAILED",
                badge_color=(16, 185, 129) if is_v else (239, 68, 68),
            )
            self.save_snapshot("08_onchain_verification.png", card, ["demo_05_verification_badge.png", "demo_08_onchain_verification.png"])

        elif stage_num == 9 and ctx.tamper_result:
            all_det = ctx.tamper_result.all_tampered_detected
            card = self._create_snapshot_card(
                title="Tamper Detection Engine",
                subtitle="Automated 5-scenario tamper simulation and field-level diff reporting.",
                stage_num=9,
                key_values={
                    "Total Attack Scenarios": str(ctx.tamper_result.total_scenarios),
                    "Detected Tamper Count": f"{ctx.tamper_result.detected_tamper_count} / {ctx.tamper_result.total_scenarios}",
                    "Detection Matrix Status": "100% TAMPER DETECTED" if all_det else "INCOMPLETE",
                    "Verification Report": "artifacts/verification_report.json",
                },
                status="100% DETECTED" if all_det else "FAILED",
                badge_color=(16, 185, 129) if all_det else (239, 68, 68),
            )
            self.save_snapshot("09_tamper_matrix.png", card, ["demo_06_tamper_diff.png", "demo_09_tamper_matrix.png"])

        elif stage_num == 10 and ctx.execution_summary:
            card = self._create_snapshot_card(
                title="Pipeline Execution Summary",
                subtitle="10-stage execution duration breakdown and diagnostic logging.",
                stage_num=10,
                key_values={
                    "Pipeline ID": ctx.execution_summary.pipeline_id,
                    "Total Duration": f"{ctx.execution_summary.total_duration_seconds:.3f} seconds",
                    "Overall Status": ctx.execution_summary.status,
                    "Diagnostic Log": "artifacts/pipeline_log.json",
                },
            )
            self.save_snapshot("10_final_summary.png", card, ["demo_10_final_summary.png"])

        return None
