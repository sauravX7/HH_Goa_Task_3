"""
app/verification/engine.py - Independent Blockchain Record Verification Engine.

Implements Requirement R6:
- Loads artifacts/canonical_post.json or passed post path/dict.
- Recomputes SHA-256 and Keccak-256 digests.
- Queries smart contract (isRegistered / getPost / verifyPost).
- Verifies event logs (PostRegistered event match).
- Determines VerificationStatus (VERIFIED vs TAMPER_DETECTED vs NOT_FOUND_ON_CHAIN).
- Compiles structured BlockchainVerificationResult and verification report.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from app.blockchain.client import BlockchainClient
from app.blockchain.events import to_json_serializable
from app.config import config
from app.hashing.canonical import serialize_canonical_json
from app.hashing.hasher import compute_keccak256_digest, compute_sha256_digest
from app.models import BlockchainVerificationResult, CanonicalMetadata, VerificationStatus
from app.verification.comparator import VerificationComparator, compare_canonical_vs_onchain

logger = logging.getLogger(__name__)


class VerificationEngine:
    """
    Core verification engine that performs independent cryptographic and on-chain verification.
    """

    def __init__(
        self,
        client: Optional[Any] = None,
        network: Optional[str] = None,
        contract_address: Optional[str] = None,
    ):
        self.network = network or config.effective_network
        self.contract_address = contract_address or config.effective_contract_address
        self.client = client

    def _get_or_create_client(self) -> Any:
        """Retrieves existing blockchain client or creates one with active configuration."""
        if self.client is not None:
            return self.client
        try:
            self.client = BlockchainClient(
                network=self.network,
                contract_address=self.contract_address,
            )
            return self.client
        except Exception as e:
            logger.warning(f"Could not initialize default BlockchainClient: {e}")
            return None

    def load_canonical_data(
        self,
        post_data_or_path: Optional[Union[Path, str, Dict[str, Any], CanonicalMetadata]] = None,
    ) -> Dict[str, Any]:
        """Loads and normalizes canonical dictionary from path, string, model, or default file."""
        if post_data_or_path is None:
            default_path = config.paths.canonical_post_file
            if not default_path.exists():
                raise FileNotFoundError(f"Canonical post file not found at default path: {default_path}")
            return json.loads(default_path.read_text(encoding="utf-8"))

        if isinstance(post_data_or_path, CanonicalMetadata):
            return post_data_or_path.model_dump()

        if isinstance(post_data_or_path, dict):
            return dict(post_data_or_path)

        path_obj = Path(post_data_or_path)
        if not path_obj.exists() or not path_obj.is_file():
            raise FileNotFoundError(f"Canonical post file not found: {post_data_or_path}")

        return json.loads(path_obj.read_text(encoding="utf-8"))

    def verify_post(
        self,
        post_data_or_path: Optional[Union[Path, str, Dict[str, Any], CanonicalMetadata]] = None,
        receipt_path: Optional[Union[Path, str]] = None,
        check_events: bool = True,
    ) -> BlockchainVerificationResult:
        """
        Executes full on-chain verification workflow against the smart contract.
        """
        canonical_dict = self.load_canonical_data(post_data_or_path)

        # 1. Compute digests
        _, canonical_bytes = serialize_canonical_json(canonical_dict)
        computed_keccak = compute_keccak256_digest(canonical_bytes).lower()
        computed_sha256 = compute_sha256_digest(canonical_bytes).lower()

        client = self._get_or_create_client()

        # Handle case where no client is available
        if client is None:
            return BlockchainVerificationResult(
                verification_status=VerificationStatus.NOT_FOUND_ON_CHAIN,
                is_verified=False,
                on_chain_exists=False,
                on_chain_content_hash="",
                computed_content_hash=computed_keccak,
                hashes_match=False,
                on_chain_metadata={},
                event_verification={"verified": False, "reason": "No blockchain client available"},
                block_number=0,
                block_timestamp=0,
                verification_timestamp=datetime.now(timezone.utc),
                rationale="Blockchain client connection is unavailable.",
            )

        # 2. Check if registered on-chain
        try:
            is_registered = client.is_registered(computed_keccak)
        except Exception as e:
            logger.warning(f"Error querying is_registered: {e}")
            is_registered = False

        if not is_registered:
            return BlockchainVerificationResult(
                verification_status=VerificationStatus.NOT_FOUND_ON_CHAIN,
                is_verified=False,
                on_chain_exists=False,
                on_chain_content_hash="",
                computed_content_hash=computed_keccak,
                hashes_match=False,
                on_chain_metadata={},
                event_verification={"verified": False, "reason": "Record not registered on chain"},
                block_number=0,
                block_timestamp=0,
                verification_timestamp=datetime.now(timezone.utc),
                rationale=f"Content hash {computed_keccak} is not registered in on-chain smart contract.",
            )

        # 3. Retrieve on-chain record
        on_chain_record: Dict[str, Any] = {}
        try:
            if hasattr(client, "get_post"):
                on_chain_record = client.get_post(computed_keccak)
            elif hasattr(client, "verify_post"):
                exists, reg_ts, src_url = client.verify_post(computed_keccak)
                on_chain_record = {
                    "contentHash": computed_keccak,
                    "sourceUrl": src_url,
                    "blockTimestamp": reg_ts,
                    "exists": exists,
                }
        except Exception as e:
            logger.warning(f"Error retrieving on-chain record for {computed_keccak}: {e}")

        # 4. Compare canonical data vs on-chain record
        comparison = VerificationComparator.compare(canonical_dict, on_chain_record)
        hashes_match = comparison["hashes_match"]
        fields_match = comparison["fields_match"]
        on_chain_hash = comparison["on_chain_content_hash"] or computed_keccak

        block_number = on_chain_record.get("blockNumber", 0)
        block_timestamp = on_chain_record.get("blockTimestamp", 0)

        # 5. Check event verification if requested
        event_info: Dict[str, Any] = {"verified": False, "event_found": False}
        if check_events:
            rec_file = Path(receipt_path) if receipt_path else config.paths.tx_receipt_file
            if rec_file.exists():
                try:
                    receipt_data = json.loads(rec_file.read_text(encoding="utf-8"))
                    decoded_events = receipt_data.get("decodedEvents", [])
                    for ev in decoded_events:
                        if ev.get("event") == "PostRegistered":
                            args = ev.get("args", {})
                            ev_hash = str(args.get("contentHash", "")).lower()
                            if ev_hash == computed_keccak:
                                event_info = {
                                    "verified": True,
                                    "event_found": True,
                                    "event_name": "PostRegistered",
                                    "transaction_hash": receipt_data.get("transactionHash"),
                                    "block_number": receipt_data.get("blockNumber"),
                                    "event_args": args,
                                }
                                if not block_number:
                                    block_number = receipt_data.get("blockNumber", 0)
                                break
                except Exception as e:
                    logger.debug(f"Could not parse receipt file for event verification: {e}")

        # 6. Determine final status
        if hashes_match and fields_match:
            status = VerificationStatus.VERIFIED
            is_verified = True
            rationale = "On-chain record verified successfully: Cryptographic digests and all metadata match."
        elif hashes_match and not fields_match:
            status = VerificationStatus.TAMPER_DETECTED
            is_verified = False
            mismatches = comparison.get("mismatched_fields", [])
            rationale = f"Tamper detected: Hash matches but {len(mismatches)} metadata field(s) mutated."
        else:
            status = VerificationStatus.TAMPER_DETECTED
            is_verified = False
            rationale = f"Tamper detected: Computed hash {computed_keccak} differs from on-chain hash {on_chain_hash}."

        return BlockchainVerificationResult(
            verification_status=status,
            is_verified=is_verified,
            on_chain_exists=True,
            on_chain_content_hash=on_chain_hash,
            computed_content_hash=computed_keccak,
            hashes_match=hashes_match,
            on_chain_metadata=to_json_serializable(on_chain_record),
            event_verification=event_info,
            block_number=int(block_number),
            block_timestamp=int(block_timestamp),
            verification_timestamp=datetime.now(timezone.utc),
            rationale=rationale,
        )

    def save_verification_report(
        self,
        report_data: Union[Dict[str, Any], BlockchainVerificationResult],
        output_path: Optional[Path] = None,
    ) -> Path:
        """Saves verification result to artifacts/verification_report.json."""
        target = output_path or config.paths.verification_report_file
        target.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(report_data, BlockchainVerificationResult):
            payload = report_data.model_dump(mode="json")
        else:
            payload = to_json_serializable(report_data)

        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info(f"Saved verification report to {target}")
        return target

    def verify_canonical_data(
        self,
        canonical_input: Union[Path, str, Dict[str, Any], CanonicalMetadata],
        receipt_path: Optional[Union[Path, str]] = None,
        check_events: bool = True,
    ) -> BlockchainVerificationResult:
        """Alias for verify_post accepting canonical data / path."""
        return self.verify_post(
            post_data_or_path=canonical_input,
            receipt_path=receipt_path,
            check_events=check_events,
        )

    def verify(
        self,
        post_data_or_path: Optional[Union[Path, str, Dict[str, Any], CanonicalMetadata]] = None,
        receipt_path: Optional[Union[Path, str]] = None,
        check_events: bool = True,
    ) -> BlockchainVerificationResult:
        """Alias for verify_post."""
        return self.verify_post(
            post_data_or_path=post_data_or_path,
            receipt_path=receipt_path,
            check_events=check_events,
        )

    def verify_hash(self, content_hash: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Direct on-chain content hash query.
        Returns (exists_bool, on_chain_record_dict_or_None).
        """
        client = self._get_or_create_client()
        if client is None:
            return (False, None)
        try:
            is_reg = bool(client.is_registered(content_hash))
            if not is_reg:
                return (False, None)
            rec = client.get_post(content_hash) if hasattr(client, "get_post") else None
            return (True, rec)
        except Exception as e:
            logger.debug(f"verify_hash error: {e}")
            return (False, None)


# Alias for backward compatibility
BlockchainVerifier = VerificationEngine
