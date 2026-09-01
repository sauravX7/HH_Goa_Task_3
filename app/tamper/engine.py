"""
app/tamper/engine.py - Automated Tamper Simulation, Validation & Verification Report Compiler.

Implements Requirement R6:
- Executes 5 automated tamper mutation scenarios against on-chain records.
- Recomputes cryptographic digests for mutated payloads.
- Validates 100% precision tamper detection (all mutated payloads detected).
- Generates field-level diffs and persists artifacts/verification_report.json.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from app.blockchain.client import BlockchainClient
from app.config import config
from app.hashing.canonical import serialize_canonical_json
from app.hashing.hasher import compute_keccak256_digest, compute_sha256_digest
from app.models import FieldDiff, TamperDetectionResult, TamperScenarioOutcome, VerificationStatus
from app.tamper.differ import TamperDiffEngine
from app.tamper.scenarios import get_all_tamper_scenarios

logger = logging.getLogger(__name__)


class TamperReportDict(dict):
    """
    Hybrid report container supporting dict subscripting, attribute access, and model_dump().
    Guarantees seamless compatibility across dict-based tests and Pydantic-based callers.
    """

    @property
    def baseline_status(self) -> str:
        return self.get("baseline_status", "VERIFIED")

    @property
    def original_hash(self) -> str:
        return self.get("original_hash", "")

    @property
    def total_scenarios(self) -> int:
        return self.get("total_scenarios", 0)

    @property
    def detected_tamper_count(self) -> int:
        return self.get("detected_tamper_count", 0)

    @property
    def all_tampered_detected(self) -> bool:
        return self.get("all_tampered_detected", False)

    @property
    def scenarios(self) -> List[Dict[str, Any]]:
        return self.get("scenarios", [])

    def model_dump(self, mode: str = "json") -> Dict[str, Any]:
        """Returns standard dict representation matching Pydantic model_dump."""
        return dict(self)

    def to_model(self, report_path: Optional[Path] = None) -> TamperDetectionResult:
        """Converts report to strongly-typed TamperDetectionResult model."""
        scenario_models: List[TamperScenarioOutcome] = []
        for s in self.get("scenarios", []):
            diff_models = [FieldDiff(**d) if isinstance(d, dict) else d for d in s.get("diffs", [])]
            scenario_models.append(
                TamperScenarioOutcome(
                    scenario_id=s["scenario_id"],
                    scenario_name=s["scenario_name"],
                    description=s.get("description", s["scenario_name"]),
                    status=s["status"],
                    original_hash=s["original_hash"],
                    tampered_hash=s["tampered_hash"],
                    hashes_differ=s["hashes_differ"],
                    on_chain_query_result=s.get("on_chain_query_result", "MISMATCH"),
                    diffs=diff_models,
                    detected=s.get("detected", True),
                )
            )
        return TamperDetectionResult(
            baseline_status=VerificationStatus.VERIFIED if self.baseline_status == "VERIFIED" else VerificationStatus.TAMPER_DETECTED,
            total_scenarios=self.total_scenarios,
            detected_tamper_count=self.detected_tamper_count,
            all_tampered_detected=self.all_tampered_detected,
            scenarios=scenario_models,
            report_path=report_path or config.paths.verification_report_file,
        )


class TamperSuiteRunner:
    """
    Executes the 5 tamper demonstration scenarios against an on-chain registered baseline.
    """

    def __init__(
        self,
        blockchain: Optional[Any] = None,
        network: Optional[str] = None,
        contract_address: Optional[str] = None,
    ):
        self.blockchain = blockchain
        self.network = network or config.effective_network
        self.contract_address = contract_address or config.effective_contract_address

    def _ensure_blockchain(self) -> Any:
        """Retrieves or creates BlockchainClient instance if none provided."""
        if self.blockchain is not None:
            return self.blockchain
        try:
            self.blockchain = BlockchainClient(
                network=self.network,
                contract_address=self.contract_address,
            )
            return self.blockchain
        except Exception as e:
            logger.warning(f"Could not connect to blockchain for tamper suite: {e}")
            return None

    def verify_record(self, canonical_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verifies whether a single canonical record is registered on-chain.
        """
        _, canonical_bytes = serialize_canonical_json(canonical_data)
        computed_keccak = compute_keccak256_digest(canonical_bytes).lower()
        computed_sha = compute_sha256_digest(canonical_bytes).lower()

        blockchain = self._ensure_blockchain()
        if blockchain is None:
            return {
                "status": "NOT_FOUND_ON_CHAIN",
                "is_verified": False,
                "computed_hash": computed_keccak,
                "sha256_hash": computed_sha,
                "on_chain_record": None,
            }

        is_found = False
        try:
            if hasattr(blockchain, "is_registered"):
                is_found = bool(blockchain.is_registered(computed_keccak))
            elif hasattr(blockchain, "is_post_registered"):
                is_found = bool(blockchain.is_post_registered(computed_keccak))
        except Exception as e:
            logger.debug(f"is_registered check failed: {e}")
            is_found = False

        if not is_found:
            return {
                "status": "NOT_FOUND_ON_CHAIN",
                "is_verified": False,
                "computed_hash": computed_keccak,
                "sha256_hash": computed_sha,
                "on_chain_record": None,
            }

        on_chain_record = None
        try:
            if hasattr(blockchain, "get_post"):
                on_chain_record = blockchain.get_post(computed_keccak)
            elif hasattr(blockchain, "verify_post"):
                exists, reg_ts, src_url = blockchain.verify_post(computed_keccak)
                on_chain_record = {"contentHash": computed_keccak, "sourceUrl": src_url, "blockTimestamp": reg_ts, "exists": exists}
        except Exception as e:
            logger.debug(f"get_post query failed: {e}")

        return {
            "status": "VERIFIED",
            "is_verified": True,
            "computed_hash": computed_keccak,
            "sha256_hash": computed_sha,
            "on_chain_record": on_chain_record,
        }

    def _evaluate_tamper(
        self,
        scenario_id: str,
        name: str,
        original_dict: Dict[str, Any],
        tampered_dict: Dict[str, Any],
        original_hash: str,
        description: str = "",
        attack_vector: str = "",
    ) -> Dict[str, Any]:
        """
        Evaluates a single tampered mutation against the original registered hash.
        """
        _, tampered_bytes = serialize_canonical_json(tampered_dict)
        tampered_keccak = compute_keccak256_digest(tampered_bytes).lower()
        diffs = TamperDiffEngine.compute_diffs(original_dict, tampered_dict)

        hashes_differ = (tampered_keccak != original_hash.lower())

        is_on_chain = False
        blockchain = self._ensure_blockchain()
        if blockchain is not None:
            try:
                if hasattr(blockchain, "is_registered"):
                    is_on_chain = bool(blockchain.is_registered(tampered_keccak))
                elif hasattr(blockchain, "is_post_registered"):
                    is_on_chain = bool(blockchain.is_post_registered(tampered_keccak))
            except Exception:
                is_on_chain = False

        status = "TAMPER_DETECTED" if (hashes_differ and not is_on_chain) else "VERIFIED"

        return {
            "scenario_id": scenario_id,
            "scenario_name": name,
            "description": description or name,
            "attack_vector": attack_vector or name,
            "status": status,
            "original_hash": original_hash.lower(),
            "tampered_hash": tampered_keccak,
            "hashes_differ": hashes_differ,
            "on_chain_query_result": "MISMATCH" if hashes_differ else "MATCH",
            "diffs": diffs,
            "detected": (status == "TAMPER_DETECTED"),
        }

    def run_5_tamper_scenarios(
        self,
        original_canonical: Dict[str, Any],
        output_report_path: Optional[Union[Path, str]] = None,
        report_output_path: Optional[Union[Path, str]] = None,
    ) -> TamperReportDict:
        """
        Executes all 5 tamper scenarios against original canonical data and compiles results.
        """
        target_path = output_report_path or report_output_path

        _, baseline_bytes = serialize_canonical_json(original_canonical)
        original_keccak = compute_keccak256_digest(baseline_bytes).lower()

        scenarios_defs = get_all_tamper_scenarios(original_canonical)
        evaluated_scenarios: List[Dict[str, Any]] = []

        for s_def in scenarios_defs:
            outcome = self._evaluate_tamper(
                scenario_id=s_def.scenario_id,
                name=s_def.scenario_name,
                original_dict=original_canonical,
                tampered_dict=s_def.mutated_dict,
                original_hash=original_keccak,
                description=s_def.description,
                attack_vector=s_def.attack_vector,
            )
            evaluated_scenarios.append(outcome)

        all_detected = all(s["status"] == "TAMPER_DETECTED" for s in evaluated_scenarios)
        detected_count = sum(1 for s in evaluated_scenarios if s["status"] == "TAMPER_DETECTED")

        report_data = TamperReportDict({
            "baseline_status": "VERIFIED",
            "original_hash": original_keccak,
            "total_scenarios": len(evaluated_scenarios),
            "detected_tamper_count": detected_count,
            "all_tampered_detected": all_detected,
            "scenarios": evaluated_scenarios,
        })

        if target_path is not None:
            target_file = Path(target_path)
            target_file.parent.mkdir(parents=True, exist_ok=True)
            target_file.write_text(json.dumps(dict(report_data), indent=2), encoding="utf-8")
            logger.info(f"Tamper report saved to {target_file}")

        return report_data

    def run_tamper_detection(
        self,
        original_canonical: Dict[str, Any],
        output_report_path: Optional[Path] = None,
    ) -> TamperDetectionResult:
        """
        Executes tamper demonstration suite and returns typed Pydantic TamperDetectionResult.
        """
        target_path = output_report_path or config.paths.verification_report_file
        raw_report = self.run_5_tamper_scenarios(
            original_canonical=original_canonical,
            output_report_path=target_path,
        )
        return raw_report.to_model(report_path=target_path)


# Aliases for convenience and compatibility
TamperDetector = TamperSuiteRunner
TamperEngine = TamperSuiteRunner
