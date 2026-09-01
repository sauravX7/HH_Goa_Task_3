"""
tests/test_tamper_detection.py - Comprehensive Tier 1 and Tier 2 tests for
Requirement R6: Independent Verification Engine, 5-Scenario Tamper Attack Matrix,
Field-Level Diff Generation, and verification_report.json Output.
"""

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pytest

from tests.conftest import (
    MockBlockchainRegistry,
    compute_keccak256_digest,
    compute_sha256_digest,
    serialize_canonical_json,
)


class TamperDiffEngine:
    """Computes field-level differences between original canonical dict and tampered dict."""
    @staticmethod
    def compute_diffs(original: Dict[str, Any], tampered: Dict[str, Any]) -> List[Dict[str, Any]]:
        diffs = []
        all_keys = sorted(set(original.keys()) | set(tampered.keys()))
        for k in all_keys:
            orig_val = original.get(k)
            tamp_val = tampered.get(k)
            if k not in original:
                diffs.append({
                    "field_name": k,
                    "original_value": "<NOT_PRESENT>",
                    "tampered_value": tamp_val,
                    "impact_description": "Field was unexpectedly added",
                })
            elif k not in tampered:
                diffs.append({
                    "field_name": k,
                    "original_value": orig_val,
                    "tampered_value": "<MISSING>",
                    "impact_description": "Mandatory canonical field was deleted",
                })
            elif orig_val != tamp_val:
                diffs.append({
                    "field_name": k,
                    "original_value": orig_val,
                    "tampered_value": tamp_val,
                    "impact_description": f"Field value modified from '{orig_val}' to '{tamp_val}'",
                })
        return diffs


class TamperSuiteRunner:
    """Executes the 5 tamper demonstration scenarios against an on-chain registered baseline."""
    def __init__(self, blockchain: MockBlockchainRegistry):
        self.blockchain = blockchain

    def verify_record(self, canonical_data: Dict[str, Any]) -> Dict[str, Any]:
        _, canonical_bytes = serialize_canonical_json(canonical_data)
        computed_keccak = compute_keccak256_digest(canonical_bytes)
        computed_sha = compute_sha256_digest(canonical_bytes)

        is_found = self.blockchain.is_registered(computed_keccak)
        if not is_found:
            return {
                "status": "NOT_FOUND_ON_CHAIN",
                "is_verified": False,
                "computed_hash": computed_keccak,
                "sha256_hash": computed_sha,
                "on_chain_record": None,
            }

        on_chain_record = self.blockchain.get_post(computed_keccak)
        return {
            "status": "VERIFIED",
            "is_verified": True,
            "computed_hash": computed_keccak,
            "sha256_hash": computed_sha,
            "on_chain_record": on_chain_record,
        }

    def run_5_tamper_scenarios(self, original_canonical: Dict[str, Any]) -> Dict[str, Any]:
        _, baseline_bytes = serialize_canonical_json(original_canonical)
        original_keccak = compute_keccak256_digest(baseline_bytes)

        scenarios = []

        # Scenario 1: Modified caption
        t1 = copy.deepcopy(original_canonical)
        t1["caption"] = t1["caption"] + " [TAMPERED_MALICIOUS_INJECTION]"
        scenarios.append(self._evaluate_tamper("SCENARIO_1_MODIFIED_CAPTION", "Modified post caption/text", original_canonical, t1, original_keccak))

        # Scenario 2: Modified timestamp
        t2 = copy.deepcopy(original_canonical)
        t2["post_timestamp"] = "2026-09-01T15:00:00Z"
        scenarios.append(self._evaluate_tamper("SCENARIO_2_MODIFIED_TIMESTAMP", "Modified post timestamp (+3 hours)", original_canonical, t2, original_keccak))

        # Scenario 3: Modified media hash
        t3 = copy.deepcopy(original_canonical)
        t3["media_sha256"] = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
        scenarios.append(self._evaluate_tamper("SCENARIO_3_MODIFIED_MEDIA_HASH", "Modified image/media SHA-256 digest", original_canonical, t3, original_keccak))

        # Scenario 4: Removed metadata field
        t4 = copy.deepcopy(original_canonical)
        t4.pop("author", None)
        scenarios.append(self._evaluate_tamper("SCENARIO_4_REMOVED_FIELD", "Removed mandatory field 'author'", original_canonical, t4, original_keccak))

        # Scenario 5: Altered source URL
        t5 = copy.deepcopy(original_canonical)
        t5["source_url"] = "https://malicious-tampered-site.org/alice/fake_post/789102"
        scenarios.append(self._evaluate_tamper("SCENARIO_5_ALTERED_SOURCE_URL", "Altered source URL to fake domain", original_canonical, t5, original_keccak))

        all_detected = all(s["status"] == "TAMPER_DETECTED" for s in scenarios)

        return {
            "baseline_status": "VERIFIED",
            "original_hash": original_keccak,
            "total_scenarios": len(scenarios),
            "detected_tamper_count": sum(1 for s in scenarios if s["status"] == "TAMPER_DETECTED"),
            "all_tampered_detected": all_detected,
            "scenarios": scenarios,
        }

    def _evaluate_tamper(
        self,
        scenario_id: str,
        name: str,
        original_dict: Dict[str, Any],
        tampered_dict: Dict[str, Any],
        original_hash: str,
    ) -> Dict[str, Any]:
        _, tampered_bytes = serialize_canonical_json(tampered_dict)
        tampered_keccak = compute_keccak256_digest(tampered_bytes)
        diffs = TamperDiffEngine.compute_diffs(original_dict, tampered_dict)

        hashes_differ = (tampered_keccak != original_hash)
        is_on_chain = self.blockchain.is_registered(tampered_keccak)

        status = "TAMPER_DETECTED" if (hashes_differ and not is_on_chain) else "VERIFIED"

        return {
            "scenario_id": scenario_id,
            "scenario_name": name,
            "status": status,
            "original_hash": original_hash,
            "tampered_hash": tampered_keccak,
            "hashes_differ": hashes_differ,
            "diffs": diffs,
        }


# ============================================================================
# Tier 1 - Feature Functional Coverage (R6)
# ============================================================================

@pytest.mark.tier1
@pytest.mark.r6
def test_verification_workflow_untampered_baseline(
    mock_blockchain: MockBlockchainRegistry,
    sample_canonical_dict: Dict[str, Any]
):
    """
    Tier 1 / R6: Untampered canonical post verified against on-chain record yields status VERIFIED.
    """
    _, canonical_bytes = serialize_canonical_json(sample_canonical_dict)
    content_hash = compute_keccak256_digest(canonical_bytes)

    # Register on chain
    mock_blockchain.register_post(
        content_hash=content_hash,
        source_url=sample_canonical_dict["source_url"],
        provider=sample_canonical_dict["search_provider"],
        author=sample_canonical_dict["author"],
        post_id=sample_canonical_dict["post_id"],
        post_timestamp=1788264000,
    )

    runner = TamperSuiteRunner(mock_blockchain)
    result = runner.verify_record(sample_canonical_dict)

    assert result["status"] == "VERIFIED"
    assert result["is_verified"] is True
    assert result["computed_hash"] == content_hash.lower()


@pytest.mark.tier1
@pytest.mark.r6
def test_tamper_scenario_1_modified_caption(
    mock_blockchain: MockBlockchainRegistry,
    sample_canonical_dict: Dict[str, Any]
):
    """
    Tier 1 / R6: Tamper Scenario 1 (Modified caption) is detected with diff on caption.
    """
    _, canonical_bytes = serialize_canonical_json(sample_canonical_dict)
    content_hash = compute_keccak256_digest(canonical_bytes)
    mock_blockchain.register_post(content_hash, sample_canonical_dict["source_url"], "p", "a", "1", 1788264000)

    runner = TamperSuiteRunner(mock_blockchain)
    report = runner.run_5_tamper_scenarios(sample_canonical_dict)

    s1 = next(s for s in report["scenarios"] if s["scenario_id"] == "SCENARIO_1_MODIFIED_CAPTION")
    assert s1["status"] == "TAMPER_DETECTED"
    assert s1["hashes_differ"] is True
    assert len(s1["diffs"]) == 1
    assert s1["diffs"][0]["field_name"] == "caption"


@pytest.mark.tier1
@pytest.mark.r6
def test_tamper_scenario_2_modified_timestamp(
    mock_blockchain: MockBlockchainRegistry,
    sample_canonical_dict: Dict[str, Any]
):
    """
    Tier 1 / R6: Tamper Scenario 2 (Modified timestamp) is detected with diff on post_timestamp.
    """
    _, canonical_bytes = serialize_canonical_json(sample_canonical_dict)
    content_hash = compute_keccak256_digest(canonical_bytes)
    mock_blockchain.register_post(content_hash, sample_canonical_dict["source_url"], "p", "a", "1", 1788264000)

    runner = TamperSuiteRunner(mock_blockchain)
    report = runner.run_5_tamper_scenarios(sample_canonical_dict)

    s2 = next(s for s in report["scenarios"] if s["scenario_id"] == "SCENARIO_2_MODIFIED_TIMESTAMP")
    assert s2["status"] == "TAMPER_DETECTED"
    assert s2["diffs"][0]["field_name"] == "post_timestamp"


@pytest.mark.tier1
@pytest.mark.r6
def test_tamper_scenario_3_modified_media_hash(
    mock_blockchain: MockBlockchainRegistry,
    sample_canonical_dict: Dict[str, Any]
):
    """
    Tier 1 / R6: Tamper Scenario 3 (Modified media hash) is detected with diff on media_sha256.
    """
    _, canonical_bytes = serialize_canonical_json(sample_canonical_dict)
    content_hash = compute_keccak256_digest(canonical_bytes)
    mock_blockchain.register_post(content_hash, sample_canonical_dict["source_url"], "p", "a", "1", 1788264000)

    runner = TamperSuiteRunner(mock_blockchain)
    report = runner.run_5_tamper_scenarios(sample_canonical_dict)

    s3 = next(s for s in report["scenarios"] if s["scenario_id"] == "SCENARIO_3_MODIFIED_MEDIA_HASH")
    assert s3["status"] == "TAMPER_DETECTED"
    assert s3["diffs"][0]["field_name"] == "media_sha256"


@pytest.mark.tier1
@pytest.mark.r6
def test_tamper_scenario_4_removed_field(
    mock_blockchain: MockBlockchainRegistry,
    sample_canonical_dict: Dict[str, Any]
):
    """
    Tier 1 / R6: Tamper Scenario 4 (Removed metadata field) is detected with diff showing <MISSING>.
    """
    _, canonical_bytes = serialize_canonical_json(sample_canonical_dict)
    content_hash = compute_keccak256_digest(canonical_bytes)
    mock_blockchain.register_post(content_hash, sample_canonical_dict["source_url"], "p", "a", "1", 1788264000)

    runner = TamperSuiteRunner(mock_blockchain)
    report = runner.run_5_tamper_scenarios(sample_canonical_dict)

    s4 = next(s for s in report["scenarios"] if s["scenario_id"] == "SCENARIO_4_REMOVED_FIELD")
    assert s4["status"] == "TAMPER_DETECTED"
    assert s4["diffs"][0]["field_name"] == "author"
    assert s4["diffs"][0]["tampered_value"] == "<MISSING>"


@pytest.mark.tier1
@pytest.mark.r6
def test_tamper_scenario_5_altered_source_url(
    mock_blockchain: MockBlockchainRegistry,
    sample_canonical_dict: Dict[str, Any]
):
    """
    Tier 1 / R6: Tamper Scenario 5 (Altered source URL) is detected with diff on source_url.
    """
    _, canonical_bytes = serialize_canonical_json(sample_canonical_dict)
    content_hash = compute_keccak256_digest(canonical_bytes)
    mock_blockchain.register_post(content_hash, sample_canonical_dict["source_url"], "p", "a", "1", 1788264000)

    runner = TamperSuiteRunner(mock_blockchain)
    report = runner.run_5_tamper_scenarios(sample_canonical_dict)

    s5 = next(s for s in report["scenarios"] if s["scenario_id"] == "SCENARIO_5_ALTERED_SOURCE_URL")
    assert s5["status"] == "TAMPER_DETECTED"
    assert s5["diffs"][0]["field_name"] == "source_url"


@pytest.mark.tier1
@pytest.mark.r6
def test_verification_report_json_persistence(
    tmp_path: Path,
    mock_blockchain: MockBlockchainRegistry,
    sample_canonical_dict: Dict[str, Any]
):
    """
    Tier 1 / R6: artifacts/verification_report.json persists all baseline and 5 tamper outcomes.
    """
    report_file = tmp_path / "artifacts" / "verification_report.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)

    _, canonical_bytes = serialize_canonical_json(sample_canonical_dict)
    content_hash = compute_keccak256_digest(canonical_bytes)
    mock_blockchain.register_post(content_hash, sample_canonical_dict["source_url"], "p", "a", "1", 1788264000)

    runner = TamperSuiteRunner(mock_blockchain)
    report_data = runner.run_5_tamper_scenarios(sample_canonical_dict)
    report_file.write_text(json.dumps(report_data, indent=2))

    assert report_file.exists()
    loaded = json.loads(report_file.read_text())
    assert loaded["total_scenarios"] == 5
    assert loaded["detected_tamper_count"] == 5
    assert loaded["all_tampered_detected"] is True


# ============================================================================
# Tier 2 - Boundary, Adversarial & Corner Cases (R6)
# ============================================================================

@pytest.mark.tier2
@pytest.mark.r6
def test_tamper_detection_multiple_simultaneous_modifications(
    mock_blockchain: MockBlockchainRegistry,
    sample_canonical_dict: Dict[str, Any]
):
    """
    Tier 2 / R6 Boundary: Multiple fields tampered simultaneously (caption, author, post_id)
    are all enumerated in the field diff output.
    """
    t_multi = copy.deepcopy(sample_canonical_dict)
    t_multi["caption"] = "Tampered caption"
    t_multi["author"] = "Eve Hacker"
    t_multi["post_id"] = "post_999999"

    diffs = TamperDiffEngine.compute_diffs(sample_canonical_dict, t_multi)
    assert len(diffs) == 3
    tampered_field_names = [d["field_name"] for d in diffs]
    assert "caption" in tampered_field_names
    assert "author" in tampered_field_names
    assert "post_id" in tampered_field_names


@pytest.mark.tier2
@pytest.mark.r6
def test_tamper_detection_unregistered_hash_lookup(
    mock_blockchain: MockBlockchainRegistry,
    sample_canonical_dict: Dict[str, Any]
):
    """
    Tier 2 / R6 Boundary: Calling verify on completely unregistered post returns NOT_FOUND_ON_CHAIN.
    """
    runner = TamperSuiteRunner(mock_blockchain)
    result = runner.verify_record(sample_canonical_dict)

    assert result["status"] == "NOT_FOUND_ON_CHAIN"
    assert result["is_verified"] is False


@pytest.mark.tier2
@pytest.mark.r6
def test_tamper_differ_type_changes(sample_canonical_dict: Dict[str, Any]):
    """
    Tier 2 / R6 Boundary: Modifying value type (e.g. string post_id to int) detected cleanly.
    """
    t_type = copy.deepcopy(sample_canonical_dict)
    t_type["post_id"] = 789102 # Changed from str to int

    diffs = TamperDiffEngine.compute_diffs(sample_canonical_dict, t_type)
    assert len(diffs) == 1
    assert diffs[0]["field_name"] == "post_id"


@pytest.mark.tier2
@pytest.mark.r6
def test_tamper_differ_case_sensitivity(sample_canonical_dict: Dict[str, Any]):
    """
    Tier 2 / R6 Boundary: Case sensitivity variations (e.g. uppercase hex vs lowercase hex)
    produce hash divergence and field diff.
    """
    t_case = copy.deepcopy(sample_canonical_dict)
    t_case["media_sha256"] = sample_canonical_dict["media_sha256"].upper()

    diffs = TamperDiffEngine.compute_diffs(sample_canonical_dict, t_case)
    assert len(diffs) == 1
    assert diffs[0]["field_name"] == "media_sha256"


@pytest.mark.tier2
@pytest.mark.r6
def test_tamper_detection_empty_or_null_values(sample_canonical_dict: Dict[str, Any]):
    """
    Tier 2 / R6 Boundary: Changing non-empty field to None or empty string flagged as tampered.
    """
    t_empty = copy.deepcopy(sample_canonical_dict)
    t_empty["caption"] = ""

    diffs = TamperDiffEngine.compute_diffs(sample_canonical_dict, t_empty)
    assert len(diffs) == 1
    assert diffs[0]["tampered_value"] == ""


# ============================================================================
# Production Module Direct Integration Tests (app.verification & app.tamper)
# ============================================================================

@pytest.mark.tier1
@pytest.mark.r6
def test_app_verification_comparator_direct(sample_canonical_dict: Dict[str, Any]):
    """Tests app.verification.comparator with matching and divergent records."""
    from app.verification.comparator import VerificationComparator, compare_canonical_vs_onchain

    on_chain_rec = {
        "contentHash": "0x" + "11" * 32,
        "sourceUrl": sample_canonical_dict["source_url"],
        "provider": sample_canonical_dict["search_provider"],
        "author": sample_canonical_dict["author"],
        "postId": sample_canonical_dict["post_id"],
        "postTimestamp": 1788264000,
    }

    # Hash mismatch test
    res = compare_canonical_vs_onchain(sample_canonical_dict, on_chain_rec)
    assert res["is_match"] is False
    assert res["hashes_match"] is False

    # Matching hash test
    from app.hashing.hasher import compute_keccak256_digest
    _, raw_bytes = serialize_canonical_json(sample_canonical_dict)
    real_hash = compute_keccak256_digest(raw_bytes)
    on_chain_rec["contentHash"] = real_hash

    res_match = VerificationComparator.compare(sample_canonical_dict, on_chain_rec)
    assert res_match["hashes_match"] is True
    assert res_match["fields_match"] is True
    assert res_match["is_match"] is True


@pytest.mark.tier1
@pytest.mark.r6
def test_app_verification_engine_direct(
    tmp_path: Path,
    mock_blockchain: MockBlockchainRegistry,
    sample_canonical_dict: Dict[str, Any],
):
    """Tests app.verification.engine.VerificationEngine workflow."""
    from app.verification.engine import VerificationEngine
    from app.models import VerificationStatus

    _, canonical_bytes = serialize_canonical_json(sample_canonical_dict)
    content_hash = compute_keccak256_digest(canonical_bytes)

    verifier = VerificationEngine(client=mock_blockchain)

    # 1. Test before registration -> NOT_FOUND_ON_CHAIN
    res_not_found = verifier.verify_post(sample_canonical_dict)
    assert res_not_found.verification_status == VerificationStatus.NOT_FOUND_ON_CHAIN
    assert res_not_found.is_verified is False

    # 2. Register post
    mock_blockchain.register_post(
        content_hash=content_hash,
        source_url=sample_canonical_dict["source_url"],
        provider=sample_canonical_dict["search_provider"],
        author=sample_canonical_dict["author"],
        post_id=sample_canonical_dict["post_id"],
        post_timestamp=1788264000,
    )

    # 3. Verify registered post -> VERIFIED
    res_verified = verifier.verify_post(sample_canonical_dict)
    assert res_verified.verification_status == VerificationStatus.VERIFIED
    assert res_verified.is_verified is True
    assert res_verified.hashes_match is True

    # 4. Direct hash verification
    exists, rec = verifier.verify_hash(content_hash)
    assert exists is True
    assert rec is not None

    # 5. Save verification report
    report_file = tmp_path / "verification_report.json"
    saved = verifier.save_verification_report(res_verified, output_path=report_file)
    assert saved.exists()


@pytest.mark.tier1
@pytest.mark.r6
def test_app_tamper_engine_and_scenarios_direct(
    tmp_path: Path,
    mock_blockchain: MockBlockchainRegistry,
    sample_canonical_dict: Dict[str, Any],
):
    """Tests app.tamper engine, scenarios, and differ modules."""
    from app.tamper.differ import TamperDiffEngine
    from app.tamper.engine import TamperSuiteRunner
    from app.tamper.scenarios import get_all_tamper_scenarios

    # Check scenario generator produces 5 scenarios
    scenarios = get_all_tamper_scenarios(sample_canonical_dict)
    assert len(scenarios) == 5

    # Register original baseline
    _, canonical_bytes = serialize_canonical_json(sample_canonical_dict)
    content_hash = compute_keccak256_digest(canonical_bytes)
    mock_blockchain.register_post(
        content_hash=content_hash,
        source_url=sample_canonical_dict["source_url"],
        provider=sample_canonical_dict["search_provider"],
        author=sample_canonical_dict["author"],
        post_id=sample_canonical_dict["post_id"],
        post_timestamp=1788264000,
    )

    runner = TamperSuiteRunner(blockchain=mock_blockchain)
    report_file = tmp_path / "verification_report.json"
    report = runner.run_5_tamper_scenarios(
        original_canonical=sample_canonical_dict,
        output_report_path=report_file,
    )

    assert report.total_scenarios == 5
    assert report.detected_tamper_count == 5
    assert report.all_tampered_detected is True
    assert report["baseline_status"] == "VERIFIED"
    assert report_file.exists()

    # Test typed model conversion
    model_res = runner.run_tamper_detection(sample_canonical_dict, output_report_path=report_file)
    assert model_res.total_scenarios == 5
    assert model_res.detected_tamper_count == 5
    assert model_res.all_tampered_detected is True
