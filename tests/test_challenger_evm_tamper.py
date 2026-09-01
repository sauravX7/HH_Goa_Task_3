"""
tests/test_challenger_evm_tamper.py - Adversarial Test Suite for Challenger 2
Tests live EVM contracts on Hardhat node, contract invariants, EVM client,
Tamper Engine adversarial scenarios, and multi-network configuration.
"""

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict
import pytest
from web3 import Web3
from web3.logs import EventLogErrorFlags

from app.blockchain.client import BlockchainClient
from app.blockchain.compiler import get_abi, get_bytecode
from app.blockchain.events import decode_event_args
from app.config import AppSettings, load_config
from app.hashing.canonical import serialize_canonical_json
from app.hashing.hasher import compute_keccak256_digest, compute_sha256_digest
from app.tamper.differ import TamperDiffEngine
from app.tamper.engine import TamperSuiteRunner
from app.tamper.scenarios import get_all_tamper_scenarios


@pytest.fixture(scope="module")
def live_blockchain_client():
    """
    Connects to the live local Hardhat node and deploys a fresh FaceProvenanceRegistry contract.
    """
    client = BlockchainClient(
        network="hardhat",
        rpc_url="http://127.0.0.1:8545",
        chain_id=31337,
    )
    client.check_connection()
    contract_addr, receipt = client.deploy_contract("FaceProvenanceRegistry")
    assert contract_addr.startswith("0x")
    assert len(contract_addr) == 42
    assert receipt["status"] == 1
    return client


@pytest.fixture
def sample_canonical():
    return {
        "author": "@satyanadella",
        "caption": "Excited to share the future of AI infrastructure at Microsoft Build 2026.",
        "media_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "post_id": "msft_post_1029384",
        "post_timestamp": "2026-09-01T12:00:00Z",
        "search_provider": "serpapi_lens",
        "similarity_score": 0.9421,
        "source_url": "https://twitter.com/satyanadella/status/189201928374",
    }


# ============================================================================
# Section 1: Live Solidity Contract Invariants & EVM Client Stress Testing
# ============================================================================

def test_live_contract_initial_state(live_blockchain_client: BlockchainClient):
    """
    Verify initial contract state on live node.
    """
    # Deploy another fresh instance to test 0 count
    client = BlockchainClient(
        network="hardhat",
        rpc_url="http://127.0.0.1:8545",
        chain_id=31337,
    )
    addr, _ = client.deploy_contract("FaceProvenanceRegistry")
    client.set_contract_address(addr)

    assert client.total_records() == 0
    hashes = client.contract.functions.getRegisteredHashes().call()
    assert len(hashes) == 0


def test_live_contract_register_post_and_raw_event_decoding(
    live_blockchain_client: BlockchainClient,
    sample_canonical: Dict[str, Any]
):
    """
    Register a post on live EVM node and verify transaction execution and raw event emission.
    Also documents the web3.py EventLogErrorFlags requirement for decode_contract_events.
    """
    _, canon_bytes = serialize_canonical_json(sample_canonical)
    keccak_hash = compute_keccak256_digest(canon_bytes)

    receipt = live_blockchain_client.register_post(
        content_hash=keccak_hash,
        source_url=sample_canonical["source_url"],
        provider=sample_canonical["search_provider"],
        author=sample_canonical["author"],
        post_id=sample_canonical["post_id"],
        post_timestamp=1788264000,
    )

    assert receipt["status"] == 1
    assert receipt["storedContentHash"] == keccak_hash.lower()
    assert receipt["blockNumber"] > 0
    assert receipt["gasUsed"] > 50000

    # Retrieve raw receipt from live chain to verify contract emitted the event
    raw_receipt = live_blockchain_client.w3.eth.get_transaction_receipt(receipt["transactionHash"])
    assert len(raw_receipt.logs) >= 1

    # Verify event decoding with proper EventLogErrorFlags
    event_cls = getattr(live_blockchain_client.contract.events, "PostRegistered")
    decoded_logs = event_cls().process_receipt(raw_receipt, errors=EventLogErrorFlags.Discard)
    assert len(decoded_logs) == 1

    decoded_args = decode_event_args(dict(decoded_logs[0]["args"]))
    assert decoded_args["contentHash"].lower() == keccak_hash.lower()
    assert decoded_args["sourceUrl"] == sample_canonical["source_url"]
    assert decoded_args["provider"] == sample_canonical["search_provider"]
    assert decoded_args["author"] == sample_canonical["author"]
    assert decoded_args["postId"] == sample_canonical["post_id"]
    assert decoded_args["postTimestamp"] == 1788264000
    assert decoded_args["registrant"].lower() == live_blockchain_client.account_address.lower()

    # Verify is_registered on chain
    assert live_blockchain_client.is_registered(keccak_hash) is True


def test_live_contract_duplicate_registration_reverts(
    live_blockchain_client: BlockchainClient,
    sample_canonical: Dict[str, Any]
):
    """
    Adversarial: Attempt duplicate post registration with same contentHash.
    Must revert with RecordAlreadyExists.
    """
    _, canon_bytes = serialize_canonical_json(sample_canonical)
    keccak_hash = compute_keccak256_digest(canon_bytes)

    # Ensure it's registered
    if not live_blockchain_client.is_registered(keccak_hash):
        live_blockchain_client.register_post(
            content_hash=keccak_hash,
            source_url=sample_canonical["source_url"],
            provider=sample_canonical["search_provider"],
            author=sample_canonical["author"],
            post_id=sample_canonical["post_id"],
            post_timestamp=1788264000,
        )

    # Duplicate registration attempt
    with pytest.raises(ValueError) as excinfo:
        live_blockchain_client.register_post(
            content_hash=keccak_hash,
            source_url="https://different-url.com",
            provider="other_provider",
            author="different_author",
            post_id="diff_id",
            post_timestamp=1788264000,
        )
    assert "RecordAlreadyExists" in str(excinfo.value)


def test_live_contract_zero_content_hash_reverts(live_blockchain_client: BlockchainClient):
    """
    Adversarial: Registering bytes32(0) must revert with InvalidContentHash.
    """
    zero_hash_1 = "0x" + "00" * 32
    zero_hash_2 = "00" * 32
    zero_hash_3 = "0x0"

    for zh in [zero_hash_1, zero_hash_3]:
        with pytest.raises(ValueError) as excinfo:
            live_blockchain_client.register_post(
                content_hash=zh,
                source_url="https://example.com",
                provider="serpapi",
                author="Alice",
                post_id="0",
                post_timestamp=1788264000,
            )
        assert "InvalidContentHash" in str(excinfo.value)


def test_live_contract_query_non_existent_hash(live_blockchain_client: BlockchainClient):
    """
    Adversarial: Querying non-existent content hash.
    - get_post must revert / raise ValueError(RecordNotFound)
    - is_registered must return False
    - verify_post must return (False, 0, "")
    """
    fake_hash = "0x" + "fa" * 32

    # is_registered check
    assert live_blockchain_client.is_registered(fake_hash) is False

    # get_post check
    with pytest.raises(ValueError) as excinfo:
        live_blockchain_client.get_post(fake_hash)
    assert "RecordNotFound" in str(excinfo.value)

    # verify_post check
    exists, reg_time, src_url = live_blockchain_client.verify_post(fake_hash)
    assert exists is False
    assert reg_time == 0
    assert src_url == ""


def test_live_contract_verify_post_event_emission(
    live_blockchain_client: BlockchainClient,
    sample_canonical: Dict[str, Any]
):
    """
    Calling verifyPost on-chain should return correct data and emit PostVerified event.
    """
    _, canon_bytes = serialize_canonical_json(sample_canonical)
    keccak_hash = compute_keccak256_digest(canon_bytes)

    exists, reg_time, src_url = live_blockchain_client.verify_post(keccak_hash)
    assert exists is True
    assert reg_time > 0
    assert src_url == sample_canonical["source_url"]


def test_live_contract_post_registry_wrapper_alias(live_blockchain_client: BlockchainClient):
    """
    Verify PostRegistry.sol inherits and operates identically to FaceProvenanceRegistry.
    """
    client = BlockchainClient(
        network="hardhat",
        rpc_url="http://127.0.0.1:8545",
        chain_id=31337,
    )
    addr, receipt = client.deploy_contract("PostRegistry")
    assert addr.startswith("0x")
    assert receipt["status"] == 1

    client.set_contract_address(addr)
    test_hash = "0x" + "ee" * 32
    receipt = client.register_post(
        content_hash=test_hash,
        source_url="https://postregistry.test/item/1",
        provider="test_provider",
        author="Tester",
        post_id="post_wrap_1",
        post_timestamp=1788264000,
    )
    assert receipt["status"] == 1
    assert client.is_registered(test_hash) is True
    rec = client.get_post(test_hash)
    assert rec["author"] == "Tester"
    assert rec["postId"] == "post_wrap_1"


def test_live_contract_multiple_sequential_registrations(live_blockchain_client: BlockchainClient):
    """
    Stress: Register 10 distinct hashes sequentially and verify indexing functions.
    """
    client = BlockchainClient(
        network="hardhat",
        rpc_url="http://127.0.0.1:8545",
        chain_id=31337,
    )
    addr, _ = client.deploy_contract("FaceProvenanceRegistry")
    client.set_contract_address(addr)

    registered_hashes = []
    for i in range(10):
        h = f"0x{i:02x}" + "ab" * 31
        registered_hashes.append(h)
        client.register_post(
            content_hash=h,
            source_url=f"https://example.com/post/{i}",
            provider="batch_tester",
            author=f"Author_{i}",
            post_id=f"id_{i}",
            post_timestamp=1788264000 + i * 100,
        )

    assert client.total_records() == 10
    on_chain_hashes = client.contract.functions.getRegisteredHashes().call()
    assert len(on_chain_hashes) == 10

    # Test getRecordByIndex
    for i in range(10):
        rec = client.contract.functions.getRecordByIndex(i).call()
        assert rec[3] == f"Author_{i}" # author
        assert rec[4] == f"id_{i}"     # postId

    # Test out-of-bounds index
    with pytest.raises(Exception):
        client.contract.functions.getRecordByIndex(10).call()


# ============================================================================
# Section 2: Multi-Network Configuration Switching
# ============================================================================

def test_multi_network_switching_hardhat_anvil_amoy():
    """
    Verify that BLOCKCHAIN_NETWORK environment variable correctly selects RPC and Chain ID.
    """
    # 1. Hardhat
    os.environ["BLOCKCHAIN_NETWORK"] = "hardhat"
    cfg = load_config()
    assert cfg.effective_network == "hardhat"
    assert cfg.effective_chain_id == 31337
    assert cfg.effective_rpc_url == "http://127.0.0.1:8545"

    # 2. Anvil
    os.environ["BLOCKCHAIN_NETWORK"] = "anvil"
    cfg = load_config()
    assert cfg.effective_network == "anvil"
    assert cfg.effective_chain_id == 31337
    assert cfg.effective_rpc_url == "http://127.0.0.1:8545"

    # 3. Polygon Amoy
    os.environ["BLOCKCHAIN_NETWORK"] = "polygon_amoy"
    cfg = load_config()
    assert cfg.effective_network == "polygon_amoy"
    assert cfg.effective_chain_id == 80002
    assert cfg.effective_rpc_url == "https://rpc-amoy.polygon.technology"

    # 4. Custom RPC URL override
    os.environ["BLOCKCHAIN_RPC_URL"] = "http://10.0.0.1:8545"
    cfg = load_config()
    assert cfg.effective_rpc_url == "http://10.0.0.1:8545"

    # Reset
    os.environ.pop("BLOCKCHAIN_NETWORK", None)
    os.environ.pop("BLOCKCHAIN_RPC_URL", None)


# ============================================================================
# Section 3: Tamper Engine Adversarial Stress Testing
# ============================================================================

def test_tamper_engine_against_live_blockchain(
    live_blockchain_client: BlockchainClient,
    sample_canonical: Dict[str, Any],
    tmp_path: Path,
):
    """
    Execute full 5-scenario tamper suite against live on-chain registration.
    Verify 100% detection rate and field-level diff precision.
    """
    _, canon_bytes = serialize_canonical_json(sample_canonical)
    keccak_hash = compute_keccak256_digest(canon_bytes)

    if not live_blockchain_client.is_registered(keccak_hash):
        live_blockchain_client.register_post(
            content_hash=keccak_hash,
            source_url=sample_canonical["source_url"],
            provider=sample_canonical["search_provider"],
            author=sample_canonical["author"],
            post_id=sample_canonical["post_id"],
            post_timestamp=1788264000,
        )

    runner = TamperSuiteRunner(blockchain=live_blockchain_client)
    report_file = tmp_path / "live_verification_report.json"
    report = runner.run_5_tamper_scenarios(sample_canonical, output_report_path=report_file)

    assert report.baseline_status == "VERIFIED"
    assert report.total_scenarios == 5
    assert report.detected_tamper_count == 5
    assert report.all_tampered_detected is True

    # Check each scenario specifics
    scenarios_by_id = {s["scenario_id"]: s for s in report["scenarios"]}

    # S1: Caption
    s1 = scenarios_by_id["SCENARIO_1_MODIFIED_CAPTION"]
    assert s1["status"] == "TAMPER_DETECTED"
    assert s1["hashes_differ"] is True
    assert s1["on_chain_query_result"] == "MISMATCH"
    assert any(d["field_name"] == "caption" for d in s1["diffs"])

    # S2: Timestamp
    s2 = scenarios_by_id["SCENARIO_2_MODIFIED_TIMESTAMP"]
    assert s2["status"] == "TAMPER_DETECTED"
    assert any(d["field_name"] == "post_timestamp" for d in s2["diffs"])

    # S3: Media SHA-256
    s3 = scenarios_by_id["SCENARIO_3_MODIFIED_MEDIA_HASH"]
    assert s3["status"] == "TAMPER_DETECTED"
    assert any(d["field_name"] == "media_sha256" for d in s3["diffs"])

    # S4: Removed field
    s4 = scenarios_by_id["SCENARIO_4_REMOVED_FIELD"]
    assert s4["status"] == "TAMPER_DETECTED"
    assert any(d["field_name"] == "author" and d["tampered_value"] == "<MISSING>" for d in s4["diffs"])

    # S5: Altered URL
    s5 = scenarios_by_id["SCENARIO_5_ALTERED_SOURCE_URL"]
    assert s5["status"] == "TAMPER_DETECTED"
    assert any(d["field_name"] == "source_url" for d in s5["diffs"])


def test_tamper_engine_multi_field_mutation_matrix(sample_canonical: Dict[str, Any]):
    """
    Adversarial: Test multiple field mutation combinations (3 fields altered at once).
    """
    mutated = dict(sample_canonical)
    mutated["caption"] = "Injected caption payload"
    mutated["author"] = "@malicious_impersonator"
    mutated["post_timestamp"] = "2020-01-01T00:00:00Z"

    diffs = TamperDiffEngine.compute_diffs(sample_canonical, mutated)
    assert len(diffs) == 3
    field_names = {d["field_name"] for d in diffs}
    assert field_names == {"caption", "author", "post_timestamp"}


def test_tamper_engine_added_field_detection(sample_canonical: Dict[str, Any]):
    """
    Adversarial: Inserting extra unauthorized fields into canonical payload.
    """
    mutated = dict(sample_canonical)
    mutated["unauthorized_field"] = "malicious_content"

    diffs = TamperDiffEngine.compute_diffs(sample_canonical, mutated)
    assert len(diffs) == 1
    assert diffs[0]["field_name"] == "unauthorized_field"
    assert diffs[0]["original_value"] == "<NOT_PRESENT>"
    assert diffs[0]["tampered_value"] == "malicious_content"
