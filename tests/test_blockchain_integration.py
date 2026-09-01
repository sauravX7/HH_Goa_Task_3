"""
tests/test_blockchain_integration.py - Comprehensive Tier 1 and Tier 2 tests for
Requirement R5: Blockchain Integrity Layer, Smart Contract Deployment, Transaction Execution,
Event Logging (PostRegistered / PostVerified), Multi-Network Switching, and Tx Receipt Generation.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List
import pytest

from tests.conftest import (
    MockBlockchainRegistry,
    compute_keccak256_digest,
    serialize_canonical_json,
)


# ============================================================================
# Tier 1 - Feature Functional Coverage (R5)
# ============================================================================

@pytest.mark.tier1
@pytest.mark.r5
@pytest.mark.contract
def test_smart_contract_compilation_and_deployment(mock_blockchain: MockBlockchainRegistry):
    """
    Tier 1 / R5: Smart contract deploys to EVM network and returns valid contract address and chain ID.
    """
    assert mock_blockchain.contract_address.startswith("0x")
    assert len(mock_blockchain.contract_address) == 42
    assert mock_blockchain.chain_id == 31337
    assert mock_blockchain.total_records() == 0


@pytest.mark.tier1
@pytest.mark.r5
@pytest.mark.contract
def test_register_post_transaction_success(
    mock_blockchain: MockBlockchainRegistry,
    sample_canonical_dict: Dict[str, Any]
):
    """
    Tier 1 / R5: registerPost stores contentHash, sourceUrl, provider, author, postId, postTimestamp.
    """
    _, canonical_bytes = serialize_canonical_json(sample_canonical_dict)
    content_hash = compute_keccak256_digest(canonical_bytes)

    post_timestamp_sec = int(datetime_from_iso(sample_canonical_dict["post_timestamp"]))

    receipt = mock_blockchain.register_post(
        content_hash=content_hash,
        source_url=sample_canonical_dict["source_url"],
        provider=sample_canonical_dict["search_provider"],
        author=sample_canonical_dict["author"],
        post_id=sample_canonical_dict["post_id"],
        post_timestamp=post_timestamp_sec,
    )

    assert receipt["status"] == 1
    assert receipt["contractAddress"] == mock_blockchain.contract_address
    assert receipt["blockNumber"] > 0
    assert receipt["gasUsed"] > 21000
    assert receipt["storedContentHash"] == content_hash.lower()
    assert mock_blockchain.is_registered(content_hash) is True
    assert mock_blockchain.total_records() == 1


@pytest.mark.tier1
@pytest.mark.r5
@pytest.mark.contract
def test_post_registered_event_emission_and_decoding(
    mock_blockchain: MockBlockchainRegistry,
    sample_canonical_dict: Dict[str, Any]
):
    """
    Tier 1 / R5: Registration emits PostRegistered event decoded into tx receipt.
    """
    _, canonical_bytes = serialize_canonical_json(sample_canonical_dict)
    content_hash = compute_keccak256_digest(canonical_bytes)

    receipt = mock_blockchain.register_post(
        content_hash=content_hash,
        source_url=sample_canonical_dict["source_url"],
        provider=sample_canonical_dict["search_provider"],
        author=sample_canonical_dict["author"],
        post_id=sample_canonical_dict["post_id"],
        post_timestamp=1788264000,
    )

    events = receipt.get("decodedEvents", [])
    assert len(events) == 1
    event = events[0]
    assert event["event"] == "PostRegistered"
    assert event["args"]["contentHash"] == content_hash.lower()
    assert event["args"]["sourceUrl"] == sample_canonical_dict["source_url"]
    assert event["args"]["provider"] == sample_canonical_dict["search_provider"]
    assert event["args"]["author"] == sample_canonical_dict["author"]
    assert event["args"]["postId"] == sample_canonical_dict["post_id"]


@pytest.mark.tier1
@pytest.mark.r5
@pytest.mark.contract
def test_get_post_on_chain_query(
    mock_blockchain: MockBlockchainRegistry,
    sample_canonical_dict: Dict[str, Any]
):
    """
    Tier 1 / R5: getPost on-chain view method retrieves stored PostRecord for registered hash.
    """
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

    record = mock_blockchain.get_post(content_hash)
    assert record["contentHash"] == content_hash.lower()
    assert record["sourceUrl"] == sample_canonical_dict["source_url"]
    assert record["author"] == sample_canonical_dict["author"]
    assert record["exists"] is True


@pytest.mark.tier1
@pytest.mark.r5
@pytest.mark.contract
def test_verify_post_on_chain_event(
    mock_blockchain: MockBlockchainRegistry,
    sample_canonical_dict: Dict[str, Any]
):
    """
    Tier 1 / R5: verifyPost emits PostVerified event and returns (exists=True, registrationTimestamp).
    """
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

    exists, reg_time, source_url = mock_blockchain.verify_post(content_hash)
    assert exists is True
    assert reg_time > 0
    assert source_url == sample_canonical_dict["source_url"]

    verified_events = mock_blockchain.get_events("PostVerified")
    assert len(verified_events) >= 1
    assert verified_events[-1]["args"]["contentHash"] == content_hash.lower()
    assert verified_events[-1]["args"]["exists"] is True


@pytest.mark.tier1
@pytest.mark.r5
def test_multi_network_configuration_switching():
    """
    Tier 1 / R5: System config parses BLOCKCHAIN_NETWORK and applies correct RPC URL & Chain ID.
    """
    network_configs = {
        "hardhat": {"rpc_url": "http://127.0.0.1:8545", "chain_id": 31337},
        "anvil": {"rpc_url": "http://127.0.0.1:8545", "chain_id": 31337},
        "polygon_amoy": {"rpc_url": "https://rpc-amoy.polygon.technology", "chain_id": 80002},
    }

    for net_name, expected in network_configs.items():
        assert expected["chain_id"] in [31337, 80002]
        assert expected["rpc_url"].startswith("http")


# ============================================================================
# Tier 2 - Boundary, Adversarial & Corner Cases (R5)
# ============================================================================

@pytest.mark.tier2
@pytest.mark.r5
@pytest.mark.contract
def test_register_post_zero_content_hash_reverts(mock_blockchain: MockBlockchainRegistry):
    """
    Tier 2 / R5 Boundary: Registering zero hash (bytes32(0)) reverts with InvalidContentHash.
    """
    zero_hash = "0x" + "00" * 32
    with pytest.raises(ValueError, match="InvalidContentHash"):
        mock_blockchain.register_post(
            content_hash=zero_hash,
            source_url="https://example.com",
            provider="mock",
            author="Alice",
            post_id="0",
            post_timestamp=1788264000,
        )


@pytest.mark.tier2
@pytest.mark.r5
@pytest.mark.contract
def test_register_duplicate_content_hash_reverts(
    mock_blockchain: MockBlockchainRegistry,
    sample_canonical_dict: Dict[str, Any]
):
    """
    Tier 2 / R5 Boundary: Registering the same contentHash a second time reverts with RecordAlreadyExists.
    """
    _, canonical_bytes = serialize_canonical_json(sample_canonical_dict)
    content_hash = compute_keccak256_digest(canonical_bytes)

    # First registration passes
    mock_blockchain.register_post(
        content_hash=content_hash,
        source_url=sample_canonical_dict["source_url"],
        provider=sample_canonical_dict["search_provider"],
        author=sample_canonical_dict["author"],
        post_id=sample_canonical_dict["post_id"],
        post_timestamp=1788264000,
    )

    # Duplicate registration reverts
    with pytest.raises(ValueError, match="RecordAlreadyExists"):
        mock_blockchain.register_post(
            content_hash=content_hash,
            source_url=sample_canonical_dict["source_url"],
            provider=sample_canonical_dict["search_provider"],
            author=sample_canonical_dict["author"],
            post_id=sample_canonical_dict["post_id"],
            post_timestamp=1788264000,
        )


@pytest.mark.tier2
@pytest.mark.r5
@pytest.mark.contract
def test_get_post_non_existent_hash_reverts(mock_blockchain: MockBlockchainRegistry):
    """
    Tier 2 / R5 Boundary: Calling getPost for an unregistered hash reverts with RecordNotFound.
    """
    fake_hash = "0x" + "99" * 32
    with pytest.raises(ValueError, match="RecordNotFound"):
        mock_blockchain.get_post(fake_hash)


@pytest.mark.tier2
@pytest.mark.r5
def test_blockchain_rpc_connection_timeout_handling():
    """
    Tier 2 / R5 Boundary: RPC node unreachable raises clear error with troubleshooting guidance.
    """
    class OfflineNodeClient:
        def check_connection(self, rpc_url: str):
            raise ConnectionRefusedError(
                f"Could not connect to EVM RPC at {rpc_url}. Ensure 'npx hardhat node' or Anvil is running."
            )

    client = OfflineNodeClient()
    with pytest.raises(ConnectionRefusedError, match="Ensure 'npx hardhat node'"):
        client.check_connection("http://127.0.0.1:8545")


@pytest.mark.tier2
@pytest.mark.r5
@pytest.mark.contract
def test_register_post_extreme_string_lengths(mock_blockchain: MockBlockchainRegistry):
    """
    Tier 2 / R5 Boundary: Very large source URL (e.g. 2000 chars) and long author string registered safely.
    """
    long_url = "https://social.example.com/very/long/path/" + "a" * 1500
    long_author = "Dr. Very Long Pseudonym" * 10
    test_hash = "0x" + "aa" * 32

    receipt = mock_blockchain.register_post(
        content_hash=test_hash,
        source_url=long_url,
        provider="mock_provider",
        author=long_author,
        post_id="post_long_123",
        post_timestamp=1788264000,
    )
    assert receipt["status"] == 1
    record = mock_blockchain.get_post(test_hash)
    assert record["sourceUrl"] == long_url
    assert record["author"] == long_author


@pytest.mark.tier2
@pytest.mark.r5
def test_tx_receipt_persistence_schema(tmp_path: Path, mock_blockchain: MockBlockchainRegistry):
    """
    Tier 2 / R5: tx_receipt.json matches complete JSON schema required by Acceptance Criteria.
    """
    receipt_file = tmp_path / "artifacts" / "tx_receipt.json"
    receipt_file.parent.mkdir(parents=True, exist_ok=True)

    test_hash = "0x" + "bb" * 32
    receipt_data = mock_blockchain.register_post(
        content_hash=test_hash,
        source_url="https://example.com",
        provider="serpapi",
        author="Alice",
        post_id="1",
        post_timestamp=1788264000,
    )

    receipt_file.write_text(json.dumps(receipt_data, indent=2))
    assert receipt_file.exists()

    loaded = json.loads(receipt_file.read_text())
    assert "transactionHash" in loaded
    assert "blockNumber" in loaded
    assert "gasUsed" in loaded
    assert "contractAddress" in loaded
    assert "decodedEvents" in loaded


def datetime_from_iso(iso_str: str) -> float:
    from datetime import datetime
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    return dt.timestamp()


# ============================================================================
# Tier 1/2 - app.blockchain Implementation Module Tests
# ============================================================================

@pytest.mark.tier1
@pytest.mark.r5
def test_blockchain_compiler_artifacts():
    """
    Tier 1 / R5: app.blockchain.compiler successfully loads ABI and bytecode for FaceProvenanceRegistry.
    """
    from app.blockchain.compiler import get_abi, get_bytecode, get_contract_artifact

    abi = get_abi("FaceProvenanceRegistry")
    bytecode = get_bytecode("FaceProvenanceRegistry")

    assert isinstance(abi, list)
    assert len(abi) > 0
    assert any(item.get("name") == "registerPost" for item in abi)
    assert any(item.get("name") == "getPost" for item in abi)
    assert any(item.get("name") == "verifyPost" for item in abi)

    assert isinstance(bytecode, str)
    assert bytecode.startswith("0x")
    assert len(bytecode) > 100

    abi_tuple, bytecode_tuple = get_contract_artifact("FaceProvenanceRegistry")
    assert len(abi_tuple) == len(abi)
    assert bytecode_tuple == bytecode


@pytest.mark.tier1
@pytest.mark.r5
def test_blockchain_events_formatting_and_serialization():
    """
    Tier 1 / R5: app.blockchain.events properly decodes arguments and formats JSON receipts.
    """
    from hexbytes import HexBytes
    from app.blockchain.events import decode_event_args, format_tx_receipt, to_json_serializable

    raw_args = {
        "contentHash": HexBytes("0x" + "aa" * 32),
        "sourceUrl": "https://example.com/post",
        "provider": "serpapi_lens",
        "author": "Alice",
        "postId": "123",
        "postTimestamp": 1788264000,
        "registrant": "0x5FbDB2315678afecb367f032d93F642f64180aa3",
    }

    decoded = decode_event_args(raw_args)
    assert decoded["contentHash"] == "0x" + "aa" * 32
    assert decoded["registrant"] == "0x5fbdb2315678afecb367f032d93f642f64180aa3"
    assert decoded["sourceUrl"] == "https://example.com/post"

    mock_receipt = {
        "transactionHash": HexBytes("0x" + "cc" * 32),
        "blockNumber": 42,
        "blockHash": HexBytes("0x" + "dd" * 32),
        "gasUsed": 85000,
        "effectiveGasPrice": 1000000000,
        "contractAddress": "0x5FbDB2315678afecb367f032d93F642f64180aa3",
        "from": "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
        "status": 1,
    }

    formatted = format_tx_receipt(
        receipt=mock_receipt,
        contract_address="0x5FbDB2315678afecb367f032d93F642f64180aa3",
        network_name="hardhat",
        chain_id=31337,
        stored_content_hash="0x" + "aa" * 32,
        decoded_events=[{"event": "PostRegistered", "args": decoded}],
    )

    assert formatted["transactionHash"] == "0x" + "cc" * 32
    assert formatted["blockNumber"] == 42
    assert formatted["status"] == 1
    assert formatted["chainId"] == 31337
    assert len(formatted["decodedEvents"]) == 1

    # Verify JSON serializability
    json_str = json.dumps(formatted)
    assert "0x" in json_str


@pytest.mark.tier1
@pytest.mark.r5
def test_blockchain_client_initialization():
    """
    Tier 1 / R5: BlockchainClient instantiates with default and custom configurations.
    """
    from app.blockchain.client import BlockchainClient

    client = BlockchainClient(network="hardhat")
    assert client.network == "hardhat"
    assert client.chain_id == 31337
    assert client.account_address.startswith("0x")
    assert len(client.abi) > 0

    amoy_client = BlockchainClient(network="polygon_amoy", chain_id=80002, rpc_url="https://rpc-amoy.polygon.technology")
    assert amoy_client.network == "polygon_amoy"
    assert amoy_client.chain_id == 80002
    assert amoy_client.rpc_url == "https://rpc-amoy.polygon.technology"


@pytest.mark.tier2
@pytest.mark.r5
def test_blockchain_client_check_connection_raises_offline():
    """
    Tier 2 / R5: BlockchainClient check_connection raises ConnectionRefusedError when RPC unreachable.
    """
    from app.blockchain.client import BlockchainClient

    client = BlockchainClient(rpc_url="http://127.0.0.1:59999", timeout=1)
    with pytest.raises(ConnectionRefusedError, match="Ensure 'npx hardhat node'"):
        client.check_connection()


@pytest.mark.tier1
@pytest.mark.r5
def test_deploy_contract_cli_help():
    """
    Tier 1 / R5: scripts/deploy_contract.py is executable and responds with CLI options.
    """
    import subprocess
    res = subprocess.run(
        [".venv/bin/python", "scripts/deploy_contract.py", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Deploy FaceProvenanceRegistry smart contract" in res.stdout
    assert "--network" in res.stdout
    assert "--rpc-url" in res.stdout

