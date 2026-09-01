"""
app/blockchain/client.py - EVM Web3 client supporting multi-network switching,
account management, contract transactions, event decoding, and receipt persistence.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from eth_account import Account
from eth_account.signers.local import LocalAccount
from hexbytes import HexBytes
from web3 import Web3
from web3.exceptions import ContractCustomError, ContractLogicError

from app.blockchain.compiler import get_abi, get_bytecode
from app.blockchain.events import decode_contract_events, format_tx_receipt, to_json_serializable
from app.config import config

logger = logging.getLogger(__name__)

# Default Hardhat deterministic test private key (Account #0)
DEFAULT_DEV_PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"


class BlockchainClient:
    """
    Client for interacting with EVM smart contracts (Hardhat, Anvil, Polygon Amoy).
    Handles deployment, post registration, on-chain queries, and event parsing.
    """

    def __init__(
        self,
        network: Optional[str] = None,
        rpc_url: Optional[str] = None,
        chain_id: Optional[int] = None,
        private_key: Optional[str] = None,
        contract_address: Optional[str] = None,
        timeout: int = 30,
    ):
        self.network = network or config.effective_network
        self.rpc_url = rpc_url or config.effective_rpc_url
        self.chain_id = chain_id or config.effective_chain_id
        self.private_key = private_key or config.effective_private_key or DEFAULT_DEV_PRIVATE_KEY
        self.contract_address = contract_address or config.effective_contract_address
        self.timeout = timeout

        # Initialize Web3 instance
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url, request_kwargs={"timeout": self.timeout}))

        # Setup local signer account
        try:
            self.account: LocalAccount = Account.from_key(self.private_key)
            self.account_address: str = self.account.address
        except Exception as e:
            logger.warning(f"Could not load account from private key: {e}")
            self.account = None
            self.account_address = "0x0000000000000000000000000000000000000000"

        self.abi: List[Dict[str, Any]] = get_abi("FaceProvenanceRegistry")
        self.bytecode: str = get_bytecode("FaceProvenanceRegistry")
        self.contract = None

        if self.contract_address and self.abi:
            try:
                checksum_addr = Web3.to_checksum_address(self.contract_address)
                self.contract = self.w3.eth.contract(address=checksum_addr, abi=self.abi)
            except Exception as e:
                logger.debug(f"Could not initialize contract instance at {self.contract_address}: {e}")

    def check_connection(self, rpc_url: Optional[str] = None) -> bool:
        """
        Validates connection to the EVM RPC node. Raises ConnectionRefusedError on failure.
        """
        url = rpc_url or self.rpc_url
        client_w3 = self.w3 if not rpc_url else Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": 5}))

        try:
            connected = client_w3.is_connected()
        except Exception as e:
            raise ConnectionRefusedError(
                f"Could not connect to EVM RPC at {url}. Ensure 'npx hardhat node' or Anvil is running. Error: {e}"
            ) from e

        if not connected:
            raise ConnectionRefusedError(
                f"Could not connect to EVM RPC at {url}. Ensure 'npx hardhat node' or Anvil is running."
            )
        return True

    def set_contract_address(self, address: str) -> None:
        """
        Sets or updates the active contract address and initializes the contract instance.
        """
        self.contract_address = address
        checksum_addr = Web3.to_checksum_address(address)
        self.contract = self.w3.eth.contract(address=checksum_addr, abi=self.abi)

    def deploy_contract(
        self,
        contract_name: str = "FaceProvenanceRegistry",
        gas_limit: int = 3000000,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Deploys the smart contract to the configured EVM network.
        Returns (deployed_contract_address, formatted_receipt).
        """
        self.check_connection()

        abi = get_abi(contract_name)
        bytecode = get_bytecode(contract_name)

        if not bytecode or bytecode == "0x":
            raise ValueError(f"Bytecode for contract {contract_name} is empty. Compile contracts first.")

        contract_factory = self.w3.eth.contract(abi=abi, bytecode=bytecode)

        nonce = self.w3.eth.get_transaction_count(self.account.address)
        gas_price = self.w3.eth.gas_price

        construct_txn = contract_factory.constructor().build_transaction({
            "from": self.account.address,
            "nonce": nonce,
            "gas": gas_limit,
            "gasPrice": gas_price,
            "chainId": self.chain_id,
        })

        signed_txn = self.account.sign_transaction(construct_txn)
        raw_tx_bytes = getattr(signed_txn, "raw_transaction", None) or getattr(signed_txn, "rawTransaction", None)
        tx_hash = self.w3.eth.send_raw_transaction(raw_tx_bytes)

        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=self.timeout)
        deployed_address = receipt.contractAddress

        self.contract_address = deployed_address
        self.contract = self.w3.eth.contract(address=deployed_address, abi=abi)

        formatted_receipt = format_tx_receipt(
            receipt=receipt,
            contract_address=deployed_address,
            network_name=self.network,
            chain_id=self.chain_id,
            stored_content_hash=None,
            decoded_events=[],
        )

        logger.info(f"Contract {contract_name} deployed at {deployed_address} (tx: {tx_hash.hex()})")
        return deployed_address, formatted_receipt

    def _normalize_content_hash_bytes(self, content_hash: str) -> bytes:
        """
        Converts a 64 or 66 character hex content hash string into 32 raw bytes.
        """
        clean_hex = content_hash.lower()
        if clean_hex.startswith("0x"):
            clean_hex = clean_hex[2:]

        if len(clean_hex) != 64:
            raise ValueError(f"Content hash must be 32 bytes (64 hex chars), got {len(clean_hex)} chars: {content_hash}")

        return bytes.fromhex(clean_hex)

    def register_post(
        self,
        content_hash: str,
        source_url: str,
        provider: str,
        author: str,
        post_id: str,
        post_timestamp: int,
        gas_limit: int = 500000,
    ) -> Dict[str, Any]:
        """
        Executes registerPost transaction on the smart contract.
        Returns standardized transaction receipt dict with decoded events.
        """
        if not self.contract:
            raise RuntimeError("Contract address is not set on BlockchainClient. Deploy or set address first.")

        # Check for zero content hash
        hash_norm = content_hash.lower()
        if not hash_norm.startswith("0x"):
            hash_norm = "0x" + hash_norm

        if hash_norm == "0x" + "00" * 32 or hash_norm == "0x0":
            raise ValueError("Execution reverted: InvalidContentHash()")

        hash_bytes = self._normalize_content_hash_bytes(content_hash)

        # Check if record already exists to provide clear exception
        try:
            exists = self.is_registered(content_hash)
            if exists:
                raise ValueError(f"Execution reverted: RecordAlreadyExists({hash_norm})")
        except (ValueError, RuntimeError):
            raise
        except Exception:
            pass

        self.check_connection()

        nonce = self.w3.eth.get_transaction_count(self.account.address)
        gas_price = self.w3.eth.gas_price

        txn = self.contract.functions.registerPost(
            hash_bytes,
            str(source_url),
            str(provider),
            str(author),
            str(post_id),
            int(post_timestamp),
        ).build_transaction({
            "from": self.account.address,
            "nonce": nonce,
            "gas": gas_limit,
            "gasPrice": gas_price,
            "chainId": self.chain_id,
        })

        signed_txn = self.account.sign_transaction(txn)
        raw_tx_bytes = getattr(signed_txn, "raw_transaction", None) or getattr(signed_txn, "rawTransaction", None)

        try:
            tx_hash = self.w3.eth.send_raw_transaction(raw_tx_bytes)
        except (ContractCustomError, ContractLogicError) as e:
            err_str = str(e)
            if "InvalidContentHash" in err_str:
                raise ValueError("Execution reverted: InvalidContentHash()") from e
            if "RecordAlreadyExists" in err_str:
                raise ValueError(f"Execution reverted: RecordAlreadyExists({hash_norm})") from e
            raise

        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=self.timeout)

        # Check transaction execution status
        if receipt.get("status") == 0:
            raise RuntimeError(f"Transaction reverted on chain: {tx_hash.hex()}")

        decoded_events = decode_contract_events(self.contract, receipt)

        formatted_receipt = format_tx_receipt(
            receipt=receipt,
            contract_address=self.contract_address,
            network_name=self.network,
            chain_id=self.chain_id,
            stored_content_hash=hash_norm,
            decoded_events=decoded_events,
        )

        return formatted_receipt

    def get_post(self, content_hash: str) -> Dict[str, Any]:
        """
        Calls getPost(bytes32) on-chain view method and returns structured dictionary.
        Reverts if record not found.
        """
        if not self.contract:
            raise RuntimeError("Contract address is not set on BlockchainClient.")

        hash_norm = content_hash.lower()
        if not hash_norm.startswith("0x"):
            hash_norm = "0x" + hash_norm

        hash_bytes = self._normalize_content_hash_bytes(content_hash)

        try:
            record_tuple = self.contract.functions.getPost(hash_bytes).call()
        except (ContractCustomError, ContractLogicError, Exception) as e:
            err_str = str(e)
            if "RecordNotFound" in err_str or "reverted" in err_str:
                raise ValueError(f"Execution reverted: RecordNotFound({hash_norm})") from e
            raise ValueError(f"Execution reverted: RecordNotFound({hash_norm})") from e

        # PostRecord struct fields:
        # [0] contentHash, [1] sourceUrl, [2] provider, [3] author, [4] postId,
        # [5] postTimestamp, [6] blockTimestamp, [7] registrant, [8] exists
        ret_hash = record_tuple[0].hex() if isinstance(record_tuple[0], (bytes, HexBytes)) else str(record_tuple[0])
        if not ret_hash.startswith("0x"):
            ret_hash = "0x" + ret_hash

        record = {
            "contentHash": ret_hash.lower(),
            "sourceUrl": str(record_tuple[1]),
            "provider": str(record_tuple[2]),
            "searchProvider": str(record_tuple[2]),
            "author": str(record_tuple[3]),
            "postId": str(record_tuple[4]),
            "postTimestamp": int(record_tuple[5]),
            "blockTimestamp": int(record_tuple[6]),
            "registrant": str(record_tuple[7]),
            "registeredBy": str(record_tuple[7]),
            "exists": bool(record_tuple[8]),
        }
        return record

    def is_registered(self, content_hash: str) -> bool:
        """
        Checks whether a content hash is registered on-chain.
        """
        if not self.contract:
            return False

        hash_bytes = self._normalize_content_hash_bytes(content_hash)
        try:
            return bool(self.contract.functions.isRegistered(hash_bytes).call())
        except Exception:
            return False

    def verify_post(self, content_hash: str) -> Tuple[bool, int, str]:
        """
        Executes verifyPost on the smart contract.
        Returns (exists, registrationTimestamp, sourceUrl).
        """
        if not self.contract:
            raise RuntimeError("Contract address is not set on BlockchainClient.")

        hash_bytes = self._normalize_content_hash_bytes(content_hash)

        try:
            # If account is available, we can execute or call
            res = self.contract.functions.verifyPost(hash_bytes).call({"from": self.account.address})
            # Returns (exists, registrationTimestamp, sourceUrl)
            return (bool(res[0]), int(res[1]), str(res[2]))
        except Exception as e:
            logger.debug(f"verifyPost call error: {e}")
            # Fallback to get_post if available
            try:
                rec = self.get_post(content_hash)
                return (True, rec["blockTimestamp"], rec["sourceUrl"])
            except Exception:
                return (False, 0, "")

    def total_records(self) -> int:
        """
        Returns the total number of registered records on-chain.
        """
        if not self.contract:
            return 0
        try:
            return int(self.contract.functions.totalRecords().call())
        except Exception:
            try:
                return int(self.contract.functions.getTotalPosts().call())
            except Exception:
                return 0

    def save_tx_receipt(self, receipt: Dict[str, Any], output_path: Optional[Path] = None) -> Path:
        """
        Saves transaction receipt dict to artifacts/tx_receipt.json.
        """
        target = output_path or config.paths.tx_receipt_file
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(to_json_serializable(receipt), f, indent=2)
        logger.info(f"Saved transaction receipt to {target}")
        return target
