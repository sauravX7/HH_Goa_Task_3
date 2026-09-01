"""
app/blockchain/events.py - Event log decoding and transaction receipt formatting.
Decodes PostRegistered and PostVerified events emitted by FaceProvenanceRegistry.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union
from hexbytes import HexBytes

logger = logging.getLogger(__name__)


def to_json_serializable(val: Any) -> Any:
    """
    Recursively converts HexBytes, bytes, and complex types to JSON-serializable primitives.
    """
    if isinstance(val, (bytes, HexBytes)):
        hex_str = val.hex()
        if not hex_str.startswith("0x"):
            hex_str = "0x" + hex_str
        return hex_str.lower()
    elif isinstance(val, dict):
        return {str(k): to_json_serializable(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple)):
        return [to_json_serializable(x) for x in val]
    elif hasattr(val, "__dict__"):
        return to_json_serializable(vars(val))
    return val


def decode_event_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes event arguments into standard format with hex strings and lowercase hashes.
    """
    clean_args: Dict[str, Any] = {}
    for k, v in args.items():
        if isinstance(v, (bytes, HexBytes)):
            h = v.hex().lower()
            clean_args[k] = "0x" + h if not h.startswith("0x") else h
        elif isinstance(v, str) and (k in ("contentHash", "registrant", "verifier") or k.endswith("Hash") or k.endswith("Address")):
            clean_args[k] = v.lower()
        else:
            clean_args[k] = to_json_serializable(v)
    return clean_args


def decode_contract_events(contract: Any, receipt: Union[Dict[str, Any], Any]) -> List[Dict[str, Any]]:
    """
    Decodes all known contract events from a Web3 transaction receipt.
    """
    decoded: List[Dict[str, Any]] = []

    # If receipt has raw AttributeDict or Web3 TxReceipt
    receipt_dict = dict(receipt) if hasattr(receipt, "items") else receipt

    tx_hash = receipt_dict.get("transactionHash")
    if isinstance(tx_hash, (bytes, HexBytes)):
        tx_hash = "0x" + tx_hash.hex().lower()

    block_number = receipt_dict.get("blockNumber", 0)
    contract_addr = getattr(contract, "address", "") or receipt_dict.get("contractAddress", "")

    # Iterate through event definitions on contract
    if hasattr(contract, "events"):
        for event_name in ["PostRegistered", "PostVerified"]:
            if hasattr(contract.events, event_name):
                try:
                    event_cls = getattr(contract.events, event_name)
                    try:
                        from web3.logs import EventLogErrorFlags
                        processed_events = event_cls().process_receipt(receipt, errors=EventLogErrorFlags.Discard)
                    except Exception:
                        processed_events = event_cls().process_receipt(receipt)
                    for pe in processed_events:
                        args = decode_event_args(dict(pe.get("args", {})))
                        # Also ensure provider/searchProvider compatibility
                        if "provider" in args and "searchProvider" not in args:
                            args["searchProvider"] = args["provider"]

                        event_dict = {
                            "event": event_name,
                            "blockNumber": pe.get("blockNumber", block_number),
                            "transactionHash": tx_hash or pe.get("transactionHash", ""),
                            "address": pe.get("address", contract_addr),
                            "args": args,
                        }
                        decoded.append(event_dict)
                except Exception as e:
                    logger.debug(f"Could not process event {event_name}: {e}")

    return decoded


def format_tx_receipt(
    receipt: Union[Dict[str, Any], Any],
    contract_address: str,
    network_name: str,
    chain_id: int,
    stored_content_hash: Optional[str] = None,
    decoded_events: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Builds the standardized tx_receipt.json data structure required by R5.
    """
    receipt_dict = dict(receipt) if hasattr(receipt, "items") else receipt

    # Normalize hashes and addresses
    tx_hash = receipt_dict.get("transactionHash", "")
    if isinstance(tx_hash, (bytes, HexBytes)):
        tx_hash = "0x" + tx_hash.hex().lower()
    elif isinstance(tx_hash, str) and not tx_hash.startswith("0x"):
        tx_hash = "0x" + tx_hash

    block_hash = receipt_dict.get("blockHash", "")
    if isinstance(block_hash, (bytes, HexBytes)):
        block_hash = "0x" + block_hash.hex().lower()

    from_addr = receipt_dict.get("from", "")
    if isinstance(from_addr, (bytes, HexBytes)):
        from_addr = "0x" + from_addr.hex().lower()

    contract_addr = contract_address or receipt_dict.get("contractAddress", "")
    if isinstance(contract_addr, (bytes, HexBytes)):
        contract_addr = "0x" + contract_addr.hex().lower()

    content_hash_clean = stored_content_hash or ""
    if content_hash_clean and not content_hash_clean.startswith("0x"):
        content_hash_clean = "0x" + content_hash_clean
    content_hash_clean = content_hash_clean.lower()

    status = receipt_dict.get("status", 1)
    if isinstance(status, str):
        try:
            status = int(status, 16 if status.startswith("0x") else 10)
        except ValueError:
            status = 1

    gas_used = receipt_dict.get("gasUsed", 0)
    effective_gas_price = receipt_dict.get("effectiveGasPrice", 0)

    events = decoded_events if decoded_events is not None else receipt_dict.get("decodedEvents", [])

    return {
        "transactionHash": str(tx_hash).lower(),
        "blockNumber": int(receipt_dict.get("blockNumber", 0)),
        "blockHash": str(block_hash).lower() if block_hash else "",
        "gasUsed": int(gas_used),
        "effectiveGasPrice": int(effective_gas_price),
        "contractAddress": str(contract_addr),
        "from": str(from_addr),
        "status": int(status),
        "network": str(network_name),
        "chainId": int(chain_id),
        "storedContentHash": content_hash_clean,
        "decodedEvents": to_json_serializable(events),
    }
