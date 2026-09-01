"""
app/blockchain - EVM smart contract compilation, deployment, Web3 client, and event processing.
"""

from app.blockchain.client import BlockchainClient
from app.blockchain.compiler import (
    compile_with_hardhat,
    compile_with_solcx,
    get_abi,
    get_bytecode,
    get_contract_artifact,
    load_artifact,
)
from app.blockchain.deployer import (
    deploy_provenance_registry,
    ensure_contract_deployed,
)
from app.blockchain.events import (
    decode_contract_events,
    decode_event_args,
    format_tx_receipt,
    to_json_serializable,
)

__all__ = [
    "BlockchainClient",
    "compile_with_hardhat",
    "compile_with_solcx",
    "get_abi",
    "get_bytecode",
    "get_contract_artifact",
    "load_artifact",
    "deploy_provenance_registry",
    "ensure_contract_deployed",
    "decode_contract_events",
    "decode_event_args",
    "format_tx_receipt",
    "to_json_serializable",
]
