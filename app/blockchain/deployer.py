"""
app/blockchain/deployer.py - Automated smart contract deployment helper.
Deploys FaceProvenanceRegistry / PostRegistry onto target EVM network (Hardhat / Anvil / Polygon Amoy).
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from app.blockchain.client import BlockchainClient
from app.config import config

logger = logging.getLogger(__name__)


def deploy_provenance_registry(
    network: Optional[str] = None,
    client: Optional[BlockchainClient] = None,
    contract_name: str = "FaceProvenanceRegistry",
    save_receipt: bool = True,
    receipt_output_path: Optional[Path] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Deploys the smart contract onto the specified network.
    Returns (contract_address, formatted_tx_receipt).
    """
    cli = client or BlockchainClient(network=network)
    logger.info(f"Initiating deployment of {contract_name} on network '{cli.network}' (RPC: {cli.rpc_url})...")

    deployed_address, receipt = cli.deploy_contract(contract_name=contract_name)

    if save_receipt:
        out_path = receipt_output_path or config.paths.tx_receipt_file
        cli.save_tx_receipt(receipt, output_path=out_path)

    logger.info(f"Deployment complete. Contract address: {deployed_address}")
    return deployed_address, receipt


def ensure_contract_deployed(
    client: Optional[BlockchainClient] = None,
    network: Optional[str] = None,
    contract_name: str = "FaceProvenanceRegistry",
) -> Tuple[BlockchainClient, str]:
    """
    Ensures that a valid contract is deployed and ready for interaction.
    If contract_address is configured and contains code on-chain, reuses it;
    otherwise, triggers a new contract deployment.
    """
    cli = client or BlockchainClient(network=network)

    if cli.contract_address:
        try:
            cli.check_connection()
            code = cli.w3.eth.get_code(cli.w3.to_checksum_address(cli.contract_address))
            if code and len(code) > 2:
                logger.info(f"Reusing existing contract deployment at {cli.contract_address}")
                cli.set_contract_address(cli.contract_address)
                return cli, cli.contract_address
        except Exception as e:
            logger.warning(f"Could not verify existing contract code at {cli.contract_address}: {e}")

    # Deploy new contract instance
    deployed_address, _ = deploy_provenance_registry(client=cli, contract_name=contract_name)
    cli.set_contract_address(deployed_address)
    return cli, deployed_address
