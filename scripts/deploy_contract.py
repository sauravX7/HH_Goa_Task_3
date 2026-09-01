#!/usr/bin/env python3
"""
scripts/deploy_contract.py - CLI tool to deploy FaceProvenanceRegistry smart contract.
Supports Hardhat, Anvil, and Polygon Amoy networks with automatic artifact compilation.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.blockchain.client import BlockchainClient
from app.blockchain.compiler import compile_with_hardhat, get_contract_artifact
from app.blockchain.deployer import deploy_provenance_registry
from app.config import config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deploy FaceProvenanceRegistry smart contract to EVM network."
    )
    parser.add_argument(
        "--network",
        "-n",
        type=str,
        default=config.effective_network,
        help="Target EVM network ('hardhat', 'anvil', 'polygon_amoy', 'localhost').",
    )
    parser.add_argument(
        "--rpc-url",
        "-r",
        type=str,
        default=None,
        help="Custom RPC URL override.",
    )
    parser.add_argument(
        "--chain-id",
        "-c",
        type=int,
        default=None,
        help="Custom Chain ID override.",
    )
    parser.add_argument(
        "--private-key",
        "-k",
        type=str,
        default=None,
        help="Account private key hex string.",
    )
    parser.add_argument(
        "--contract-name",
        type=str,
        default="FaceProvenanceRegistry",
        help="Contract name to deploy (FaceProvenanceRegistry or PostRegistry).",
    )
    parser.add_argument(
        "--output-receipt",
        "-o",
        type=Path,
        default=config.paths.tx_receipt_file,
        help="Path to save transaction receipt JSON.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=True,
        help="Compile Solidity contracts before deployment (default: True).",
    )
    parser.add_argument(
        "--no-compile",
        action="store_false",
        dest="compile",
        help="Skip compilation step.",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("deploy_contract")

    print("=" * 70)
    print(f"Deploying {args.contract_name} to EVM Network: {args.network}")
    print("=" * 70)

    if args.compile:
        logger.info("Verifying / Compiling Solidity smart contracts...")
        compile_with_hardhat()

    try:
        client = BlockchainClient(
            network=args.network,
            rpc_url=args.rpc_url,
            chain_id=args.chain_id,
            private_key=args.private_key,
        )

        logger.info(f"Target RPC: {client.rpc_url}")
        logger.info(f"Chain ID: {client.chain_id}")
        logger.info(f"Deployer Account: {client.account_address}")

        contract_addr, receipt = deploy_provenance_registry(
            network=args.network,
            client=client,
            contract_name=args.contract_name,
            save_receipt=True,
            receipt_output_path=args.output_receipt,
        )

        print("\n" + "=" * 70)
        print("🎉 DEPLOYMENT SUCCESSFUL!")
        print("=" * 70)
        print(f"  Contract Name:     {args.contract_name}")
        print(f"  Contract Address:  {contract_addr}")
        print(f"  Network:           {args.network} (Chain ID: {client.chain_id})")
        print(f"  Transaction Hash:  {receipt.get('transactionHash')}")
        print(f"  Block Number:      {receipt.get('blockNumber')}")
        print(f"  Gas Used:          {receipt.get('gasUsed')}")
        print(f"  Receipt Saved To:  {args.output_receipt}")
        print("=" * 70 + "\n")

        return 0

    except ConnectionRefusedError as e:
        logger.error(f"\n❌ RPC Connection Failed: {e}")
        logger.error("Troubleshooting: Start a local node using 'npx hardhat node' or anvil, then retry.")
        return 1
    except Exception as e:
        logger.error(f"\n❌ Deployment Failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
