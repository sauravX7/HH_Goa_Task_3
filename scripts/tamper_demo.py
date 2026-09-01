#!/usr/bin/env python3
"""
scripts/tamper_demo.py - Automated 5-Scenario Tamper Attack Demonstration Suite.
Simulates text injection, timestamp forgery, media hash substitution, mandatory field deletion,
and URL redirection attacks against blockchain-registered baselines.

Usage:
    python scripts/tamper_demo.py [--canonical artifacts/canonical_post.json] [--network hardhat]
"""

import argparse
import json
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.blockchain.client import BlockchainClient
from app.cli.ui import ConsoleUI
from app.config import config
from app.tamper.engine import TamperDetector


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute 5-Scenario Automated Tamper Attack Demonstration."
    )
    parser.add_argument(
        "--canonical",
        "--post",
        "-c",
        "-p",
        type=Path,
        default=config.paths.canonical_post_file,
        dest="canonical",
        help="Path to baseline canonical_post.json (default: artifacts/canonical_post.json).",
    )
    parser.add_argument(
        "--network",
        "-n",
        type=str,
        default=config.effective_network,
        help="Target EVM network ('hardhat', 'anvil', 'polygon_amoy').",
    )
    parser.add_argument(
        "--contract",
        type=str,
        default=None,
        help="Deployed smart contract address override.",
    )
    parser.add_argument(
        "--output-report",
        "--output",
        "-o",
        type=Path,
        default=config.paths.verification_report_file,
        dest="output_report",
        help="Path to save verification_report.json.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if not args.canonical.exists():
        print(f"Error: Baseline canonical post file does not exist at {args.canonical}")
        return 1

    canonical_data = json.loads(args.canonical.read_text(encoding="utf-8"))

    client = BlockchainClient(
        network=args.network,
        contract_address=args.contract,
    )
    detector = TamperDetector(client)

    print("=" * 70)
    print("EXECUTING 5-SCENARIO TAMPER ATTACK DEMONSTRATION")
    print("=" * 70)

    result = detector.run_5_tamper_scenarios(
        original_canonical=canonical_data,
        report_output_path=args.output_report,
    )

    ui = ConsoleUI()
    ui.render_tamper_matrix(result.model_dump(mode="json"))

    print(f"\nReport persisted to: {args.output_report}")
    print(f"Total Scenarios Evaluated : {result.total_scenarios}")
    print(f"Tamper Scenarios Detected: {result.detected_tamper_count} / {result.total_scenarios}")

    if result.all_tampered_detected:
        print("\n" + "=" * 70)
        print("✓ SUCCESS: 100% of malicious tamper alterations detected by cryptographic hashing.")
        print("=" * 70)
        return 0
    else:
        print("\n" + "=" * 70)
        print("✗ FAILURE: Some tamper attacks went undetected.")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
