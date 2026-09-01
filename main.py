#!/usr/bin/env python3
"""
main.py - Root entry point for the Face Provenance & Blockchain Verification Pipeline.
Executes the full 10-stage pipeline with Rich visual UI and demo recording support.

Usage:
    python main.py --image <path_to_face.jpg> [--demo] [--threshold 0.60] [--network hardhat]
"""

import sys
from app.cli.app import cli_app

if __name__ == "__main__":
    cli_app()
