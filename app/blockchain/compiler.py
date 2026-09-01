"""
app/blockchain/compiler.py - Smart contract compilation and artifact management.
Resolves Hardhat compiled artifacts (ABI & bytecode) with fallback to py-solc-x compilation.
"""

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONTRACTS_DIR = PROJECT_ROOT / "contracts"
HARDHAT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "contracts"


def get_artifact_path(contract_name: str = "FaceProvenanceRegistry") -> Path:
    """
    Returns the expected file path for the compiled Hardhat artifact JSON.
    """
    # Primary Hardhat artifact path
    primary_path = HARDHAT_ARTIFACTS_DIR / "contracts" / f"{contract_name}.sol" / f"{contract_name}.json"
    if primary_path.exists():
        return primary_path

    # Check direct artifacts directory
    alt_path = HARDHAT_ARTIFACTS_DIR / f"{contract_name}.json"
    if alt_path.exists():
        return alt_path

    # Check alternative contract source filename mapping
    for sol_file in CONTRACTS_DIR.glob("*.sol"):
        candidate = HARDHAT_ARTIFACTS_DIR / "contracts" / sol_file.name / f"{contract_name}.json"
        if candidate.exists():
            return candidate

    return primary_path


def compile_with_hardhat(cwd: Optional[Path] = None) -> bool:
    """
    Compiles Solidity smart contracts using Hardhat CLI (`npx hardhat compile`).
    """
    working_dir = cwd or PROJECT_ROOT
    npx_bin = shutil.which("npx")
    if not npx_bin:
        logger.warning("npx executable not found in PATH; cannot compile via Hardhat CLI.")
        return False

    try:
        cmd = [npx_bin, "hardhat", "compile"]
        logger.info(f"Running command: {' '.join(cmd)} in {working_dir}")
        res = subprocess.run(
            cmd,
            cwd=str(working_dir),
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(f"Hardhat compilation succeeded: {res.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Hardhat compilation failed with code {e.returncode}: {e.stderr or e.stdout}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error running hardhat compile: {e}")
        return False


def compile_with_solcx(
    contract_file_path: Optional[Path] = None,
    contract_name: str = "FaceProvenanceRegistry",
    solc_version: str = "0.8.20",
) -> Dict[str, Any]:
    """
    Fallback compilation using py-solc-x.
    """
    try:
        import solcx
        installed = solcx.get_installed_solc_versions()
        version_str = solc_version.lstrip("v")
        if not any(str(v).startswith(version_str) for v in installed):
            try:
                solcx.install_solc(solc_version)
            except Exception as e:
                logger.warning(f"Could not install solc {solc_version}: {e}")

        contract_path = contract_file_path or (CONTRACTS_DIR / f"{contract_name}.sol")
        if not contract_path.exists():
            raise FileNotFoundError(f"Contract file not found: {contract_path}")

        compiled = solcx.compile_files(
            [str(contract_path)],
            output_values=["abi", "bin"],
            solc_version=solc_version,
            optimize=True,
            optimize_runs=200,
        )

        for key, val in compiled.items():
            if contract_name in key:
                return {
                    "abi": val.get("abi", []),
                    "bytecode": "0x" + val.get("bin", "").lstrip("0x"),
                    "contractName": contract_name,
                }

        # Return first compiled item if key not matched exactly
        first_val = next(iter(compiled.values()))
        return {
            "abi": first_val.get("abi", []),
            "bytecode": "0x" + first_val.get("bin", "").lstrip("0x"),
            "contractName": contract_name,
        }
    except Exception as e:
        logger.error(f"py-solc-x compilation failed: {e}")
        raise RuntimeError(f"Solidity compilation failed: {e}") from e


def load_artifact(contract_name: str = "FaceProvenanceRegistry") -> Dict[str, Any]:
    """
    Loads compiled artifact JSON from Hardhat artifacts.
    If artifact is missing, triggers Hardhat compilation.
    """
    artifact_path = get_artifact_path(contract_name)
    if not artifact_path.exists():
        logger.info(f"Artifact {artifact_path} not found. Attempting hardhat compile...")
        compile_with_hardhat()

    if not artifact_path.exists():
        logger.info("Hardhat artifact still missing; attempting solcx compilation...")
        return compile_with_solcx(contract_name=contract_name)

    with open(artifact_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Normalize bytecode format (ensure 0x prefix)
    bytecode = data.get("bytecode", "")
    if isinstance(bytecode, str) and not bytecode.startswith("0x") and bytecode:
        data["bytecode"] = "0x" + bytecode

    return data


def get_abi(contract_name: str = "FaceProvenanceRegistry") -> List[Dict[str, Any]]:
    """
    Returns the ABI list for the given contract.
    """
    artifact = load_artifact(contract_name)
    return artifact.get("abi", [])


def get_bytecode(contract_name: str = "FaceProvenanceRegistry") -> str:
    """
    Returns the deployment bytecode string for the given contract.
    """
    artifact = load_artifact(contract_name)
    bytecode = artifact.get("bytecode", "")
    if not bytecode.startswith("0x") and bytecode:
        bytecode = "0x" + bytecode
    return bytecode


def get_contract_artifact(contract_name: str = "FaceProvenanceRegistry") -> Tuple[List[Dict[str, Any]], str]:
    """
    Convenience function returning (ABI, Bytecode) tuple for a contract.
    """
    artifact = load_artifact(contract_name)
    abi = artifact.get("abi", [])
    bytecode = artifact.get("bytecode", "")
    if not bytecode.startswith("0x") and bytecode:
        bytecode = "0x" + bytecode
    return abi, bytecode
