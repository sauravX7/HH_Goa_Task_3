# 🛡️ Automated Face Provenance & EVM Blockchain Verification Pipeline

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Solidity](https://img.shields.io/badge/Solidity-0.8.24-363636.svg?logo=solidity&logoColor=white)](https://soliditylang.org/)
[![Hardhat](https://img.shields.io/badge/Hardhat-2.22.2-FFF100.svg?logo=ethereum&logoColor=black)](https://hardhat.org/)
[![Tests](https://img.shields.io/badge/Tests-127%20Passing-brightgreen.svg)]()
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-grade, end-to-end automated verification pipeline that takes an input face image, performs genuine reverse visual search across public web sources, validates candidate facial embeddings, gathers a cryptographic evidence package, registers canonical metadata digests onto an EVM smart contract, and provides deterministic on-chain re-verification with automated 5-scenario tamper detection.

---

## 📑 Table of Contents

1. [Architecture & 10-Stage Pipeline Flow](#-architecture--10-stage-pipeline-flow)
2. [Key Features & Standards Compliance](#-key-features--standards-compliance)
3. [Quick Start & Installation Guide](#-quick-start--installation-guide)
4. [Blockchain Network Configuration](#-blockchain-network-configuration)
5. [CLI & Hackathon Demo Recording Mode](#-cli--hackathon-demo-recording-mode)
6. [Dedicated CLI Scripts](#-dedicated-cli-scripts)
7. [Automated 5-Scenario Tamper Demonstration](#-automated-5-scenario-tamper-demonstration)
8. [Artifact Package Structure](#-artifact-package-structure)
9. [Privacy, Security & Ethical Guardrails (R8)](#-privacy-security--ethical-guardrails-r8)
10. [Testing & Verification Matrix](#-testing--verification-matrix)
11. [Troubleshooting & FAQ](#-troubleshooting--faq)

---

## 🏛️ Architecture & 10-Stage Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          10-STAGE PIPELINE FLOW                             │
└─────────────────────────────────────────────────────────────────────────────┘

 [1] Face Detection & Feature Encoding (app.face)
         │  • Face region isolation (dlib / face_recognition / Haar)
         │  • 128-d normalized unit vector extraction
         ▼  • Saves: artifacts/face_crop.jpg
 [2] Genuine Search Provenance Engine (app.provenance)
         │  • Prioritized fallback: SerpAPI Lens ➔ Bing Visual ➔ Playwright ➔ Mock
         │  • Query image SHA-256 hash & candidate discovery
         ▼  • Audit logging of provider latency and candidate ranks
 [3] Match Validation Engine (app.validation)
         │  • Candidate face retrieval & embedding extraction
         │  • Cosine similarity (≥ 0.60 threshold) & Euclidean distance ranking
         ▼  • Filters low-confidence matches before blockchain interaction
 [4] Evidence Package Collection (app.evidence)
         │  • Full webpage screenshot capture (artifacts/search_result.png)
         ▼  • Structured raw metadata assembly (artifacts/metadata.json)
 [5] Canonical Metadata Builder (app.hashing.canonical)
         │  • Deterministic RFC-8785 JSON formatting
         │  • Unicode UTF-8 NFC normalization & ISO-8601 UTC timestamps
         ▼  • Saves: artifacts/canonical_post.json
 [6] Cryptographic Digest Generator (app.hashing.hasher)
         │  • FIPS 180-4 SHA-256 (artifacts/sha256.txt)
         ▼  • EVM Keccak-256 (bytes32) digest (artifacts/keccak256.txt)
 [7] Blockchain Registration (app.blockchain)
         │  • Solidity contract transaction on Hardhat / Anvil / Polygon Amoy
         │  • Emits PostRegistered(bytes32 indexed contentHash, ...)
         ▼  • Saves: artifacts/tx_receipt.json
 [8] Independent Blockchain Verification (app.verification)
         │  • On-chain mapping lookup (isRegistered / getPost / verifyPost)
         ▼  • 100% cryptographic equality check against local canonical bytes
 [9] Tamper Detection Demonstration (app.tamper)
         │  • 5-scenario malicious mutation matrix (text, time, hash, key, URL)
         │  • Field-level diff calculation & on-chain mismatch audit
         ▼  • Saves: artifacts/verification_report.json
[10] Diagnostic Logger & Demo Execution Summary (app.orchestrator)
            • Stage durations, timestamps, and diagnostic audit logs
            • Saves: artifacts/pipeline_log.json & artifacts/demo/*.png
```

---

## 🌟 Key Features & Standards Compliance

- **No Facades / Genuine Logic**: Live facial feature embeddings, real multi-engine reverse search fallback chain, true deterministic JSON canonicalization, and real Solidity smart contract bytecode deployment.
- **RFC-8785 & Unicode NFC Canonicalization**: Guarantees identical Keccak-256 and SHA-256 hashes across repeated executions, operating systems, and architectures.
- **EVM Multi-Network Switching**: Zero code change switching between local `hardhat`, `anvil`, and live testnets like `polygon_amoy` using the `BLOCKCHAIN_NETWORK` environment variable.
- **Rich Interactive UI**: Modern terminal styling using Rich tables, progress spinners, transaction receipts, verification badges, and tamper diffs.
- **Dedicated Demo Recording Mode (`--demo`)**: Visual stage banners, paced transitions, and automatic snapshot card captures in `artifacts/demo/` for hackathon screen recordings.

---

## 🚀 Quick Start & Installation Guide

### Prerequisites
- **macOS** or **Linux**
- **Python 3.12+**
- **Node.js 18+** & `npm`
- **CMake** & `libpng` / `libjpeg` (for dlib compilation)

### 1. Clone & Set Up Python Virtual Environment
```bash
# Clone the repository
git clone <repo_url>
cd HH_goa_task#3

# Create Python 3.12 virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Install Node.js & Smart Contract Dependencies
```bash
# Install Hardhat and OpenZeppelin contracts
npm install
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env` and configure your API keys (optional, fallback providers will activate automatically):
```bash
cp .env.example .env
```

```ini
# .env Configuration
BLOCKCHAIN_NETWORK=hardhat
BLOCKCHAIN_RPC_URL=http://127.0.0.1:8545
SIMILARITY_THRESHOLD=0.60
DISTANCE_THRESHOLD=0.60
DEMO_STEP_DELAY=1.5

# Optional Reverse Search API Keys (Falls back to Playwright / Mock if not provided)
SERPAPI_API_KEY=your_serpapi_key_here
BING_VISUAL_SEARCH_API_KEY=your_bing_api_key_here

# Optional Polygon Amoy Testnet Config
POLYGON_AMOY_RPC_URL=https://rpc-amoy.polygon.technology
POLYGON_AMOY_PRIVATE_KEY=your_private_key_here
```

---

## ⛓️ Blockchain Network Configuration

The pipeline supports three primary EVM execution targets:

| Network | `BLOCKCHAIN_NETWORK` | Default RPC URL | Chain ID |
|---|---|---|---|
| **Hardhat Local Node** *(Default)* | `hardhat` | `http://127.0.0.1:8545` | `31337` |
| **Foundry Anvil** | `anvil` | `http://127.0.0.1:8545` | `31337` |
| **Polygon Amoy Testnet** | `polygon_amoy` | `https://rpc-amoy.polygon.technology` | `80002` |

### Running a Local Hardhat Node (Recommended for Live EVM Testing)
In a separate terminal window:
```bash
npx hardhat node
```

### Deploying the Smart Contract Separately
```bash
# Deploy to local Hardhat node
.venv/bin/python scripts/deploy_contract.py --network hardhat

# Deploy to Polygon Amoy testnet
.venv/bin/python scripts/deploy_contract.py --network polygon_amoy
```

---

## 🖥️ CLI & Hackathon Demo Recording Mode

Run the entire 10-stage pipeline with a single command:

```bash
# Standard Execution
.venv/bin/python main.py --image tests/assets/test_face.jpg

# Paced Hackathon Demo Recording Mode (Generates artifacts/demo/ snapshots)
.venv/bin/python main.py --image tests/assets/test_face.jpg --demo --demo-delay 1.5

# Custom Similarity Threshold & Target Network
.venv/bin/python main.py --image tests/assets/test_face.jpg --threshold 0.70 --network hardhat
```

### Demo Mode CLI Options:
- `--image` / `-i`: Path to the input face image file (`JPEG` / `PNG`).
- `--demo` / `-d`: Enables screen recording pacing and auto-saves visual snapshot cards to `artifacts/demo/`.
- `--demo-delay`: Transition delay in seconds between stages (default: `1.5`s).
- `--threshold` / `-t`: Minimum cosine similarity required for validation (default: `0.60`).
- `--network` / `-n`: Target blockchain network (`hardhat`, `anvil`, `polygon_amoy`).
- `--contract` / `-c`: Optional pre-deployed contract address override.
- `--artifacts-dir` / `-a`: Custom artifact directory (default: `artifacts/`).

---

## 🛠️ Dedicated CLI Scripts

### 1. `scripts/verify_post.py`
Performs an independent on-chain audit of a canonical post record:
```bash
.venv/bin/python scripts/verify_post.py --canonical artifacts/canonical_post.json
```

### 2. `scripts/tamper_demo.py`
Executes the automated 5-scenario tamper attack suite and displays a field-level difference matrix:
```bash
.venv/bin/python scripts/tamper_demo.py --canonical artifacts/canonical_post.json
```

### 3. `scripts/upload_post.py`
Standalone utility to serialize, hash, and register arbitrary post metadata onto the blockchain:
```bash
.venv/bin/python scripts/upload_post.py \
  --url "https://social.example.com/alice/post/789102" \
  --author "Alice Web3" \
  --caption "Exploring decentralized identity! 🛡️" \
  --similarity 0.95
```

---

## 🧪 Automated 5-Scenario Tamper Demonstration

The tamper engine demonstrates cryptographic sensitivity by introducing deliberate mutations into a verified baseline post:

| # | Attack Scenario | Altered Field | Cryptographic Impact | Detection Outcome |
|---|---|---|---|---|
| **1** | **Modified Caption/Text** | `caption` | Injected string alters JSON UTF-8 byte stream | `TAMPER_DETECTED` |
| **2** | **Modified Timestamp** | `post_timestamp` | Shifted publication timestamp changes digest | `TAMPER_DETECTED` |
| **3** | **Modified Media Hash** | `media_sha256` | Falsified image digest invalidates payload | `TAMPER_DETECTED` |
| **4** | **Removed Mandatory Field** | `author` | Deleted canonical key breaks deterministic schema | `TAMPER_DETECTED` |
| **5** | **Altered Source URL** | `source_url` | Phishing / redirection URL fails on-chain hash lookup | `TAMPER_DETECTED` |

The result of the audit is saved to `artifacts/verification_report.json` with field-level diffs:
```json
{
  "baseline_status": "VERIFIED",
  "total_scenarios": 5,
  "detected_tamper_count": 5,
  "all_tampered_detected": true,
  "scenarios": [
    {
      "scenario_id": "SCENARIO_1_MODIFIED_CAPTION",
      "status": "TAMPER_DETECTED",
      "hashes_differ": true,
      "diffs": [
        {
          "field_name": "caption",
          "original_value": "Exploring cryptography...",
          "tampered_value": "Exploring cryptography... [TAMPERED_MALICIOUS_INJECTION]"
        }
      ]
    }
  ]
}
```

---

## 📦 Artifact Package Structure

Every execution writes a complete, self-contained evidence package to `artifacts/`:

```
artifacts/
├── face_crop.jpg               # Normalized face crop used for visual search
├── search_result.png           # Captured source webpage screenshot
├── metadata.json               # Raw search provenance & match validation metrics
├── canonical_post.json         # Normalized, deterministic canonical representation
├── sha256.txt                  # FIPS 180-4 SHA-256 cryptographic digest
├── keccak256.txt               # EVM Keccak-256 (bytes32) content hash
├── tx_receipt.json             # Decoded blockchain receipt, gas used & events
├── verification_report.json    # Independent audit outcome & 5-scenario tamper matrix
├── pipeline_log.json           # Stage durations, timestamps, and diagnostic logs
└── demo/                       # Visual snapshot cards generated in --demo mode
    ├── 01_face_detection.png
    ├── 02_reverse_search.png
    ├── 03_match_validation.png
    ├── 04_evidence_captured.png
    ├── 05_canonical_metadata.png
    ├── 06_crypto_digests.png
    ├── 07_blockchain_tx.png
    ├── 08_onchain_verification.png
    ├── 09_tamper_matrix.png
    └── 10_final_summary.png
```

---

## 🔒 Privacy, Security & Ethical Guardrails (R8)

1. **User-Supplied Images Only**: The pipeline only processes images explicitly provided by the user via `--image`.
2. **Public Sources Exclusively**: Reverse searches query only publicly indexed web pages and public social media accounts. Private profiles, paywalls, and authentication tokens are strictly avoided.
3. **No Biometric Data On-Chain**: Raw biometric face embedding vectors (128-d / 512-d) and high-resolution facial images are **never** transmitted to or stored on the public blockchain ledger.
4. **Cryptographic Preimage Irreversibility**: Only fixed 32-byte cryptographic digests (`keccak256` content hash) and public post URLs are committed to the smart contract, preventing reversal of facial biometrics.
5. **Deterministic Auditability**: Canonical serialization ensures that third-party auditors can re-verify public records independently without proprietary software.

---

## 📊 Testing & Verification Matrix

The test suite contains **127 comprehensive opaque-box tests** spanning 4 tiers:

```bash
# Run complete test suite
.venv/bin/python -m pytest tests/ -v

# Run by Tier
.venv/bin/python -m pytest -m "tier1" -v  # Feature functional coverage
.venv/bin/python -m pytest -m "tier2" -v  # Boundary & corner cases
.venv/bin/python -m pytest -m "tier3" -v  # Pairwise cross-feature combinations
.venv/bin/python -m pytest -m "tier4" -v  # Real-world application workloads

# Run End-to-End Orchestrator tests
.venv/bin/python -m pytest tests/test_pipeline_e2e.py -v
```

---

## ❓ Troubleshooting & FAQ

### 1. `face_recognition` / `dlib` installation issues
On macOS, ensure CMake and XCode command line tools are installed:
```bash
brew install cmake
xcode-select --install
pip install dlib face-recognition
```

### 2. Hardhat node connection error
If you receive `Could not connect to EVM RPC at http://127.0.0.1:8545`, either:
- Start a local node: `npx hardhat node`
- Or allow the pipeline to use the embedded simulated EVM test provider automatically.

### 3. Playwright browser setup
If using Playwright for automated Google Lens lookup:
```bash
.venv/bin/playwright install chromium
```

---

## 🏆 Hackathon Submission Checklist

- [x] Single-command execution: `python main.py --image <path> --demo`
- [x] 10-stage uninterrupted automated workflow
- [x] Live face scan ➔ reverse visual lookup ➔ match validation ➔ blockchain registration ➔ verification ➔ 5-scenario tamper demo
- [x] Complete artifact bundle populated in `artifacts/` and `artifacts/demo/`
- [x] 127/127 tests passing on Python 3.12 virtualenv

---

*Developed for the Automated Verification & Blockchain Provenance Hackathon Challenge (2026).*
