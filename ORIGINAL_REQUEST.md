# Original User Request

## 2026-09-01T14:08:41Z

Build a production-grade, end-to-end automated verification pipeline that takes an input face image, performs genuine reverse image search to discover matching online social/web posts, performs automated facial embedding match validation, gathers a cryptographic evidence package, registers the structured record, events, and hash onto an EVM blockchain smart contract, and provides deterministic on-chain re-verification with automated tamper detection and a dedicated hackathon recording demo mode.

Working directory: /Users/ssv/HH_goa_task#3
Integrity mode: development

## Architecture Pipeline Flow

```
Face Input
    │
    ▼
Face Detection & Encoding
    │
    ▼
Reverse Search Engine (SerpAPI Lens / Bing Visual / Playwright Fallback)
    │
    ▼
Match Validation Engine (Candidate face embedding comparison & threshold ranking)
    │
    ▼
Search Provenance & Evidence Collector (artifacts/ & artifacts/demo/ generation)
    │
    ▼
Canonical Metadata Builder (Deterministic Unicode/UTC/Key-sorted normalization)
    │
    ▼
SHA-256 & Keccak-256 Hasher (sha256.txt, keccak256.txt)
    │
    ▼
Blockchain Registration (Solidity Smart Contract with Event Emission on Hardhat/Anvil/Amoy)
    │
    ▼
Blockchain Verification Engine (Event decoding & on-chain re-verification)
    │
    ▼
Tamper Detection Engine (5-scenario simulation & field-level diff reporting)
    │
    ▼
Artifact Logger & Demo CLI Output (pipeline_log.json, Rich/Typer UI, --demo pacing)
```

## Requirements

### R1. Face Identification & Feature Processing
Detect and extract facial regions and 128/512-d embeddings from an input image using an industry-standard library (e.g., `face_recognition`, `OpenCV`, or `InsightFace`). Generate a normalized crop of the detected face saved to `artifacts/face_crop.jpg` and prepare the search payload for visual lookup.

### R2. Genuine Search Provenance Engine
Perform a live, non-hardcoded reverse image search to locate matching web/social media post candidates. Implement a prioritized search strategy with automatic fallback:
1. Google Lens via SerpAPI
2. Bing Visual Search API
3. Playwright browser automation for Google Lens
4. Optional fallback providers (TinEye / Yandex)

Every search query must record a complete provenance object (provider name, query image hash, timestamp, query ID, returned URLs, candidate ranks, confidence scores, thumbnail URLs) persisted into `artifacts/metadata.json`.

### R3. Match Validation Engine
Insert a dedicated validation layer to verify candidate image authenticity before blockchain registration:
* Generate face embeddings for candidate images returned by the search provider.
* Compare candidate embeddings with the query face embedding using cosine similarity or Euclidean face distance.
* Rank candidates, select the highest-confidence match above a configurable threshold (e.g. similarity >= 0.60, configurable via `config.yaml` or `.env`), and reject below-threshold candidates.
* Persist validation metrics into `artifacts/metadata.json`: `similarity_score`, `provider_confidence`, `selected_rank`, `validation_status`, and `rejected_candidates`.

### R4. Search Provenance, Evidence Collection & Canonical Hashing
Automatically capture and persist an immutable evidence package for each execution:
* **Directory Outputs (`artifacts/`)**:
  - `face_crop.jpg` — Cropped face image used for search.
  - `search_result.png` — Screenshot of the discovered social/web post or source page.
  - `metadata.json` — Raw extracted structured metadata, match validation scores, and search provenance.
  - `canonical_post.json` — Normalized, deterministic JSON representation.
  - `sha256.txt` — SHA-256 cryptographic digest of canonical metadata.
  - `keccak256.txt` — Keccak-256 digest matching EVM on-chain representation.
  - `tx_receipt.json` — Blockchain transaction receipt, block number, gas used, decoded events, and contract address.
  - `verification_report.json` — Structured verification outcome, hash comparisons, and tamper analysis.
  - `pipeline_log.json` — Complete execution log with stage durations, timings, and diagnostic logs.
* **Canonical Serialization Rules**:
  - Sort JSON keys alphabetically.
  - Normalize Unicode strings to UTF-8 NFC.
  - Normalize all timestamps to ISO-8601 UTC.
  - Trim extraneous whitespace from all text fields.
  - Strip volatile/non-deterministic runtime fields.
  - Enforce deterministic ordering for arrays.

### R5. Blockchain Integrity Layer, Smart Contract & Event Logging
Develop and deploy a Solidity smart contract capable of storing structured post records with event logging:
* **On-Chain Schema**: Content hash (`bytes32`), source URL (`string`), search provider (`string`), author/account (`string`), post identifier (`string`), post timestamp (`uint256`), and block timestamp (`uint256`).
* **Event Logging**: Emit Solidity events on registration (`PostRegistered(bytes32 indexed contentHash, string sourceUrl, string provider, uint256 timestamp)`) and verification queries. CLI and `web3.py` must decode events and include them in `artifacts/tx_receipt.json`.
* **Configurable Deployment Targets**: Supported deployment targets must be configurable through an environment variable (`BLOCKCHAIN_NETWORK`) with supported values: `hardhat` (default), `anvil`, and `polygon_amoy`. The pipeline should use the selected network without code changes.
* **Output Persistence**: Record full transaction metadata (tx hash, contract address, block number, gas used, network name, stored hash, decoded events) into `artifacts/tx_receipt.json`.

### R6. Independent Verification & Tamper Detection Engine
Provide a dedicated integrity verification and tamper demonstration module:
1. **Verification Workflow**: Fetch on-chain record and events, reload local `canonical_post.json`, recompute SHA-256 and Keccak-256 digests, compare against on-chain values, and confirm match.
2. **Tamper Demonstration Scenarios**: Automated demonstration testing 5 tamper scenarios:
   - Modified caption/text
   - Modified timestamp
   - Modified image/media hash
   - Removed metadata field
   - Altered source URL
3. **Verification Report**: Save `artifacts/verification_report.json` detailing verification status (`VERIFIED` vs `TAMPER_DETECTED`), expected hash, computed hash, specific altered fields, block number, timestamp, and pass/fail rationale.
4. **Dedicated CLI Scripts**:
   - `scripts/verify_post.py` — Verifies given post data against on-chain records.
   - `scripts/tamper_demo.py` — Runs full automated tamper detection suite and reports field-level diffs.
   - `scripts/upload_post.py` — Standalone script to upload arbitrary post metadata to chain.

### R7. Pipeline Orchestrator, Error Recovery & Demo Recording Mode
Deliver a robust orchestration engine coordinating all 10 stages:
* **Orchestration & State Management**: Typed data passing between stages, stage duration timing, and structured diagnostic logging.
* **Failure Recovery & Retries**: Exponential backoff and graceful recovery for:
  - No face detected in input
  - No reverse search results found
  - Search provider unavailability / API rate limits
  - Browser automation timeouts
  - Blockchain deployment / transaction failure
  - Verification mismatches
* **Demo Mode for Screen Recording (`python main.py --image <path> --demo`)**:
  - Numbered visual stage banners with clean Rich/Typer progress output.
  - Paced stage transitions suitable for screen recording.
  - Automatic screenshot capture at major milestones saved to `artifacts/demo/`.
  - Highlighted transaction hashes, block numbers, decoded events, and verification badges.

### R8. Privacy, Security & Ethical Guardrails
* Process only user-supplied images.
* Query only publicly accessible web and social media sources (never bypass private account controls or paywalls).
* Store only cryptographic digests and minimal public metadata on-chain (avoid raw personal images on public ledgers).
* Provide clear documentation on privacy policies, rate limits, search limitations, and ethical considerations.

## Acceptance Criteria

### Face Processing, Search & Match Validation
- [ ] Pipeline isolates/encodes facial features from input images and outputs `artifacts/face_crop.jpg`.
- [ ] Search step executes live network/API/browser queries without hardcoded fallback results.
- [ ] Match Validation Engine compares candidate face embeddings, rejects low-confidence results, and logs similarity scores before any blockchain upload.

### Canonical Hashing & Determinism
- [ ] Canonical serialization follows all normalization rules (Unicode NFC, ISO-8601 UTC, sorted keys, whitespace trimming).
- [ ] Identical input produces identical SHA-256 and Keccak-256 hashes across repeated executions and platforms.
- [ ] Automated determinism tests pass.

### Blockchain Layer & Event Logging
- [ ] Solidity smart contract compiles and deploys to local EVM node / testnet via `BLOCKCHAIN_NETWORK` config.
- [ ] Post registration emits Solidity events, which are decoded via `web3.py` and saved to `artifacts/tx_receipt.json`.
- [ ] Verification reads data directly from the blockchain contract.

### Independent Verification & Tamper Detection
- [ ] Original untampered data returns status `VERIFIED` with matching on-chain hashes.
- [ ] Tampered metadata (altered text, date, image, URL, or missing fields) returns `TAMPER_DETECTED` and identifies changed fields.
- [ ] `verification_report.json` outputs exact hash comparisons and failure reasons.

### Orchestration, Error Recovery & Demo Mode
- [ ] Pipeline handles edge cases (no face, rate limits, network timeouts) gracefully with actionable diagnostic logs in `pipeline_log.json`.
- [ ] `--demo` mode executes end-to-end with paced transitions and auto-generates recording assets in `artifacts/demo/`.
- [ ] Repository is structured with `app/evidence/`, `app/hashing/`, `app/provenance/`, `app/validation/`, `app/orchestrator/`, `artifacts/`, `contracts/`, `scripts/`, and `tests/`.
- [ ] `README.md` covers full architecture, setup commands, blockchain configuration, verification workflow, tamper demo, screen recording guide, privacy guardrails, and known limitations.

### Hackathon Submission Success Criteria (Non-Negotiable)
- [ ] The entire pipeline executes from a single command (`python main.py --image <path> --demo`) without manual intervention between stages.
- [ ] The screen recording demonstrates: Face Scan → Live Search Result → Match Validation → Blockchain Registration → Blockchain Verification → Tamper Detection in one uninterrupted run.
- [ ] All generated artifacts required for judging are available inside the `artifacts/` directory immediately after execution.
