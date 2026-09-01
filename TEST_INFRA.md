# TEST_INFRA.md — Dual-Track Testing Infrastructure Specification

## 1. Executive Summary & Testing Philosophy

The automated verification pipeline defined in `ORIGINAL_REQUEST.md` integrates computer vision (face detection and embedding extraction), multi-engine reverse visual search provenance, facial embedding similarity validation, deterministic canonical metadata normalization, dual cryptographic hashing (SHA-256 and Keccak-256), EVM Solidity smart contract registration and event logging, independent on-chain verification, 5-scenario tamper detection, and a paced screen-recording demo CLI.

To ensure total system integrity and reliability without hardcoded facades or shortcuts, this test infrastructure adheres to a **Dual-Track Testing Architecture**:
1. **Track 1: Progressive Milestone Testability** — Independent, self-contained unit and modular tests with rich mock fixtures, synthetic image generators, and mock EVM environments enabling isolated execution across any CI or offline development setup.
2. **Track 2: Comprehensive 4-Tier Opaque-Box Verification** — End-to-end evaluation validating functional coverage, adversarial boundary conditions, cross-module pairwise interactions, and full real-world application workflows against authoritative specifications.

---

## 2. 4-Tier Test Suite Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     4-TIER TEST SUITE ARCHITECTURE                      │
└─────────────────────────────────────────────────────────────────────────┘
   ▲
   │  ┌────────────────────────────────────────────────────────────────┐
   ├──┤ Tier 4: Real-World Application Workloads                       │
   │  │ • Happy path E2E pipeline (Scan -> Search -> Verify -> Tamper) │
   │  │ • Search provider fallback chain E2E (SerpAPI -> Bing -> Mock) │
   │  │ • Below-threshold rejection E2E (No invalid on-chain upload)   │
   │  │ • Demo screen-recording mode E2E with artifact snapshots       │
   │  │ • Standalone CLI scripts lifecycle (deploy, upload, verify)    │
   │  └────────────────────────────────────────────────────────────────┘
   │  ┌────────────────────────────────────────────────────────────────┐
   ├──┤ Tier 3: Cross-Feature Combinations                             │
   │  │ • Pairwise integration across adjacent and non-adjacent stages  │
   │  │ • R1 Face Crop <-> R2 Search Provenance Hash Binding           │
   │  │ • R1 Query Vector <-> R3 Candidate Embedding Distance          │
   │  │ • R3 Selected Match <-> R4 Canonical Metadata Builder          │
   │  │ • R4 Keccak-256 Digest <-> R5 EVM Content Hash & Event Logs    │
   │  │ • R5 On-Chain Event <-> R6 Verification & Tamper Diff Engine   │
   │  └────────────────────────────────────────────────────────────────┘
   │  ┌────────────────────────────────────────────────────────────────┐
   ├──┤ Tier 2: Boundary, Adversarial & Corner Cases                   │
   │  │ • >=5 edge/corner cases per feature (R1 to R8)                 │
   │  │ • Zero/blank/corrupted images, multi-face boundary clamping    │
   │  │ • API rate-limits (HTTP 429), network timeouts, empty results  │
   │  │ • Unicode NFC / decomposed accents, emojis, non-UTC offsets    │
   │  │ • Zero hash (bytes32(0)) reverts, duplicate registrations      │
   │  │ • All 5 tamper attack vectors + missing metadata fields        │
   │  └────────────────────────────────────────────────────────────────┘
   │  ┌────────────────────────────────────────────────────────────────┐
   └──┤ Tier 1: Feature Functional Coverage (>=5 tests per R1–R8)      │
      │ • Primary behavior & happy paths for each requirement          │
      │ • R1: Face detection, 128-d/512-d embeddings, crop generation  │
      │ • R2: SerpAPI, Bing, Playwright fallback, provenance logging    │
      │ • R3: Cosine similarity, Euclidean distance, candidate ranking │
      │ • R4: Unicode NFC, sorted keys, ISO-8601 UTC, SHA-256/Keccak  │
      │ • R5: Smart contract deploy, registerPost, PostRegistered logs │
      │ • R6: On-chain re-verification, 5-scenario tamper detection    │
      │ • R7: 10-stage orchestration, pipeline_log.json, demo UI       │
      │ • R8: Privacy guardrails (no PII on-chain, public web only)    │
      └────────────────────────────────────────────────────────────────┘
```

---

## 3. Requirements Traceability Matrix (R1 to R8)

| Requirement | Description | Target Test Module | Tier 1 (Coverage) | Tier 2 (Boundary) | Tier 3 (Cross) | Tier 4 (Workload) |
|---|---|---|---|---|---|---|
| **R1** | Face Identification & Feature Processing | `tests/test_face_detection.py` | >= 6 tests | >= 6 tests | Yes (R1+R2, R1+R3) | Happy Path / Demo |
| **R2** | Search Provenance Engine & Fallback | `tests/test_search_provenance.py` | >= 6 tests | >= 6 tests | Yes (R2+R3) | Fallback Chain |
| **R3** | Match Validation & Similarity Ranking | `tests/test_match_validation.py` | >= 5 tests | >= 6 tests | Yes (R3+R4) | Low-Confidence Rejection |
| **R4** | Evidence Collection & Canonical Hashing | `tests/test_canonical_hashing.py` | >= 7 tests | >= 6 tests | Yes (R4+R5) | Full Evidence Pipeline |
| **R5** | Blockchain Integrity Layer & Smart Contract | `tests/test_blockchain_integration.py` | >= 6 tests | >= 6 tests | Yes (R5+R6) | Registration & Audit |
| **R6** | Verification & 5-Scenario Tamper Engine | `tests/test_tamper_detection.py` | >= 7 tests | >= 5 tests | Yes (R6+R7) | Tamper Detection Flow |
| **R7** | Orchestration, Error Recovery & Demo Mode | `tests/test_pipeline_e2e.py` | >= 6 tests | >= 5 tests | Yes (R7+R8) | Screen Recording Demo |
| **R8** | Privacy, Security & Ethical Guardrails | `tests/test_pipeline_e2e.py` | >= 5 tests | >= 4 tests | Yes (R8+R5) | Privacy Compliance |

---

## 4. Expected Output Derivation & Authoritative Sources

Every single test case derives its expected assertions from authoritative mathematical definitions, cryptographic standards, or specifications in `ORIGINAL_REQUEST.md`:

1. **Facial Feature Embeddings (R1)**:
   - Feature vectors must be float arrays of dimension 128 (dlib/SFace) or 512 (InsightFace/ArcFace).
   - Normalized L2 norm: $\sqrt{\sum x_i^2} \approx 1.0 \pm 10^{-4}$.
   - Normalized face crop must be a valid RGB image saved to `artifacts/face_crop.jpg`.

2. **Search Provenance Schema (R2)**:
   - Search provenance recorded in `artifacts/metadata.json` must contain `provider_used`, `query_image_hash` (matching SHA-256 of `face_crop.jpg`), `query_timestamp` (ISO-8601 UTC), `query_id`, and `candidates` list.
   - Fallback chain must record each attempt, latency, and success status.

3. **Similarity Metrics (R3)**:
   - Cosine Similarity: $\text{CosineSim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$.
   - Euclidean Distance: $\text{Dist}(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|_2 = \sqrt{2 - 2 \cdot \text{CosineSim}(\mathbf{u}, \mathbf{v})}$ for unit vectors.
   - Matches with $\text{CosineSim} \ge \text{SIMILARITY_THRESHOLD}$ (default 0.60) are accepted; lower candidates are rejected and logged.

4. **Canonical Hashing Determinism (R4)**:
   - Unicode NFC Normalization: `unicodedata.normalize('NFC', text)` — decomposed accents (e.g. `e` + `\u0301`) must serialize identically to composed accents (`\u00e9`).
   - ISO-8601 UTC: `2026-09-01T14:30:00+05:30` -> `2026-09-01T09:00:00Z`.
   - Key Sorting: JSON dictionary keys sorted alphabetically at every level.
   - Separators: `(',', ':')` with no whitespace around punctuation.
   - SHA-256 Digest: Standard FIPS 180-4 `hashlib.sha256(canonical_bytes).hexdigest()`.
   - Keccak-256 Digest: Standard Ethereum Keccak-256 `Web3.keccak(canonical_bytes).hex()`.

5. **Solidity Smart Contract (R5)**:
   - Contract must implement struct `PostRecord(bytes32 contentHash, string sourceUrl, string provider, string author, string postId, uint256 postTimestamp, uint256 blockTimestamp, address registrant, bool exists)`.
   - Event `PostRegistered(bytes32 indexed contentHash, string sourceUrl, string provider, string author, string postId, uint256 postTimestamp, uint256 registrationTimestamp, address indexed registrant)`.
   - Reverts on `bytes32(0)` with `InvalidContentHash()` or duplicate registration with `RecordAlreadyExists(bytes32)`.

6. **Tamper Attack Matrix (R6)**:
   - All 5 specified tamper scenarios must trigger `TAMPER_DETECTED` with exact field diffs:
     - Scenario A: Modified caption (`caption != original`).
     - Scenario B: Modified timestamp (`post_timestamp != original`).
     - Scenario C: Modified media hash (`media_sha256 != original`).
     - Scenario D: Removed metadata field (`<MISSING>`).
     - Scenario E: Altered source URL (`source_url != original`).

---

## 5. Adversarial & Edge Case Verification Strategy

1. **Encoding & Escaping Integrity**:
   - Mixed UTF-8 scripts: Japanese Kanji (`漢字`), Cyrillic (`Тест`), Arabic (`اختبار`), Devanagari (`परीक्षण`), accented Latin (`é, à, ö, ñ`), and multi-byte emojis (`🛡️, 🔍, ⛓️, 📸`).
   - Escaping test cases for quotes (`"`), newlines (`\n`), backslashes (`\\`), tabs, and control characters.
2. **Boundary Conditions**:
   - Empty input images (0-byte file), extreme image resolutions (16x16 vs 4096x4096), grayscale (1 channel) and RGBA (4 channel) inputs.
   - Exact threshold equality test ($\text{similarity} = 0.6000$).
   - Multiple faces in image (ensuring deterministic primary face bounding box selection).
   - Zero search results found on web.
   - HTTP 429 rate limit simulation triggering fallback.
   - Smart contract zero hash (`0x0000...0000`) and duplicate hash collision handling.
3. **Robustness & Error Recovery**:
   - Network drop / RPC timeout simulation.
   - Headless browser missing display simulation.
   - Corrupted candidate thumbnail downloads (HTTP 404/500).

---

## 6. Pytest Configuration & Test Markers

The test suite utilizes `pytest` with registered custom markers configured in `pytest.ini` / `conftest.py`:

| Marker | Description |
|---|---|
| `tier1` | Tier 1 Feature Coverage tests |
| `tier2` | Tier 2 Boundary and Corner Case tests |
| `tier3` | Tier 3 Cross-Feature Combination tests |
| `tier4` | Tier 4 Real-World Application Workload tests |
| `r1` to `r8` | Requirement-specific tests (R1 through R8) |
| `e2e` | Full end-to-end integration workflows |
| `contract` | EVM smart contract and blockchain tests |
| `hashing` | Cryptographic canonicalization and hashing tests |

---

## 7. Execution Commands

```bash
# Run the complete test suite
.venv/bin/python -m pytest tests/ -v

# Run by Tier
.venv/bin/python -m pytest -m "tier1" -v
.venv/bin/python -m pytest -m "tier2" -v
.venv/bin/python -m pytest -m "tier3" -v
.venv/bin/python -m pytest -m "tier4" -v

# Run by Requirement Feature
.venv/bin/python -m pytest -m "r1" -v
.venv/bin/python -m pytest -m "r2" -v
.venv/bin/python -m pytest -m "r3" -v
.venv/bin/python -m pytest -m "r4" -v
.venv/bin/python -m pytest -m "r5" -v
.venv/bin/python -m pytest -m "r6" -v
.venv/bin/python -m pytest -m "r7" -v
.venv/bin/python -m pytest -m "r8" -v

# Run with detailed summary and execution timings
.venv/bin/python -m pytest tests/ --durations=10 -ra
```
