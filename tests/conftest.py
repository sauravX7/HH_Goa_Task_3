"""
tests/conftest.py - Pytest configuration, fixtures, synthetic image generators,
and mock EVM/RPC helpers for the automated face verification pipeline test suite.
"""

import hashlib
import json
import math
import os
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
import pytest

# Ensure root directory is in sys.path
ROOT_DIR = Path(__file__).parent.parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ============================================================================
# Synthetic Image Generators
# ============================================================================

def generate_face_image_pattern(
    width: int = 300,
    height: int = 300,
    face_bbox: Tuple[int, int, int, int] = (50, 50, 250, 250), # left, top, right, bottom
    face_color: Tuple[int, int, int] = (235, 190, 150),
    eye_color: Tuple[int, int, int] = (40, 40, 60),
    mouth_color: Tuple[int, int, int] = (180, 50, 50),
    bg_color: Tuple[int, int, int] = (245, 245, 250),
) -> Image.Image:
    """
    Generates a synthetic RGB image containing a clear, detectable face pattern.
    """
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    left, top, right, bottom = face_bbox
    fw = right - left
    fh = bottom - top

    # Draw face oval
    draw.ellipse([left, top, right, bottom], fill=face_color, outline=(150, 100, 70), width=2)

    # Eyes
    eye_radius = max(3, fw // 15)
    left_eye_center = (left + fw // 3, top + fh // 3)
    right_eye_center = (left + (2 * fw) // 3, top + fh // 3)

    draw.ellipse(
        [
            left_eye_center[0] - eye_radius,
            left_eye_center[1] - eye_radius,
            left_eye_center[0] + eye_radius,
            left_eye_center[1] + eye_radius,
        ],
        fill=eye_color,
    )
    draw.ellipse(
        [
            right_eye_center[0] - eye_radius,
            right_eye_center[1] - eye_radius,
            right_eye_center[0] + eye_radius,
            right_eye_center[1] + eye_radius,
        ],
        fill=eye_color,
    )

    # Nose
    nose_top = (left + fw // 2, top + fh // 2 - fh // 10)
    nose_bottom = (left + fw // 2, top + fh // 2 + fh // 10)
    draw.line([nose_top, nose_bottom], fill=(120, 80, 60), width=2)

    # Mouth
    mouth_box = [
        left + fw // 3,
        top + (2 * fh) // 3,
        left + (2 * fw) // 3,
        top + (2 * fh) // 3 + fh // 10,
    ]
    draw.arc(mouth_box, start=0, end=180, fill=mouth_color, width=3)

    return img


def generate_multi_face_image(
    width: int = 600,
    height: int = 300,
    face_count: int = 3,
) -> Image.Image:
    """
    Generates a synthetic image with multiple distinct faces.
    """
    img = Image.new("RGB", (width, height), (240, 240, 240))
    draw = ImageDraw.Draw(img)

    slot_width = width // face_count
    for i in range(face_count):
        left = i * slot_width + 20
        top = 40
        right = left + slot_width - 40
        bottom = top + 200

        # Draw face
        color = ((i * 50 + 150) % 255, (i * 70 + 120) % 255, (i * 90 + 90) % 255)
        draw.ellipse([left, top, right, bottom], fill=color, outline=(50, 50, 50), width=2)
        # Eyes
        draw.ellipse([left + 25, top + 50, left + 35, top + 60], fill=(20, 20, 20))
        draw.ellipse([right - 35, top + 50, right - 25, top + 60], fill=(20, 20, 20))
        # Mouth
        draw.line([left + 30, top + 120, right - 30, top + 120], fill=(150, 30, 30), width=2)

    return img


def generate_blank_image(width: int = 200, height: int = 200, color: Tuple[int, int, int] = (100, 150, 200)) -> Image.Image:
    """
    Generates a uniform landscape / background image with NO faces.
    """
    return Image.new("RGB", (width, height), color)


# ============================================================================
# Vector & Similarity Helpers
# ============================================================================

def generate_normalized_vector(dim: int = 128, seed: Optional[int] = None) -> List[float]:
    """
    Generates a unit-normalized random vector in R^dim.
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random
    vec = rng.randn(dim)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return [float(x) for x in vec]


def synthesize_candidate_vector_with_similarity(
    query_vector: List[float],
    target_similarity: float,
    seed: Optional[int] = 42,
) -> List[float]:
    """
    Synthesizes a unit vector v such that cosine_similarity(query_vector, v) == target_similarity.
    Formula: v = s * u + sqrt(1 - s^2) * u_perp
    """
    u = np.array(query_vector, dtype=np.float64)
    u = u / np.linalg.norm(u)
    dim = len(u)

    rng = np.random.RandomState(seed)
    random_vec = rng.randn(dim)
    # Gram-Schmidt orthogonalization
    proj = np.dot(random_vec, u) * u
    u_perp = random_vec - proj
    u_perp = u_perp / np.linalg.norm(u_perp)

    s = max(-1.0, min(1.0, target_similarity))
    v = s * u + math.sqrt(max(0.0, 1.0 - s * s)) * u_perp
    v = v / np.linalg.norm(v)
    return [float(x) for x in v]


def compute_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Computes cosine similarity between two float vectors."""
    a = np.array(vec1, dtype=np.float64)
    b = np.array(vec2, dtype=np.float64)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_euclidean_distance(vec1: List[float], vec2: List[float]) -> float:
    """Computes Euclidean distance between two float vectors."""
    a = np.array(vec1, dtype=np.float64)
    b = np.array(vec2, dtype=np.float64)
    return float(np.linalg.norm(a - b))


# ============================================================================
# Canonical Serialization & Hashing Reference Helpers
# ============================================================================

def normalize_text_nfc(text: str) -> str:
    """Normalizes string to Unicode NFC format and trims leading/trailing whitespace."""
    if not isinstance(text, str):
        return text
    return unicodedata.normalize("NFC", text).strip()


def normalize_iso8601_utc(ts_str: str) -> str:
    """
    Parses timestamp string in any standard format / timezone and normalizes to YYYY-MM-DDTHH:MM:SSZ.
    """
    ts_str = ts_str.strip()
    if ts_str.endswith("Z"):
        # Format: 2026-09-01T14:30:00Z
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    else:
        dt = datetime.fromisoformat(ts_str)
    
    dt_utc = dt.astimezone(timezone.utc)
    return dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def serialize_canonical_json(data: Dict[str, Any]) -> Tuple[str, bytes]:
    """
    Serializes a dictionary into canonical deterministic JSON according to R4 rules:
    - Unicode NFC normalization
    - Trimmed whitespace
    - Sorted keys
    - Standard separators (',', ':')
    - UTF-8 bytes output
    """
    def _normalize_obj(obj: Any) -> Any:
        if isinstance(obj, str):
            return normalize_text_nfc(obj)
        elif isinstance(obj, dict):
            return {k.strip(): _normalize_obj(v) for k, v in sorted(obj.items())}
        elif isinstance(obj, list):
            return [_normalize_obj(x) for x in obj]
        elif isinstance(obj, float):
            return round(obj, 6)
        return obj

    normalized = _normalize_obj(data)
    json_str = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return json_str, json_str.encode("utf-8")


def compute_sha256_digest(canonical_bytes: bytes) -> str:
    """Returns 64-character lowercase hex SHA-256 digest."""
    return hashlib.sha256(canonical_bytes).hexdigest().lower()


def compute_keccak256_digest(canonical_bytes: bytes) -> str:
    """
    Returns 66-character lowercase hex Keccak-256 digest with '0x' prefix matching EVM.
    Uses web3.Web3.keccak if available, else Cryptodome or sha3.
    """
    try:
        from web3 import Web3
        res = Web3.keccak(canonical_bytes).hex().lower()
        if not res.startswith("0x"):
            res = "0x" + res
        return res
    except Exception:
        from Crypto.Hash import keccak
        k = keccak.new(digest_bits=256)
        k.update(canonical_bytes)
        return "0x" + k.hexdigest().lower()


# ============================================================================
# Mock EVM Blockchain Registry Simulator
# ============================================================================

class MockBlockchainRegistry:
    """
    Simulates the Solidity FaceProvenanceRegistry / PostRegistry smart contract in memory.
    Implements all contract methods, event logs, revert errors, and query interfaces.
    """
    def __init__(self, contract_address: str = "0x5FbDB2315678afecb367f032d93F642f64180aa3", chain_id: int = 31337):
        self.contract_address = contract_address
        self.chain_id = chain_id
        self.network_name = "hardhat_mock"
        self._records: Dict[str, Dict[str, Any]] = {}
        self._registered_hashes: List[str] = []
        self._events: List[Dict[str, Any]] = []
        self._tx_receipts: Dict[str, Dict[str, Any]] = {}
        self.block_number = 100
        self.deployer_address = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"

    def register_post(
        self,
        content_hash: str,
        source_url: str,
        provider: str,
        author: str,
        post_id: str,
        post_timestamp: int,
        sender: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes registerPost transaction.
        Reverts on bytes32(0) or duplicate contentHash.
        """
        sender = sender or self.deployer_address
        content_hash_norm = content_hash.lower()
        if not content_hash_norm.startswith("0x"):
            content_hash_norm = "0x" + content_hash_norm

        if content_hash_norm == "0x" + "00" * 32 or content_hash_norm == "0x0":
            raise ValueError("Execution reverted: InvalidContentHash()")

        if content_hash_norm in self._records:
            raise ValueError(f"Execution reverted: RecordAlreadyExists({content_hash_norm})")

        self.block_number += 1
        current_block_time = int(time.time())

        record = {
            "contentHash": content_hash_norm,
            "sourceUrl": source_url,
            "provider": provider,
            "author": author,
            "postId": post_id,
            "postTimestamp": int(post_timestamp),
            "blockTimestamp": current_block_time,
            "registrant": sender,
            "exists": True,
        }
        self._records[content_hash_norm] = record
        self._registered_hashes.append(content_hash_norm)

        tx_hash = "0x" + hashlib.sha256(f"{content_hash_norm}:{self.block_number}".encode()).hexdigest()
        block_hash = "0x" + hashlib.sha256(f"block:{self.block_number}".encode()).hexdigest()

        event_log = {
            "event": "PostRegistered",
            "blockNumber": self.block_number,
            "transactionHash": tx_hash,
            "address": self.contract_address,
            "args": {
                "contentHash": content_hash_norm,
                "sourceUrl": source_url,
                "provider": provider,
                "author": author,
                "postId": post_id,
                "postTimestamp": int(post_timestamp),
                "registrationTimestamp": current_block_time,
                "registrant": sender,
            },
        }
        self._events.append(event_log)

        receipt = {
            "transactionHash": tx_hash,
            "blockNumber": self.block_number,
            "blockHash": block_hash,
            "gasUsed": 85420,
            "effectiveGasPrice": 1000000000,
            "contractAddress": self.contract_address,
            "from": sender,
            "status": 1,
            "network": self.network_name,
            "chainId": self.chain_id,
            "storedContentHash": content_hash_norm,
            "decodedEvents": [event_log],
        }
        self._tx_receipts[tx_hash] = receipt
        return receipt

    def get_post(self, content_hash: str) -> Dict[str, Any]:
        """Retrieves PostRecord from on-chain mapping or reverts if not found."""
        content_hash_norm = content_hash.lower()
        if not content_hash_norm.startswith("0x"):
            content_hash_norm = "0x" + content_hash_norm

        if content_hash_norm not in self._records:
            raise ValueError(f"Execution reverted: RecordNotFound({content_hash_norm})")

        return dict(self._records[content_hash_norm])

    def is_registered(self, content_hash: str) -> bool:
        """Returns True if contentHash exists in mapping."""
        content_hash_norm = content_hash.lower()
        if not content_hash_norm.startswith("0x"):
            content_hash_norm = "0x" + content_hash_norm
        return content_hash_norm in self._records

    def verify_post(self, content_hash: str, sender: Optional[str] = None) -> Tuple[bool, int, str]:
        """
        Executes verifyPost and emits PostVerified event.
        Returns (exists, registrationTimestamp, sourceUrl).
        """
        sender = sender or self.deployer_address
        content_hash_norm = content_hash.lower()
        if not content_hash_norm.startswith("0x"):
            content_hash_norm = "0x" + content_hash_norm

        is_found = content_hash_norm in self._records
        reg_time = self._records[content_hash_norm]["blockTimestamp"] if is_found else 0
        source_url = self._records[content_hash_norm]["sourceUrl"] if is_found else ""

        self._events.append({
            "event": "PostVerified",
            "blockNumber": self.block_number,
            "address": self.contract_address,
            "args": {
                "contentHash": content_hash_norm,
                "exists": is_found,
                "verifier": sender,
                "timestamp": int(time.time()),
            },
        })
        return (is_found, reg_time, source_url)

    def total_records(self) -> int:
        return len(self._registered_hashes)

    def get_events(self, event_name: Optional[str] = None) -> List[Dict[str, Any]]:
        if event_name:
            return [e for e in self._events if e.get("event") == event_name]
        return list(self._events)


# ============================================================================
# Pytest Fixtures
# ============================================================================

@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Provides an isolated temporary directory for test outputs."""
    return tmp_path


@pytest.fixture
def synthetic_face_image_path(tmp_path: Path) -> Path:
    """Generates a synthetic single-face image on disk."""
    path = tmp_path / "test_face_input.jpg"
    img = generate_face_image_pattern(width=300, height=300)
    img.save(path, format="JPEG", quality=95)
    return path


@pytest.fixture
def synthetic_multi_face_image_path(tmp_path: Path) -> Path:
    """Generates a synthetic multi-face image on disk."""
    path = tmp_path / "test_multi_face_input.jpg"
    img = generate_multi_face_image(width=600, height=300, face_count=3)
    img.save(path, format="JPEG", quality=95)
    return path


@pytest.fixture
def synthetic_blank_image_path(tmp_path: Path) -> Path:
    """Generates a landscape / non-face image on disk."""
    path = tmp_path / "test_blank_input.jpg"
    img = generate_blank_image(width=300, height=300)
    img.save(path, format="JPEG", quality=95)
    return path


@pytest.fixture
def mock_blockchain() -> MockBlockchainRegistry:
    """Provides a fresh instance of MockBlockchainRegistry."""
    return MockBlockchainRegistry()


@pytest.fixture
def sample_canonical_dict() -> Dict[str, Any]:
    """Provides a standard baseline canonical metadata dictionary."""
    return {
        "author": "Alice Web3",
        "caption": "Exploring cryptography and decentralized identity! 🛡️ #blockchain",
        "media_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "post_id": "post_789102",
        "post_timestamp": "2026-09-01T12:00:00Z",
        "search_provider": "serpapi_google_lens",
        "similarity_score": 0.8845,
        "source_url": "https://social.example.com/alice/status/789102",
    }


@pytest.fixture
def sample_search_candidates() -> List[Dict[str, Any]]:
    """Provides a mock list of reverse search candidate results."""
    return [
        {
            "rank": 1,
            "title": "Alice Web3 on Social: Exploring Cryptography",
            "source_url": "https://social.example.com/alice/status/789102",
            "thumbnail_url": "https://social.example.com/thumbnails/alice_thumb.jpg",
            "image_url": "https://social.example.com/media/alice_full.jpg",
            "snippet": "Exploring cryptography and decentralized identity! #blockchain",
            "author": "Alice Web3",
            "post_date": "2026-09-01T12:00:00Z",
            "provider_confidence": 0.94,
        },
        {
            "rank": 2,
            "title": "Bob Dev: Blockchain Identity Patterns",
            "source_url": "https://social.example.com/bob/status/456123",
            "thumbnail_url": "https://social.example.com/thumbnails/bob_thumb.jpg",
            "image_url": "https://social.example.com/media/bob_full.jpg",
            "snippet": "Discussing identity verification on EVM chains.",
            "author": "Bob Dev",
            "post_date": "2026-08-28T10:00:00Z",
            "provider_confidence": 0.65,
        },
        {
            "rank": 3,
            "title": "Random Meme Page - Lookalike photo",
            "source_url": "https://social.example.com/memes/999888",
            "thumbnail_url": "https://social.example.com/thumbnails/meme_thumb.jpg",
            "image_url": "https://social.example.com/media/meme_full.jpg",
            "snippet": "Funny lookalike photo collection",
            "author": "MemeBot",
            "post_date": "2026-07-15T08:00:00Z",
            "provider_confidence": 0.32,
        },
    ]
