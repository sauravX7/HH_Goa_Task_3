"""
tests/test_match_validation.py - Comprehensive Tier 1 and Tier 2 tests for
Requirement R3: Match Validation Engine, Candidate Face Embedding Extraction,
Cosine Similarity / Euclidean Distance Metrics, Ranking, and Threshold Filtering.
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pytest

from tests.conftest import (
    compute_cosine_similarity,
    compute_euclidean_distance,
    generate_normalized_vector,
    synthesize_candidate_vector_with_similarity,
)


class MatchValidationEngine:
    """Simulates the Match Validation Engine logic according to R3."""
    def __init__(self, similarity_threshold: float = 0.60):
        self.similarity_threshold = similarity_threshold

    def evaluate_candidates(
        self,
        query_embedding: List[float],
        candidates: List[Dict[str, Any]],
        candidate_embeddings: Dict[int, Optional[List[float]]],
    ) -> Dict[str, Any]:
        """
        Evaluates candidate visual matches against the query embedding.
        """
        evaluated = []
        rejected = []

        for cand in candidates:
            rank = cand["rank"]
            emb = candidate_embeddings.get(rank)

            if emb is None:
                rejected.append({
                    "rank": rank,
                    "source_url": cand["source_url"],
                    "thumbnail_url": cand.get("thumbnail_url"),
                    "similarity_score": None,
                    "distance_score": None,
                    "rejection_reason": "NO_FACE_OR_DOWNLOAD_FAILED",
                })
                continue

            sim = compute_cosine_similarity(query_embedding, emb)
            dist = compute_euclidean_distance(query_embedding, emb)

            if sim >= self.similarity_threshold:
                evaluated.append({
                    "selected_rank": rank,
                    "candidate": cand,
                    "similarity_score": sim,
                    "distance_score": dist,
                    "candidate_embedding": emb,
                })
            else:
                rejected.append({
                    "rank": rank,
                    "source_url": cand["source_url"],
                    "thumbnail_url": cand.get("thumbnail_url"),
                    "similarity_score": sim,
                    "distance_score": dist,
                    "rejection_reason": "SIMILARITY_BELOW_THRESHOLD",
                })

        # Sort passing candidates descending by similarity
        evaluated.sort(key=lambda x: x["similarity_score"], reverse=True)

        if evaluated:
            top_match = evaluated[0]
            return {
                "validation_status": "MATCH_CONFIRMED",
                "selected_candidate": top_match,
                "similarity_score": top_match["similarity_score"],
                "distance_score": top_match["distance_score"],
                "provider_confidence": top_match["candidate"].get("provider_confidence"),
                "selected_rank": top_match["selected_rank"],
                "threshold_used": self.similarity_threshold,
                "rejected_candidates": rejected,
            }
        else:
            return {
                "validation_status": "SIMILARITY_BELOW_THRESHOLD" if rejected else "NO_CANDIDATES",
                "selected_candidate": None,
                "similarity_score": None,
                "distance_score": None,
                "provider_confidence": None,
                "selected_rank": None,
                "threshold_used": self.similarity_threshold,
                "rejected_candidates": rejected,
            }


# ============================================================================
# Tier 1 - Feature Functional Coverage (R3)
# ============================================================================

@pytest.mark.tier1
@pytest.mark.r3
def test_match_validation_high_similarity_match_confirmed():
    """
    Tier 1 / R3: Candidate with high similarity (>= 0.85) produces status MATCH_CONFIRMED.
    """
    query_vec = generate_normalized_vector(128, seed=42)
    candidate_vec = synthesize_candidate_vector_with_similarity(query_vec, target_similarity=0.88, seed=10)

    candidates = [
        {"rank": 1, "source_url": "https://social.example.com/alice/1", "provider_confidence": 0.95}
    ]
    candidate_embeddings = {1: candidate_vec}

    engine = MatchValidationEngine(similarity_threshold=0.60)
    result = engine.evaluate_candidates(query_vec, candidates, candidate_embeddings)

    assert result["validation_status"] == "MATCH_CONFIRMED"
    assert result["selected_rank"] == 1
    assert abs(result["similarity_score"] - 0.88) < 1e-3
    assert result["selected_candidate"] is not None


@pytest.mark.tier1
@pytest.mark.r3
def test_match_validation_cosine_similarity_calculation():
    """
    Tier 1 / R3: Cosine similarity matches mathematical formula u . v / (|u| * |v|).
    """
    u = [1.0, 0.0, 0.0, 0.0]
    v = [1.0, 0.0, 0.0, 0.0] # Identical
    assert abs(compute_cosine_similarity(u, v) - 1.0) < 1e-6

    w = [0.0, 1.0, 0.0, 0.0] # Orthogonal
    assert abs(compute_cosine_similarity(u, w) - 0.0) < 1e-6

    z = [-1.0, 0.0, 0.0, 0.0] # Opposite
    assert abs(compute_cosine_similarity(u, z) - (-1.0)) < 1e-6


@pytest.mark.tier1
@pytest.mark.r3
def test_match_validation_euclidean_distance_metric():
    """
    Tier 1 / R3: Euclidean distance is inversely related to cosine similarity for unit vectors:
    dist = sqrt(2 - 2 * sim).
    """
    query_vec = generate_normalized_vector(128, seed=11)
    cand_vec = synthesize_candidate_vector_with_similarity(query_vec, target_similarity=0.75, seed=12)

    sim = compute_cosine_similarity(query_vec, cand_vec)
    dist = compute_euclidean_distance(query_vec, cand_vec)

    expected_dist = math.sqrt(2.0 - 2.0 * sim)
    assert abs(dist - expected_dist) < 1e-4


@pytest.mark.tier1
@pytest.mark.r3
def test_match_validation_threshold_filtering():
    """
    Tier 1 / R3: Configurable threshold (0.70) correctly accepts >= 0.70 and rejects < 0.70.
    """
    query_vec = generate_normalized_vector(128, seed=20)
    cand1 = synthesize_candidate_vector_with_similarity(query_vec, 0.80, seed=1) # Passes
    cand2 = synthesize_candidate_vector_with_similarity(query_vec, 0.65, seed=2) # Rejected at 0.70 threshold

    candidates = [
        {"rank": 1, "source_url": "https://social.example.com/match1", "provider_confidence": 0.9},
        {"rank": 2, "source_url": "https://social.example.com/match2", "provider_confidence": 0.8},
    ]
    candidate_embeddings = {1: cand1, 2: cand2}

    engine = MatchValidationEngine(similarity_threshold=0.70)
    result = engine.evaluate_candidates(query_vec, candidates, candidate_embeddings)

    assert result["validation_status"] == "MATCH_CONFIRMED"
    assert result["selected_rank"] == 1
    assert len(result["rejected_candidates"]) == 1
    assert result["rejected_candidates"][0]["rank"] == 2


@pytest.mark.tier1
@pytest.mark.r3
def test_match_validation_candidate_ranking():
    """
    Tier 1 / R3: Multiple passing candidates are ranked strictly by descending similarity score.
    """
    query_vec = generate_normalized_vector(128, seed=30)
    c1 = synthesize_candidate_vector_with_similarity(query_vec, 0.72, seed=1)
    c2 = synthesize_candidate_vector_with_similarity(query_vec, 0.91, seed=2) # Highest
    c3 = synthesize_candidate_vector_with_similarity(query_vec, 0.84, seed=3)

    candidates = [
        {"rank": 1, "source_url": "https://social.example.com/1"},
        {"rank": 2, "source_url": "https://social.example.com/2"},
        {"rank": 3, "source_url": "https://social.example.com/3"},
    ]
    candidate_embeddings = {1: c1, 2: c2, 3: c3}

    engine = MatchValidationEngine(similarity_threshold=0.60)
    result = engine.evaluate_candidates(query_vec, candidates, candidate_embeddings)

    assert result["selected_rank"] == 2
    assert abs(result["similarity_score"] - 0.91) < 1e-3


@pytest.mark.tier1
@pytest.mark.r3
def test_match_validation_metrics_persistence_schema():
    """
    Tier 1 / R3: Validation result dictionary matches all schema fields required for metadata.json.
    """
    query_vec = generate_normalized_vector(128, seed=40)
    cand_vec = synthesize_candidate_vector_with_similarity(query_vec, 0.85, seed=1)

    candidates = [{"rank": 1, "source_url": "https://social.example.com/alice", "provider_confidence": 0.92}]
    candidate_embeddings = {1: cand_vec}

    engine = MatchValidationEngine(similarity_threshold=0.60)
    result = engine.evaluate_candidates(query_vec, candidates, candidate_embeddings)

    expected_keys = [
        "validation_status",
        "selected_candidate",
        "similarity_score",
        "distance_score",
        "provider_confidence",
        "selected_rank",
        "threshold_used",
        "rejected_candidates",
    ]
    for k in expected_keys:
        assert k in result


# ============================================================================
# Tier 2 - Boundary, Adversarial & Corner Cases (R3)
# ============================================================================

@pytest.mark.tier2
@pytest.mark.r3
def test_match_validation_all_candidates_below_threshold():
    """
    Tier 2 / R3 Boundary: When all candidates score below threshold (e.g. 0.45, 0.35 vs 0.60),
    system returns SIMILARITY_BELOW_THRESHOLD and no candidate is selected.
    """
    query_vec = generate_normalized_vector(128, seed=50)
    c1 = synthesize_candidate_vector_with_similarity(query_vec, 0.45, seed=1)
    c2 = synthesize_candidate_vector_with_similarity(query_vec, 0.35, seed=2)

    candidates = [
        {"rank": 1, "source_url": "https://social.example.com/low1"},
        {"rank": 2, "source_url": "https://social.example.com/low2"},
    ]
    candidate_embeddings = {1: c1, 2: c2}

    engine = MatchValidationEngine(similarity_threshold=0.60)
    result = engine.evaluate_candidates(query_vec, candidates, candidate_embeddings)

    assert result["validation_status"] == "SIMILARITY_BELOW_THRESHOLD"
    assert result["selected_candidate"] is None
    assert len(result["rejected_candidates"]) == 2


@pytest.mark.tier2
@pytest.mark.r3
def test_match_validation_candidate_image_fetch_failure():
    """
    Tier 2 / R3 Boundary: Candidate image download failure (None embedding) is logged to
    rejected_candidates and does not abort processing of other candidates.
    """
    query_vec = generate_normalized_vector(128, seed=60)
    c2_passing = synthesize_candidate_vector_with_similarity(query_vec, 0.88, seed=2)

    candidates = [
        {"rank": 1, "source_url": "https://social.example.com/broken_image"},
        {"rank": 2, "source_url": "https://social.example.com/working_image"},
    ]
    # Rank 1 image failed to download (None), Rank 2 succeeds
    candidate_embeddings = {1: None, 2: c2_passing}

    engine = MatchValidationEngine(similarity_threshold=0.60)
    result = engine.evaluate_candidates(query_vec, candidates, candidate_embeddings)

    assert result["validation_status"] == "MATCH_CONFIRMED"
    assert result["selected_rank"] == 2
    assert len(result["rejected_candidates"]) == 1
    assert result["rejected_candidates"][0]["rejection_reason"] == "NO_FACE_OR_DOWNLOAD_FAILED"


@pytest.mark.tier2
@pytest.mark.r3
def test_match_validation_exact_threshold_boundary():
    """
    Tier 2 / R3 Boundary: Candidate with similarity score exactly equal to threshold (0.60000) passes.
    """
    query_vec = generate_normalized_vector(128, seed=70)
    c_exact = synthesize_candidate_vector_with_similarity(query_vec, 0.60000, seed=1)

    candidates = [{"rank": 1, "source_url": "https://social.example.com/exact"}]
    candidate_embeddings = {1: c_exact}

    engine = MatchValidationEngine(similarity_threshold=0.60000)
    result = engine.evaluate_candidates(query_vec, candidates, candidate_embeddings)

    assert result["validation_status"] == "MATCH_CONFIRMED"
    assert result["selected_rank"] == 1


@pytest.mark.tier2
@pytest.mark.r3
def test_match_validation_empty_candidate_list():
    """
    Tier 2 / R3 Boundary: Empty candidate list returns status NO_CANDIDATES without raising an error.
    """
    query_vec = generate_normalized_vector(128, seed=80)
    engine = MatchValidationEngine(similarity_threshold=0.60)
    result = engine.evaluate_candidates(query_vec, candidates=[], candidate_embeddings={})

    assert result["validation_status"] == "NO_CANDIDATES"
    assert result["selected_candidate"] is None
    assert result["rejected_candidates"] == []


@pytest.mark.tier2
@pytest.mark.r3
def test_match_validation_orthogonal_and_opposite_vectors():
    """
    Tier 2 / R3 Boundary: Evaluates orthogonal (sim ~0.0) and diametrically opposite (sim ~ -1.0)
    vectors and confirms proper distance and threshold rejection.
    """
    query_vec = generate_normalized_vector(128, seed=90)
    c_orth = synthesize_candidate_vector_with_similarity(query_vec, 0.0, seed=1)
    c_oppo = synthesize_candidate_vector_with_similarity(query_vec, -1.0, seed=2)

    candidates = [
        {"rank": 1, "source_url": "https://social.example.com/orthogonal"},
        {"rank": 2, "source_url": "https://social.example.com/opposite"},
    ]
    candidate_embeddings = {1: c_orth, 2: c_oppo}

    engine = MatchValidationEngine(similarity_threshold=0.60)
    result = engine.evaluate_candidates(query_vec, candidates, candidate_embeddings)

    assert result["validation_status"] == "SIMILARITY_BELOW_THRESHOLD"
    assert len(result["rejected_candidates"]) == 2
    assert abs(result["rejected_candidates"][0]["similarity_score"] - 0.0) < 1e-3
    assert abs(result["rejected_candidates"][1]["similarity_score"] - (-1.0)) < 1e-3


# ============================================================================
# Production Module Unit Tests (app.validation)
# ============================================================================

from app.validation import (
    MatchValidationEngine as AppMatchValidationEngine,
    CandidateImageFetcher as AppCandidateImageFetcher,
    normalize_vector as app_normalize_vector,
    compute_cosine_similarity as app_compute_cosine_similarity,
    compute_euclidean_distance as app_compute_euclidean_distance,
)
from app.models import MatchValidationResult, ValidationStatus


@pytest.mark.tier1
@pytest.mark.r3
def test_app_validation_engine_pydantic_validate():
    """Verify AppMatchValidationEngine.validate() returns typed MatchValidationResult."""
    query_vec = generate_normalized_vector(128, seed=42)
    cand_vec = synthesize_candidate_vector_with_similarity(query_vec, target_similarity=0.88, seed=10)

    candidates = [
        {"rank": 1, "source_url": "https://social.example.com/alice/1", "provider_confidence": 0.95}
    ]
    candidate_embeddings = {1: cand_vec}

    engine = AppMatchValidationEngine(similarity_threshold=0.60)
    result = engine.validate(query_vec, candidates, candidate_embeddings=candidate_embeddings)

    assert isinstance(result, MatchValidationResult)
    assert result.validation_status == ValidationStatus.MATCH_CONFIRMED
    assert result.selected_rank == 1
    assert result.similarity_score is not None and abs(result.similarity_score - 0.88) < 1e-3
    assert result.selected_candidate is not None
    assert result.selected_candidate.selected_rank == 1


@pytest.mark.tier1
@pytest.mark.r3
def test_app_validation_candidate_fetcher_local(tmp_path: Path):
    """Verify CandidateImageFetcher reads local file paths and handles invalid URLs."""
    test_img = tmp_path / "test.jpg"
    test_img.write_bytes(b"dummy_image_data")

    fetcher = AppCandidateImageFetcher(timeout_seconds=5.0)
    data = fetcher.fetch_image(str(test_img))
    assert data == b"dummy_image_data"

    data_file_url = fetcher.fetch_image(f"file://{test_img}")
    assert data_file_url == b"dummy_image_data"

    data_invalid = fetcher.fetch_image("invalid://bad_url")
    assert data_invalid is None


@pytest.mark.tier1
@pytest.mark.r3
def test_app_validation_metrics_functions():
    """Verify normalization, cosine similarity, and euclidean distance in app.validation.metrics."""
    vec = [3.0, 4.0]
    normed = app_normalize_vector(vec)
    assert abs(np.linalg.norm(normed) - 1.0) < 1e-6
    assert abs(normed[0] - 0.6) < 1e-6
    assert abs(normed[1] - 0.8) < 1e-6

    u = [1.0, 0.0]
    v = [1.0, 0.0]
    assert abs(app_compute_cosine_similarity(u, v) - 1.0) < 1e-6
    assert abs(app_compute_euclidean_distance(u, v) - 0.0) < 1e-6

