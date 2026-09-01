"""Match validation, candidate visual retrieval, and similarity scoring package."""

from app.validation.candidate_fetcher import CandidateImageFetcher
from app.validation.engine import MatchValidationEngine
from app.validation.metrics import (
    compute_cosine_similarity,
    compute_euclidean_distance,
    cosine_similarity,
    euclidean_distance,
    normalize_vector,
)

__all__ = [
    "MatchValidationEngine",
    "CandidateImageFetcher",
    "normalize_vector",
    "compute_cosine_similarity",
    "compute_euclidean_distance",
    "cosine_similarity",
    "euclidean_distance",
]
