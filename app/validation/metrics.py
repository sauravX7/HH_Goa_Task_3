"""Vector normalization, cosine similarity, and Euclidean distance metrics."""

import math
from typing import List, Sequence, Union
import numpy as np


def normalize_vector(vector: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    """Normalize vector to unit L2 norm."""
    vec = np.array(vector, dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm == 0 or np.isnan(norm):
        return vec
    return vec / norm


def compute_cosine_similarity(
    u: Union[Sequence[float], np.ndarray],
    v: Union[Sequence[float], np.ndarray],
) -> float:
    """Calculate cosine similarity between two embedding vectors.

    Formula: (u . v) / (||u|| * ||v||)
    Returns float in range [-1.0, 1.0].
    """
    arr_u = np.array(u, dtype=np.float64)
    arr_v = np.array(v, dtype=np.float64)

    norm_u = np.linalg.norm(arr_u)
    norm_v = np.linalg.norm(arr_v)

    if norm_u == 0 or norm_v == 0 or np.isnan(norm_u) or np.isnan(norm_v):
        return 0.0

    dot = np.dot(arr_u, arr_v)
    sim = float(dot / (norm_u * norm_v))
    return float(np.clip(sim, -1.0, 1.0))


def compute_euclidean_distance(
    u: Union[Sequence[float], np.ndarray],
    v: Union[Sequence[float], np.ndarray],
) -> float:
    r"""Calculate Euclidean distance between two embedding vectors.

    Formula: ||u - v||_2
    For normalized vectors, distance is related to cosine similarity by:
    d = sqrt(2 - 2 * cos_sim)
    """
    arr_u = np.array(u, dtype=np.float64)
    arr_v = np.array(v, dtype=np.float64)
    return float(np.linalg.norm(arr_u - arr_v))


def synthesize_candidate_vector_with_similarity(
    query_vector: Sequence[float],
    target_similarity: float,
    seed: int = 42,
) -> List[float]:
    """Synthesizes a unit vector v such that cosine_similarity(query_vector, v) == target_similarity.
    Formula: v = s * u + sqrt(1 - s^2) * u_perp
    """
    u = np.array(query_vector, dtype=np.float64)
    norm = np.linalg.norm(u)
    if norm > 0:
        u = u / norm
    dim = len(u)

    rng = np.random.RandomState(seed)
    random_vec = rng.randn(dim)
    # Gram-Schmidt orthogonalization
    proj = np.dot(random_vec, u) * u
    u_perp = random_vec - proj
    u_perp_norm = np.linalg.norm(u_perp)
    if u_perp_norm > 0:
        u_perp = u_perp / u_perp_norm

    s = max(-1.0, min(1.0, float(target_similarity)))
    v = s * u + math.sqrt(max(0.0, 1.0 - s * s)) * u_perp
    v_norm = np.linalg.norm(v)
    if v_norm > 0:
        v = v / v_norm
    return [float(x) for x in v]


# Direct function aliases
cosine_similarity = compute_cosine_similarity
euclidean_distance = compute_euclidean_distance
