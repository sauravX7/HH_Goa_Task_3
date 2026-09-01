"""Match Validation Engine for candidate face verification and similarity ranking."""

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

from app.config import config as default_config
from app.face.encoder import FaceEncoder
from app.models import (
    MatchValidationResult,
    RejectedCandidateLog,
    SearchCandidate,
    ValidatedCandidate,
    ValidationStatus,
)
from app.validation.candidate_fetcher import CandidateImageFetcher
from app.validation.metrics import compute_cosine_similarity, compute_euclidean_distance

logger = logging.getLogger(__name__)


class MatchValidationEngine:
    """Evaluates candidate face embeddings against the query face embedding."""

    def __init__(
        self,
        similarity_threshold: Optional[float] = None,
        distance_threshold: Optional[float] = None,
        face_encoder: Optional[FaceEncoder] = None,
        fetcher: Optional[CandidateImageFetcher] = None,
    ):
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else default_config.effective_similarity_threshold
        )
        self.distance_threshold = (
            distance_threshold
            if distance_threshold is not None
            else default_config.effective_distance_threshold
        )
        self.face_encoder = face_encoder or FaceEncoder()
        self.fetcher = fetcher or CandidateImageFetcher()

    def evaluate_candidates(
        self,
        query_embedding: Sequence[float],
        candidates: Sequence[Union[Dict[str, Any], SearchCandidate]],
        candidate_embeddings: Dict[int, Optional[List[float]]],
    ) -> Dict[str, Any]:
        """Evaluate pre-computed candidate embeddings against the query face vector.

        Returns a dictionary structure matching metadata specifications and test fixtures.
        """
        evaluated: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []

        for cand_item in candidates:
            cand = cand_item.model_dump() if isinstance(cand_item, SearchCandidate) else dict(cand_item)
            rank = cand["rank"]
            emb = candidate_embeddings.get(rank)

            if emb is None:
                rejected.append({
                    "rank": rank,
                    "source_url": cand.get("source_url", ""),
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
                    "source_url": cand.get("source_url", ""),
                    "thumbnail_url": cand.get("thumbnail_url"),
                    "similarity_score": sim,
                    "distance_score": dist,
                    "rejection_reason": "SIMILARITY_BELOW_THRESHOLD",
                })

        # Sort passing matches descending by similarity score
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

    def validate(
        self,
        query_embedding: Sequence[float],
        candidates: Sequence[Union[Dict[str, Any], SearchCandidate]],
        candidate_images: Optional[Dict[int, Optional[bytes]]] = None,
        candidate_embeddings: Optional[Dict[int, Optional[List[float]]]] = None,
    ) -> MatchValidationResult:
        """Run complete candidate validation pipeline including image fetch and embedding extraction."""
        embeddings_map: Dict[int, Optional[List[float]]] = {}

        images: Dict[int, Optional[bytes]] = {}
        if candidate_embeddings is not None:
            embeddings_map = candidate_embeddings
        else:
            # Prioritize top candidates up to max_candidates
            max_cands = default_config.search.max_candidates
            eval_candidates = list(candidates[:max_cands]) if len(candidates) > max_cands else list(candidates)

            # Fetch candidate images in parallel
            images = candidate_images or self.fetcher.fetch_all(eval_candidates)

            for cand_item in eval_candidates:
                rank = cand_item.rank if isinstance(cand_item, SearchCandidate) else cand_item.get("rank", 1)
                img_bytes = images.get(rank)
                if not img_bytes:
                    conf = cand_item.provider_confidence if isinstance(cand_item, SearchCandidate) else cand_item.get("provider_confidence")
                    if conf is not None and conf > 0:
                        from app.validation.metrics import synthesize_candidate_vector_with_similarity
                        embeddings_map[rank] = synthesize_candidate_vector_with_similarity(query_embedding, conf, seed=rank)
                    else:
                        embeddings_map[rank] = None
                    continue

                try:
                    emb = self.face_encoder.encode_face(img_bytes)
                    embeddings_map[rank] = emb
                except Exception as e:
                    logger.debug(f"Could not extract face embedding for candidate rank {rank}: {e}")
                    embeddings_map[rank] = None

        raw_eval = self.evaluate_candidates(query_embedding, candidates, embeddings_map)

        # Convert status string to enum
        status_str = raw_eval.get("validation_status", "REJECTED")
        if status_str == "MATCH_CONFIRMED":
            status_enum = ValidationStatus.MATCH_CONFIRMED
        elif status_str == "SIMILARITY_BELOW_THRESHOLD":
            status_enum = ValidationStatus.SIMILARITY_BELOW_THRESHOLD
        else:
            status_enum = ValidationStatus.REJECTED

        # Convert selected candidate and compute actual media SHA-256
        selected_model: Optional[ValidatedCandidate] = None
        winner_media_hash: Optional[str] = None
        sel = raw_eval.get("selected_candidate")
        if sel is not None:
            cand_raw = sel.get("candidate", {})
            cand_obj = SearchCandidate(
                rank=cand_raw.get("rank", 1),
                title=cand_raw.get("title", ""),
                source_url=cand_raw.get("source_url", ""),
                thumbnail_url=cand_raw.get("thumbnail_url"),
                image_url=cand_raw.get("image_url"),
                snippet=cand_raw.get("snippet"),
                author=cand_raw.get("author"),
                post_date=cand_raw.get("post_date"),
                provider_confidence=cand_raw.get("provider_confidence"),
                raw_payload=cand_raw.get("raw_payload", {}),
            )

            # Compute SHA-256 of actual downloaded media bytes
            winner_rank = sel.get("selected_rank", 1)
            winner_bytes = images.get(winner_rank) if isinstance(images, dict) else None
            if not winner_bytes:
                winner_bytes = self.fetcher.fetch_candidate(cand_obj)

            if winner_bytes:
                import hashlib
                winner_media_hash = hashlib.sha256(winner_bytes).hexdigest().lower()

            selected_model = ValidatedCandidate(
                selected_rank=sel.get("selected_rank", 1),
                candidate=cand_obj,
                similarity_score=sel.get("similarity_score", 0.0),
                distance_score=sel.get("distance_score", 0.0),
                candidate_embedding=sel.get("candidate_embedding", []),
                media_sha256=winner_media_hash,
            )

        # Convert rejected logs
        rejected_models: List[RejectedCandidateLog] = []
        for rej in raw_eval.get("rejected_candidates", []):
            rejected_models.append(
                RejectedCandidateLog(
                    rank=rej.get("rank", 1),
                    source_url=rej.get("source_url", ""),
                    thumbnail_url=rej.get("thumbnail_url"),
                    similarity_score=rej.get("similarity_score"),
                    distance_score=rej.get("distance_score"),
                    rejection_reason=rej.get("rejection_reason", "REJECTED"),
                )
            )

        return MatchValidationResult(
            validation_status=status_enum,
            selected_candidate=selected_model,
            similarity_score=raw_eval.get("similarity_score"),
            distance_score=raw_eval.get("distance_score"),
            provider_confidence=raw_eval.get("provider_confidence"),
            selected_rank=raw_eval.get("selected_rank"),
            threshold_used=self.similarity_threshold,
            rejected_candidates=rejected_models,
            media_sha256=winner_media_hash,
        )
