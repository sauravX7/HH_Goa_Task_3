"""Deterministic Mock Search Provider for testing and offline development."""

import hashlib
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from app.provenance.base import SearchProvider


class MockSearchProvider(SearchProvider):
    """Deterministic offline search provider for testing fallback priority and candidate evaluation."""

    def __init__(
        self,
        name: str = "mock_search_provider",
        is_available: bool = True,
        should_fail: bool = False,
        rate_limit: bool = False,
        return_empty: bool = False,
        mock_candidates: Optional[List[Dict[str, Any]]] = None,
        latency: float = 0.01,
        timeout_seconds: int = 15,
        **kwargs: Any,
    ):
        super().__init__(name=name, timeout_seconds=timeout_seconds)
        self._available = is_available
        self.should_fail = should_fail
        self.rate_limit = rate_limit
        self.return_empty = return_empty
        self.mock_candidates = mock_candidates
        self.latency = latency

    def is_available(self) -> bool:
        """Return provider availability status."""
        return self._available

    def search(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """Return simulated candidate search results or simulate operational errors."""
        if self.latency > 0:
            time.sleep(self.latency)

        if self.rate_limit:
            raise RuntimeError("HTTP 429: Too Many Requests / Rate Limit Exceeded")

        if self.should_fail:
            raise ConnectionError(f"Mock endpoint unreachable for provider {self.name}")

        if self.return_empty:
            return {"candidates": [], "total_results": 0}

        if self.mock_candidates is not None:
            return {
                "candidates": self.mock_candidates,
                "total_results": len(self.mock_candidates),
            }

        # Deterministic default candidates
        return {
            "candidates": [
                {
                    "rank": 1,
                    "title": "Alice Web3 - Visual Lookup Match",
                    "source_url": "https://social.example.com/alice/post/100",
                    "thumbnail_url": "https://social.example.com/thumbs/alice.jpg",
                    "image_url": "https://social.example.com/images/alice.jpg",
                    "provider_confidence": 0.95,
                    "author": "Alice Web3",
                    "post_date": "2026-09-01T10:00:00Z",
                    "snippet": "Verified web3 profile picture and cryptographic badge",
                },
                {
                    "rank": 2,
                    "title": "Lookalike Profile - Public Web",
                    "source_url": "https://social.example.com/lookalike/post/200",
                    "thumbnail_url": "https://social.example.com/thumbs/lookalike.jpg",
                    "image_url": "https://social.example.com/images/lookalike.jpg",
                    "provider_confidence": 0.62,
                    "author": "Lookalike User",
                    "post_date": "2026-08-15T09:00:00Z",
                    "snippet": "Public avatar photo matching search crop",
                },
            ],
            "total_results": 2,
        }
