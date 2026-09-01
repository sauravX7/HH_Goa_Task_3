"""Abstract Base Search Provider interface and shared provenance utilities."""

import abc
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from app.models import SearchCandidate, SearchProvenanceResult, ProviderAttemptLog


class SearchProvider(abc.ABC):
    """Abstract base class for all visual reverse search providers."""

    def __init__(self, name: str, timeout_seconds: int = 15):
        self.name = name
        self.timeout_seconds = timeout_seconds

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check if the search provider is properly configured and reachable."""
        raise NotImplementedError

    @abc.abstractmethod
    def search(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """Execute visual search with the given image crop.

        Returns a dictionary with 'candidates' (List[Dict[str, Any]] or List[SearchCandidate])
        and 'total_results' (int).
        """
        raise NotImplementedError

    @staticmethod
    def compute_image_sha256(image_path: Union[str, Path]) -> str:
        """Calculate the SHA-256 cryptographic digest of the image file."""
        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"Image path not found: {p}")
        return hashlib.sha256(p.read_bytes()).hexdigest().lower()


# Backward compatibility alias
BaseSearchProvider = SearchProvider
