"""Prioritized multi-provider visual search orchestrator with provenance tracking."""

import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from app.config import AppSettings, config as default_config
from app.models import (
    ProviderAttemptLog,
    SearchCandidate,
    SearchProvenanceResult,
)
from app.provenance.base import SearchProvider
from app.provenance.bing_visual import BingVisualProvider
from app.provenance.mock_provider import MockSearchProvider
from app.provenance.playwright_lens import PlaywrightLensProvider
from app.provenance.serpapi_lens import SerpApiLensProvider


class SearchProvenanceEngine:
    """Orchestrates prioritized visual reverse search with fallback chain and audit logging."""

    def __init__(
        self,
        providers: Optional[List[SearchProvider]] = None,
        config: Optional[AppSettings] = None,
        max_retries: int = 2,
        backoff_factor: float = 0.5,
    ):
        self.config = config or default_config
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

        if providers is not None:
            self.providers = providers
        else:
            self.providers = self._build_default_providers()

    def _build_default_providers(self) -> List[SearchProvider]:
        """Instantiate genuine search providers in priority order according to configuration."""
        provider_map = {
            "serpapi_lens": SerpApiLensProvider,
            "serpapi_google_lens": SerpApiLensProvider,
            "bing_visual": BingVisualProvider,
            "bing_visual_search": BingVisualProvider,
            "playwright_lens": PlaywrightLensProvider,
            "playwright_google_lens": PlaywrightLensProvider,
        }

        instances: List[SearchProvider] = []
        for name in self.config.search.provider_priority:
            norm_name = name.lower().strip()
            # Ignore test mock providers in production runtime
            if norm_name in ("mock", "mock_search_provider"):
                continue
            cls = provider_map.get(norm_name)
            if cls:
                try:
                    instances.append(cls(timeout_seconds=self.config.search.timeout_seconds))
                except TypeError:
                    instances.append(cls())
                except Exception as e:
                    logger.debug(f"Could not instantiate provider {cls}: {e}")

        if not instances:
            # Default genuine provider fallback chain: SerpAPI -> Bing -> Playwright
            instances = [
                SerpApiLensProvider(timeout_seconds=self.config.search.timeout_seconds),
                BingVisualProvider(timeout_seconds=self.config.search.timeout_seconds),
                PlaywrightLensProvider(timeout_seconds=self.config.search.timeout_seconds),
            ]
        return instances

    def execute_search(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """Execute visual search through the priority fallback chain.

        Returns a dictionary compatible with metadata serialization and legacy test fixtures.
        """
        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"Input face crop not found at {p}")

        image_bytes = p.read_bytes()
        query_image_hash = hashlib.sha256(image_bytes).hexdigest().lower()
        now_utc = datetime.now(timezone.utc)
        query_timestamp_iso = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        query_id = f"query_{int(now_utc.timestamp())}_{query_image_hash[:8]}"

        fallback_history: List[Dict[str, Any]] = []
        selected_result: Optional[Dict[str, Any]] = None
        selected_provider_name: Optional[str] = None

        for provider in self.providers:
            if not provider.is_available():
                fallback_history.append({
                    "provider_name": provider.name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "success": False,
                    "latency_seconds": 0.0,
                    "error_message": "Provider not available / missing API key",
                    "candidates_found": 0,
                })
                continue

            t0 = time.time()
            try:
                res = provider.search(p)
                latency = time.time() - t0

                if not isinstance(res, dict):
                    raise ValueError(f"Provider returned invalid non-dict response: {type(res)}")

                candidates = res.get("candidates", [])
                if not isinstance(candidates, list):
                    candidates = []

                candidates_found = len(candidates)
                fallback_history.append({
                    "provider_name": provider.name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "success": True,
                    "latency_seconds": round(latency, 4),
                    "error_message": None,
                    "candidates_found": candidates_found,
                })

                if candidates_found > 0:
                    selected_result = res
                    selected_provider_name = provider.name
                    break

            except Exception as e:
                latency = time.time() - t0
                fallback_history.append({
                    "provider_name": provider.name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "success": False,
                    "latency_seconds": round(latency, 4),
                    "error_message": str(e),
                    "candidates_found": 0,
                })

        final_candidates = selected_result.get("candidates", []) if selected_result else []

        return {
            "provider_used": selected_provider_name or "none",
            "query_image_hash": query_image_hash,
            "query_timestamp": query_timestamp_iso,
            "query_id": query_id,
            "candidates": final_candidates,
            "fallback_history": fallback_history,
            "total_results_found": len(final_candidates),
        }

    def search(self, image_path: Union[str, Path]) -> SearchProvenanceResult:
        """Execute search and return typed Pydantic SearchProvenanceResult."""
        raw_dict = self.execute_search(image_path)

        candidates_models: List[SearchCandidate] = []
        for c in raw_dict.get("candidates", []):
            if isinstance(c, SearchCandidate):
                candidates_models.append(c)
            elif isinstance(c, dict):
                candidates_models.append(
                    SearchCandidate(
                        rank=c.get("rank", 1),
                        title=c.get("title", ""),
                        source_url=c.get("source_url", ""),
                        thumbnail_url=c.get("thumbnail_url"),
                        image_url=c.get("image_url"),
                        snippet=c.get("snippet"),
                        author=c.get("author"),
                        post_date=c.get("post_date"),
                        provider_confidence=c.get("provider_confidence"),
                        raw_payload=c.get("raw_payload", {}),
                    )
                )

        attempt_logs: List[ProviderAttemptLog] = []
        for a in raw_dict.get("fallback_history", []):
            attempt_logs.append(
                ProviderAttemptLog(
                    provider_name=a.get("provider_name", "unknown"),
                    success=a.get("success", False),
                    latency_seconds=a.get("latency_seconds", 0.0),
                    error_message=a.get("error_message"),
                    candidates_found=a.get("candidates_found", 0),
                )
            )

        return SearchProvenanceResult(
            provider_used=raw_dict.get("provider_used", "none"),
            query_image_hash=raw_dict.get("query_image_hash", ""),
            query_timestamp=datetime.now(timezone.utc),
            query_id=raw_dict.get("query_id", ""),
            candidates=candidates_models,
            fallback_history=attempt_logs,
            total_results_found=raw_dict.get("total_results_found", len(candidates_models)),
        )


# Backward compatibility alias
FallbackSearchEngine = SearchProvenanceEngine
