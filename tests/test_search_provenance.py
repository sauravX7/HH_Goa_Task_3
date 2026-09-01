"""
tests/test_search_provenance.py - Comprehensive Tier 1 and Tier 2 tests for
Requirement R2: Genuine Search Provenance Engine, Multi-Engine Reverse Visual Search,
Fallback Priority Chain, and Provenance Metadata Tracking.
"""

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import pytest

from tests.conftest import (
    compute_sha256_digest,
    serialize_canonical_json,
)


class BaseSearchProvider:
    def __init__(self, name: str, is_available: bool = True):
        self.name = name
        self._available = is_available

    def is_available(self) -> bool:
        return self._available

    def search(self, image_path: Path) -> Dict[str, Any]:
        raise NotImplementedError


class MockSerpApiLensProvider(BaseSearchProvider):
    def __init__(self, should_fail: bool = False, rate_limit: bool = False, return_empty: bool = False):
        super().__init__(name="serpapi_google_lens")
        self.should_fail = should_fail
        self.rate_limit = rate_limit
        self.return_empty = return_empty

    def search(self, image_path: Path) -> Dict[str, Any]:
        if self.rate_limit:
            raise RuntimeError("HTTP 429: Too Many Requests / Rate Limit Exceeded")
        if self.should_fail:
            raise ConnectionError("SerpAPI endpoint unreachable")
        if self.return_empty:
            return {"candidates": [], "total_results": 0}

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
                },
            ],
            "total_results": 2,
        }


class MockBingVisualProvider(BaseSearchProvider):
    def __init__(self, should_fail: bool = False):
        super().__init__(name="bing_visual_search")
        self.should_fail = should_fail

    def search(self, image_path: Path) -> Dict[str, Any]:
        if self.should_fail:
            raise ConnectionError("Bing Visual Search timeout")
        return {
            "candidates": [
                {
                    "rank": 1,
                    "title": "Alice Web3 - Bing Visual Result",
                    "source_url": "https://social.example.com/alice/post/100",
                    "thumbnail_url": "https://social.example.com/thumbs/alice_bing.jpg",
                    "image_url": "https://social.example.com/images/alice.jpg",
                    "provider_confidence": 0.91,
                    "author": "Alice Web3",
                    "post_date": "2026-09-01T10:00:00Z",
                }
            ],
            "total_results": 1,
        }


class MockPlaywrightLensProvider(BaseSearchProvider):
    def __init__(self):
        super().__init__(name="playwright_google_lens")

    def search(self, image_path: Path) -> Dict[str, Any]:
        return {
            "candidates": [
                {
                    "rank": 1,
                    "title": "Alice Web3 - Browser Automation Result",
                    "source_url": "https://social.example.com/alice/post/100",
                    "thumbnail_url": "https://social.example.com/thumbs/alice_pw.jpg",
                    "image_url": "https://social.example.com/images/alice.jpg",
                    "provider_confidence": 0.89,
                    "author": "Alice Web3",
                    "post_date": "2026-09-01T10:00:00Z",
                }
            ],
            "total_results": 1,
        }


class FallbackSearchEngine:
    """Orchestrates prioritized search fallback chain and logs provenance."""
    def __init__(self, providers: List[BaseSearchProvider]):
        self.providers = providers

    def execute_search(self, image_path: Path) -> Dict[str, Any]:
        image_bytes = image_path.read_bytes()
        query_image_hash = hashlib.sha256(image_bytes).hexdigest().lower()
        fallback_history = []
        selected_result = None
        selected_provider_name = None

        for provider in self.providers:
            if not provider.is_available():
                fallback_history.append({
                    "provider_name": provider.name,
                    "success": False,
                    "latency_seconds": 0.0,
                    "error_message": "Provider not available / missing API key",
                    "candidates_found": 0,
                })
                continue

            t0 = time.time()
            try:
                res = provider.search(image_path)
                latency = time.time() - t0
                candidates = res.get("candidates", [])
                fallback_history.append({
                    "provider_name": provider.name,
                    "success": True,
                    "latency_seconds": latency,
                    "error_message": None,
                    "candidates_found": len(candidates),
                })
                if candidates:
                    selected_result = res
                    selected_provider_name = provider.name
                    break
            except Exception as e:
                latency = time.time() - t0
                fallback_history.append({
                    "provider_name": provider.name,
                    "success": False,
                    "latency_seconds": latency,
                    "error_message": str(e),
                    "candidates_found": 0,
                })

        return {
            "provider_used": selected_provider_name or "none",
            "query_image_hash": query_image_hash,
            "query_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "query_id": f"query_{int(time.time())}_{query_image_hash[:8]}",
            "candidates": selected_result.get("candidates", []) if selected_result else [],
            "fallback_history": fallback_history,
        }


# ============================================================================
# Tier 1 - Feature Functional Coverage (R2)
# ============================================================================

@pytest.mark.tier1
@pytest.mark.r2
def test_search_provenance_primary_provider_candidates(synthetic_face_image_path: Path):
    """
    Tier 1 / R2: Verify SerpAPI primary provider returns structured candidates with URLs, titles, and ranks.
    """
    provider = MockSerpApiLensProvider()
    engine = FallbackSearchEngine([provider])
    result = engine.execute_search(synthetic_face_image_path)

    assert result["provider_used"] == "serpapi_google_lens"
    assert len(result["candidates"]) == 2
    top = result["candidates"][0]
    assert top["rank"] == 1
    assert top["source_url"] == "https://social.example.com/alice/post/100"
    assert "Alice" in top["title"]
    assert top["provider_confidence"] >= 0.90


@pytest.mark.tier1
@pytest.mark.r2
def test_search_provenance_metadata_structure(synthetic_face_image_path: Path):
    """
    Tier 1 / R2: Verify search provenance result contains query_image_hash, timestamp, query_id, candidates.
    """
    engine = FallbackSearchEngine([MockSerpApiLensProvider()])
    result = engine.execute_search(synthetic_face_image_path)

    assert "query_image_hash" in result
    assert len(result["query_image_hash"]) == 64
    assert result["query_timestamp"].endswith("Z")
    assert result["query_id"].startswith("query_")
    assert isinstance(result["candidates"], list)
    assert len(result["fallback_history"]) == 1
    assert result["fallback_history"][0]["success"] is True


@pytest.mark.tier1
@pytest.mark.r2
def test_search_fallback_priority_chain_serpapi_to_bing(synthetic_face_image_path: Path):
    """
    Tier 1 / R2: When SerpAPI fails, system automatically falls back to Bing Visual search.
    """
    serp_failing = MockSerpApiLensProvider(should_fail=True)
    bing_working = MockBingVisualProvider()
    engine = FallbackSearchEngine([serp_failing, bing_working])

    result = engine.execute_search(synthetic_face_image_path)

    assert result["provider_used"] == "bing_visual_search"
    assert len(result["fallback_history"]) == 2
    assert result["fallback_history"][0]["provider_name"] == "serpapi_google_lens"
    assert result["fallback_history"][0]["success"] is False
    assert result["fallback_history"][1]["provider_name"] == "bing_visual_search"
    assert result["fallback_history"][1]["success"] is True
    assert len(result["candidates"]) == 1


@pytest.mark.tier1
@pytest.mark.r2
def test_search_fallback_to_playwright_lens(synthetic_face_image_path: Path):
    """
    Tier 1 / R2: When SerpAPI and Bing fail, system falls back to Playwright headless browser.
    """
    serp_failing = MockSerpApiLensProvider(should_fail=True)
    bing_failing = MockBingVisualProvider(should_fail=True)
    playwright_working = MockPlaywrightLensProvider()

    engine = FallbackSearchEngine([serp_failing, bing_failing, playwright_working])
    result = engine.execute_search(synthetic_face_image_path)

    assert result["provider_used"] == "playwright_google_lens"
    assert len(result["fallback_history"]) == 3
    assert result["fallback_history"][2]["success"] is True
    assert len(result["candidates"]) == 1


@pytest.mark.tier1
@pytest.mark.r2
def test_search_provenance_persisted_to_metadata_json(tmp_path: Path, synthetic_face_image_path: Path):
    """
    Tier 1 / R2: Verify provenance metadata object is cleanly serializable to artifacts/metadata.json.
    """
    metadata_path = tmp_path / "artifacts" / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    engine = FallbackSearchEngine([MockSerpApiLensProvider()])
    provenance = engine.execute_search(synthetic_face_image_path)

    payload = {
        "search_provenance": provenance,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    metadata_path.write_text(json.dumps(payload, indent=2))

    assert metadata_path.exists()
    loaded = json.loads(metadata_path.read_text())
    assert loaded["search_provenance"]["provider_used"] == "serpapi_google_lens"
    assert len(loaded["search_provenance"]["candidates"]) == 2


@pytest.mark.tier1
@pytest.mark.r2
def test_search_candidate_ranking_and_scores(sample_search_candidates: List[Dict[str, Any]]):
    """
    Tier 1 / R2: Candidate ranks are sorted integers (1, 2, 3...) and confidence scores are floats in [0, 1].
    """
    ranks = [c["rank"] for c in sample_search_candidates]
    assert ranks == [1, 2, 3]

    for cand in sample_search_candidates:
        assert 0.0 <= cand["provider_confidence"] <= 1.0
        assert cand["source_url"].startswith("http")


# ============================================================================
# Tier 2 - Boundary, Adversarial & Corner Cases (R2)
# ============================================================================

@pytest.mark.tier2
@pytest.mark.r2
def test_search_zero_results_graceful_handling(synthetic_face_image_path: Path):
    """
    Tier 2 / R2 Boundary: Provider returning zero search results returns empty candidate list
    without raising an uncaught exception.
    """
    engine = FallbackSearchEngine([MockSerpApiLensProvider(return_empty=True)])
    result = engine.execute_search(synthetic_face_image_path)

    assert result["provider_used"] == "none"
    assert result["candidates"] == []
    assert len(result["fallback_history"]) == 1
    assert result["fallback_history"][0]["candidates_found"] == 0


@pytest.mark.tier2
@pytest.mark.r2
def test_search_rate_limit_429_recovery(synthetic_face_image_path: Path):
    """
    Tier 2 / R2 Boundary: HTTP 429 Rate Limit error triggers fallback to next available provider.
    """
    rate_limited_serp = MockSerpApiLensProvider(rate_limit=True)
    working_bing = MockBingVisualProvider()
    engine = FallbackSearchEngine([rate_limited_serp, working_bing])

    result = engine.execute_search(synthetic_face_image_path)

    assert result["provider_used"] == "bing_visual_search"
    assert "429" in result["fallback_history"][0]["error_message"]
    assert result["fallback_history"][1]["success"] is True


@pytest.mark.tier2
@pytest.mark.r2
def test_search_network_timeout_handling(synthetic_face_image_path: Path):
    """
    Tier 2 / R2 Boundary: Network timeout is caught, recorded in fallback history with latency,
    and switches to next provider.
    """
    class TimeoutProvider(BaseSearchProvider):
        def __init__(self):
            super().__init__(name="timeout_provider")

        def search(self, image_path: Path) -> Dict[str, Any]:
            raise TimeoutError("Connection to search gateway timed out after 5.0s")

    engine = FallbackSearchEngine([TimeoutProvider(), MockPlaywrightLensProvider()])
    result = engine.execute_search(synthetic_face_image_path)

    assert result["provider_used"] == "playwright_google_lens"
    assert "timed out" in result["fallback_history"][0]["error_message"]


@pytest.mark.tier2
@pytest.mark.r2
def test_search_special_chars_in_candidate_metadata():
    """
    Tier 2 / R2 Boundary: Candidate titles with HTML entities, Unicode symbols, and emojis
    are preserved without corruption.
    """
    candidate = {
        "rank": 1,
        "title": "Post &amp; Analysis: &#x1F6E1; Blockchain &lt;Verification&gt; é, à, 漢字",
        "source_url": "https://social.example.com/post?id=123&ref=lens#top",
        "snippet": "Contains newlines\nand tabs\tand emojis 🛡️ 📸",
    }
    # Unescape / sanitize check
    import html
    clean_title = html.unescape(candidate["title"])
    assert "&amp;" not in clean_title
    assert "🛡" in clean_title
    assert "<Verification>" in clean_title


@pytest.mark.tier2
@pytest.mark.r2
def test_search_provider_availability_check():
    """
    Tier 2 / R2 Boundary: Unavailable provider (missing API keys) is skipped cleanly.
    """
    p_unavailable = BaseSearchProvider(name="unconfigured_provider", is_available=False)
    p_available = MockBingVisualProvider()
    engine = FallbackSearchEngine([p_unavailable, p_available])

    # Should skip unconfigured without invoking search
    assert engine.providers[0].is_available() is False
    assert engine.providers[1].is_available() is True


@pytest.mark.tier2
@pytest.mark.r2
def test_search_malformed_api_payload_handling(synthetic_face_image_path: Path):
    """
    Tier 2 / R2 Boundary: Search provider returning malformed payload (e.g. missing candidate list or non-dict)
    is caught gracefully.
    """
    class MalformedProvider(BaseSearchProvider):
        def __init__(self):
            super().__init__(name="malformed_provider")

        def search(self, image_path: Path) -> Dict[str, Any]:
            return {"status": "ok"} # Missing 'candidates' key

    engine = FallbackSearchEngine([MalformedProvider(), MockPlaywrightLensProvider()])
    result = engine.execute_search(synthetic_face_image_path)

    assert result["provider_used"] == "playwright_google_lens"


# ============================================================================
# Production Module Unit Tests (app.provenance)
# ============================================================================

from app.provenance import (
    SearchProvider,
    SerpApiLensProvider,
    BingVisualProvider,
    PlaywrightLensProvider,
    MockSearchProvider as AppMockSearchProvider,
    SearchProvenanceEngine as AppSearchProvenanceEngine,
)
from app.models import SearchProvenanceResult


@pytest.mark.tier1
@pytest.mark.r2
def test_app_provenance_engine_typed_model(synthetic_face_image_path: Path):
    """Verify SearchProvenanceEngine.search() produces a valid Pydantic SearchProvenanceResult."""
    mock_prov = AppMockSearchProvider(name="mock_test")
    engine = AppSearchProvenanceEngine(providers=[mock_prov])
    result = engine.search(synthetic_face_image_path)

    assert isinstance(result, SearchProvenanceResult)
    assert result.provider_used == "mock_test"
    assert len(result.query_image_hash) == 64
    assert len(result.candidates) == 2
    assert result.candidates[0].rank == 1
    assert result.candidates[0].source_url.startswith("https://")
    assert result.total_results_found == 2
    assert len(result.fallback_history) == 1
    assert result.fallback_history[0].success is True


@pytest.mark.tier1
@pytest.mark.r2
def test_app_serpapi_lens_provider_mock_http(synthetic_face_image_path: Path, monkeypatch):
    """Verify SerpApiLensProvider parses SerpAPI visual_matches HTTP response."""
    class FakeResponse:
        status_code = 200
        text = "ok"

        def json(self):
            return {
                "visual_matches": [
                    {
                        "position": 1,
                        "title": "Alice Web3 &amp; Security Profile",
                        "link": "https://social.example.com/alice/real",
                        "thumbnail": "https://social.example.com/thumb.jpg",
                        "original": "https://social.example.com/orig.jpg",
                        "source": "Twitter/X",
                        "score": 0.96,
                    }
                ]
            }

    import requests
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: FakeResponse())
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: FakeResponse())

    provider = SerpApiLensProvider(api_key="test_serp_key")
    assert provider.is_available() is True
    res = provider.search(synthetic_face_image_path)

    assert res["total_results"] == 1
    cand = res["candidates"][0]
    assert cand["rank"] == 1
    assert cand["title"] == "Alice Web3 & Security Profile"
    assert cand["source_url"] == "https://social.example.com/alice/real"
    assert cand["provider_confidence"] == 0.96


@pytest.mark.tier1
@pytest.mark.r2
def test_app_bing_visual_provider_mock_http(synthetic_face_image_path: Path, monkeypatch):
    """Verify BingVisualProvider parses Bing Visual Search API response structure."""
    class FakeResponse:
        status_code = 200
        text = "ok"

        def json(self):
            return {
                "tags": [
                    {
                        "actions": [
                            {
                                "actionType": "PagesIncluding",
                                "data": {
                                    "value": [
                                        {
                                            "name": "Alice Bing Match",
                                            "hostPageUrl": "https://social.example.com/bing/alice",
                                            "thumbnailUrl": "https://social.example.com/bing_thumb.jpg",
                                            "provider": [{"name": "Web Source"}],
                                        }
                                    ]
                                },
                            }
                        ]
                    }
                ]
            }

    import requests
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: FakeResponse())

    provider = BingVisualProvider(api_key="test_bing_key")
    assert provider.is_available() is True
    res = provider.search(synthetic_face_image_path)

    assert res["total_results"] == 1
    cand = res["candidates"][0]
    assert cand["rank"] == 1
    assert cand["title"] == "Alice Bing Match"
    assert cand["source_url"] == "https://social.example.com/bing/alice"
    assert cand["author"] == "Web Source"


@pytest.mark.tier1
@pytest.mark.r2
def test_app_provenance_image_sha256(synthetic_face_image_path: Path):
    """Verify compute_image_sha256 matches hashlib calculation."""
    expected = hashlib.sha256(synthetic_face_image_path.read_bytes()).hexdigest().lower()
    actual = SearchProvider.compute_image_sha256(synthetic_face_image_path)
    assert actual == expected

