"""Bing Visual Search API reverse visual lookup provider."""

import html
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import requests

from app.config import config
from app.provenance.base import SearchProvider


class BingVisualProvider(SearchProvider):
    """Bing Visual Search API provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: str = "https://api.bing.microsoft.com/v7.0/images/visualsearch",
        timeout_seconds: int = 15,
        name: str = "bing_visual_search",
    ):
        super().__init__(name=name, timeout_seconds=timeout_seconds)
        self.api_key = (
            api_key
            if api_key is not None
            else (config.bing_visual_search_api_key or os.getenv("BING_VISUAL_SEARCH_API_KEY"))
        )
        self.endpoint = endpoint

    def is_available(self) -> bool:
        """Return True if Bing Visual Search API key is configured."""
        return bool(self.api_key and len(self.api_key.strip()) > 0)

    def search(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """Execute live Bing Visual Search against Microsoft Cognitive Services endpoint."""
        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"Search image not found at {p}")

        if not self.is_available():
            raise RuntimeError("Bing Visual Search API key is not configured / unavailable")

        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
        }

        try:
            with open(p, "rb") as f:
                files = {"image": (p.name, f, "image/jpeg")}
                response = requests.post(
                    self.endpoint,
                    headers=headers,
                    files=files,
                    timeout=self.timeout_seconds,
                )

            if response.status_code == 429:
                raise RuntimeError("HTTP 429: Too Many Requests / Rate Limit Exceeded")

            if response.status_code != 200:
                raise RuntimeError(
                    f"Bing Visual Search failed with status {response.status_code}: {response.text}"
                )

            data = response.json()
            if not isinstance(data, dict):
                raise ValueError(f"Malformed Bing response: expected dict, got {type(data)}")

        except requests.exceptions.Timeout as e:
            raise TimeoutError(f"Bing Visual Search timed out after {self.timeout_seconds}s: {e}") from e
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(f"Bing Visual Search endpoint unreachable: {e}") from e

        # Extract items from Bing tags/actions structure
        candidates: List[Dict[str, Any]] = []
        tags = data.get("tags", [])
        if not isinstance(tags, list):
            tags = []

        rank_counter = 1
        for tag in tags:
            if not isinstance(tag, dict):
                continue
            actions = tag.get("actions", [])
            if not isinstance(actions, list):
                continue

            for action in actions:
                if not isinstance(action, dict):
                    continue
                action_data = action.get("data", {})
                if not isinstance(action_data, dict):
                    continue
                values = action_data.get("value", [])
                if not isinstance(values, list):
                    continue

                for item in values:
                    if not isinstance(item, dict):
                        continue

                    raw_title = item.get("name") or "Bing Visual Match"
                    title = html.unescape(str(raw_title)).strip()

                    source_url = item.get("hostPageUrl") or item.get("webSearchUrl") or ""
                    thumbnail_url = item.get("thumbnailUrl")
                    image_url = item.get("contentUrl") or thumbnail_url
                    post_date = item.get("datePublished")
                    snippet = item.get("snippet")

                    # Extract author from provider list if present
                    author = None
                    providers = item.get("provider")
                    if isinstance(providers, list) and len(providers) > 0 and isinstance(providers[0], dict):
                        author = providers[0].get("name")
                    elif isinstance(providers, str):
                        author = providers

                    confidence = max(0.20, 0.92 - (rank_counter - 1) * 0.05)

                    candidate = {
                        "rank": rank_counter,
                        "title": title,
                        "source_url": source_url,
                        "thumbnail_url": thumbnail_url,
                        "image_url": image_url,
                        "snippet": snippet,
                        "author": author,
                        "post_date": post_date,
                        "provider_confidence": round(float(confidence), 4),
                        "raw_payload": item,
                    }
                    candidates.append(candidate)
                    rank_counter += 1

        return {
            "candidates": candidates,
            "total_results": len(candidates),
            "raw_response": data,
        }
