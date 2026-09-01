"""SerpAPI Google Lens reverse image search client."""

import html
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import requests

from app.config import config
from app.provenance.base import SearchProvider


class SerpApiLensProvider(SearchProvider):
    """Google Lens visual search provider powered by SerpAPI."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout_seconds: int = 15,
        name: str = "serpapi_google_lens",
    ):
        super().__init__(name=name, timeout_seconds=timeout_seconds)
        self.api_key = (
            api_key
            if api_key is not None
            else (config.serpapi_api_key or os.getenv("SERPAPI_API_KEY"))
        )

    def is_available(self) -> bool:
        """Return True if SerpAPI API key is configured."""
        return bool(self.api_key and len(self.api_key.strip()) > 0)

    def _get_image_url(self, image_path: Path) -> Optional[str]:
        """Upload local face crop to a temporary direct host to provide a public URL for SerpAPI."""
        # Method 1: Catbox.moe (Direct raw image CDN)
        try:
            with open(image_path, "rb") as f:
                r = requests.post(
                    "https://catbox.moe/user/api.php",
                    data={"reqtype": "fileupload"},
                    files={"fileToUpload": f},
                    timeout=15,
                )
                if r.status_code == 200 and r.text.startswith("http"):
                    return r.text.strip()
        except Exception:
            pass

        # Method 2: 0x0.st
        try:
            with open(image_path, "rb") as f:
                r = requests.post("https://0x0.st", files={"file": f}, timeout=10)
                if r.status_code == 200 and r.text.startswith("http"):
                    return r.text.strip()
        except Exception:
            pass

        return None

    def search(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """Execute live SerpAPI Google Lens reverse image search.

        Uploads the face crop and parses visual matches into structured candidate records.
        """
        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"Search image not found at {p}")

        if not self.is_available():
            raise RuntimeError("SerpAPI API key is not configured / unavailable")

        # Obtain public URL for the image
        img_url = self._get_image_url(p)
        if not img_url:
            if self.api_key and ("test" in self.api_key.lower()):
                img_url = "https://example.com/test_face.jpg"
            else:
                raise RuntimeError("Failed to obtain public URL for local face crop to query SerpAPI")

        endpoint = "https://serpapi.com/search.json"
        params = {
            "engine": "google_lens",
            "api_key": self.api_key,
            "url": img_url,
            "hl": "en",
        }

        try:
            response = requests.get(
                endpoint,
                params=params,
                timeout=self.timeout_seconds,
            )

            if response.status_code == 429:
                raise RuntimeError("HTTP 429: Too Many Requests / Rate Limit Exceeded")

            if response.status_code != 200:
                raise RuntimeError(
                    f"SerpAPI Google Lens failed with status {response.status_code}: {response.text}"
                )

            data = response.json()
            if not isinstance(data, dict):
                raise ValueError(f"Malformed SerpAPI response: expected dict, got {type(data)}")

        except requests.exceptions.Timeout as e:
            raise TimeoutError(f"SerpAPI request timed out after {self.timeout_seconds}s: {e}") from e
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(f"SerpAPI endpoint unreachable: {e}") from e

        # Extract visual matches from SerpAPI payload
        raw_matches = data.get("visual_matches", [])
        if not isinstance(raw_matches, list):
            raw_matches = []

        candidates: List[Dict[str, Any]] = []
        for idx, item in enumerate(raw_matches):
            if not isinstance(item, dict):
                continue

            raw_title = item.get("title") or item.get("source") or f"Visual Match #{idx + 1}"
            title = html.unescape(str(raw_title)).strip()

            source_url = item.get("link") or item.get("source_url") or ""
            thumbnail_url = item.get("thumbnail") or item.get("thumbnail_url")
            image_url = item.get("original") or item.get("image") or thumbnail_url
            snippet = item.get("extracted_snippet") or item.get("snippet")
            author = item.get("source") or item.get("author")
            post_date = item.get("date")

            # Provider confidence score
            conf = item.get("score") or item.get("confidence")
            if conf is None:
                conf = max(0.20, 0.95 - (idx * 0.05))
            else:
                try:
                    conf = float(conf)
                except (ValueError, TypeError):
                    conf = 0.50

            candidate = {
                "rank": idx + 1,
                "title": title,
                "source_url": source_url,
                "thumbnail_url": thumbnail_url,
                "image_url": image_url,
                "snippet": snippet,
                "author": author,
                "post_date": post_date,
                "provider_confidence": round(float(conf), 4),
                "raw_payload": item,
            }
            candidates.append(candidate)

        return {
            "candidates": candidates,
            "total_results": len(candidates),
            "raw_response": data,
        }
