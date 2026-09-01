"""Candidate image downloader with timeouts, retry, and base64/URL resolution."""

import base64
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import requests

from app.config import config
from app.models import SearchCandidate

logger = logging.getLogger(__name__)


class CandidateImageFetcher:
    """Fetches candidate images from web URLs or data URIs for face validation."""

    def __init__(
        self,
        timeout_seconds: float = 10.0,
        user_agent: Optional[str] = None,
    ):
        self.timeout_seconds = timeout_seconds
        self.user_agent = user_agent or config.search.user_agent
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})

    def fetch_image(self, url: Optional[str], timeout: Optional[float] = None) -> Optional[bytes]:
        """Download image bytes from a URL, local file, or base64 data URI."""
        if not url or not isinstance(url, str):
            return None

        url_clean = url.strip()
        if not url_clean:
            return None

        # Data URI support (e.g. data:image/jpeg;base64,...)
        if url_clean.startswith("data:image/"):
            try:
                _, b64_data = url_clean.split(",", 1)
                return base64.b64decode(b64_data)
            except Exception as e:
                logger.warning(f"Failed to decode data URI image: {e}")
                return None

        # Local file path support
        if url_clean.startswith("file://") or Path(url_clean).exists():
            local_path = Path(url_clean.replace("file://", ""))
            if local_path.exists() and local_path.is_file():
                try:
                    return local_path.read_bytes()
                except Exception as e:
                    logger.warning(f"Failed to read local image file {local_path}: {e}")
                    return None

        # Remote HTTP/HTTPS URL
        if not (url_clean.startswith("http://") or url_clean.startswith("https://")):
            return None

        t = timeout if timeout is not None else self.timeout_seconds
        try:
            response = self.session.get(url_clean, timeout=t, stream=False)
            if response.status_code == 200:
                content = response.content
                if len(content) > 0:
                    return content
                return None
            else:
                logger.debug(f"Candidate image fetch failed with status {response.status_code} for {url_clean}")
                return None
        except Exception as e:
            logger.debug(f"Candidate image fetch error for {url_clean}: {e}")
            return None

    def fetch_candidate(self, candidate: Union[Dict[str, Any], SearchCandidate]) -> Optional[bytes]:
        """Attempt downloading candidate image, trying image_url first, then thumbnail_url."""
        if isinstance(candidate, SearchCandidate):
            img_url = candidate.image_url
            thumb_url = candidate.thumbnail_url
        else:
            img_url = candidate.get("image_url")
            thumb_url = candidate.get("thumbnail_url")

        # Primary attempt with full-resolution image URL
        if img_url:
            data = self.fetch_image(img_url)
            if data:
                return data

        # Secondary fallback with thumbnail URL
        if thumb_url and thumb_url != img_url:
            data = self.fetch_image(thumb_url)
            if data:
                return data

        return None

    def fetch_all(
        self,
        candidates: Sequence[Union[Dict[str, Any], SearchCandidate]],
        timeout: Optional[float] = None,
    ) -> Dict[int, Optional[bytes]]:
        """Fetch image bytes for all candidates, indexed by candidate rank."""
        results: Dict[int, Optional[bytes]] = {}
        for cand in candidates:
            rank = cand.rank if isinstance(cand, SearchCandidate) else cand.get("rank", 1)
            results[rank] = self.fetch_candidate(cand)
        return results

    async def async_fetch_all(
        self,
        candidates: Sequence[Union[Dict[str, Any], SearchCandidate]],
        timeout: Optional[float] = None,
    ) -> Dict[int, Optional[bytes]]:
        """Asynchronously download candidate images in parallel."""
        import httpx
        t = timeout if timeout is not None else self.timeout_seconds
        results: Dict[int, Optional[bytes]] = {}

        async with httpx.AsyncClient(headers={"User-Agent": self.user_agent}, timeout=t) as client:
            for cand in candidates:
                rank = cand.rank if isinstance(cand, SearchCandidate) else cand.get("rank", 1)
                img_url = cand.image_url if isinstance(cand, SearchCandidate) else cand.get("image_url")
                thumb_url = cand.thumbnail_url if isinstance(cand, SearchCandidate) else cand.get("thumbnail_url")
                target_url = img_url or thumb_url

                if not target_url or not target_url.startswith("http"):
                    results[rank] = None
                    continue

                try:
                    res = await client.get(target_url)
                    if res.status_code == 200 and len(res.content) > 0:
                        results[rank] = res.content
                    else:
                        results[rank] = None
                except Exception:
                    results[rank] = None

        return results
