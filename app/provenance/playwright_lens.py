"""Playwright headless browser Google Lens visual search automation."""

import html
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from app.config import config
from app.provenance.base import SearchProvider

logger = logging.getLogger(__name__)


class PlaywrightLensProvider(SearchProvider):
    """Automates Google Lens reverse visual search using a headless browser."""

    def __init__(
        self,
        headless: bool = True,
        timeout_seconds: int = 20,
        name: str = "playwright_google_lens",
    ):
        super().__init__(name=name, timeout_seconds=timeout_seconds)
        self.headless = headless

    def is_available(self) -> bool:
        """Check if Playwright package and browser are available."""
        try:
            import playwright  # noqa: F401
            return True
        except ImportError:
            return False

    def search(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """Execute browser-driven reverse image lookup via Google Lens."""
        p = Path(image_path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Search image not found at {p}")

        if not self.is_available():
            raise RuntimeError("Playwright is not installed or available in this environment")

        from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

        candidates: List[Dict[str, Any]] = []

        try:
            with sync_playwright() as p_play:
                browser = p_play.chromium.launch(
                    headless=self.headless,
                    args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
                )
                context = browser.new_context(
                    user_agent=config.search.user_agent,
                    viewport={"width": 1280, "height": 800},
                )
                page = context.new_page()
                page.set_default_timeout(self.timeout_seconds * 1000)

                # Navigate to Google Images / Lens upload interface
                page.goto("https://images.google.com?hl=en", wait_until="domcontentloaded")

                # Look for camera/lens icon or file input
                # Google Images has a camera button that reveals an upload input
                camera_button = page.query_selector("div[aria-label*='Search by image'], div[jsname*='R5mgy'], button[aria-label*='Search by image']")
                if camera_button:
                    camera_button.click()
                    page.wait_for_timeout(500)

                file_input = page.query_selector("input[type='file'], input[name='encoded_image']")
                if file_input:
                    file_input.set_input_files(str(p))
                    # Wait for results page navigation / dynamic match cards
                    page.wait_for_load_state("networkidle", timeout=self.timeout_seconds * 1000)
                else:
                    # Alternative direct navigation if Lens direct upload is reachable
                    page.goto("https://lens.google.com", wait_until="domcontentloaded")
                    file_input = page.query_selector("input[type='file']")
                    if file_input:
                        file_input.set_input_files(str(p))
                        page.wait_for_load_state("networkidle", timeout=self.timeout_seconds * 1000)

                # Extract visual search cards from the DOM
                elements = page.query_selector_all("a[href^='http']:has(img), div[data-item-id] a, div.Vd9M6 a")
                rank = 1
                seen_urls = set()

                for el in elements:
                    href = el.get_attribute("href")
                    if not href or href in seen_urls or "google.com" in href:
                        continue
                    seen_urls.add(href)

                    title_el = el.query_selector("div, span, h3")
                    title_text = title_el.inner_text() if title_el else f"Web Result #{rank}"
                    title = html.unescape(title_text).strip()

                    img_el = el.query_selector("img")
                    thumb_src = img_el.get_attribute("src") if img_el else None

                    candidates.append({
                        "rank": rank,
                        "title": title or f"Visual Match #{rank}",
                        "source_url": href,
                        "thumbnail_url": thumb_src,
                        "image_url": thumb_src,
                        "snippet": None,
                        "author": None,
                        "post_date": None,
                        "provider_confidence": round(max(0.20, 0.90 - (rank - 1) * 0.05), 4),
                        "raw_payload": {"href": href, "title": title},
                    })
                    rank += 1
                    if rank > config.search.max_candidates:
                        break

                browser.close()

        except PlaywrightTimeoutError as e:
            raise TimeoutError(f"Playwright Google Lens automation timed out: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Playwright Google Lens automation failed: {e}") from e

        return {
            "candidates": candidates,
            "total_results": len(candidates),
        }
