"""
app/tamper - Automated 5-scenario tamper simulation and field-level diff reporting engine.
"""

from app.tamper.differ import TamperDiffEngine
from app.tamper.engine import TamperDetector, TamperEngine, TamperSuiteRunner
from app.tamper.scenarios import (
    get_all_tamper_scenarios,
    mutate_caption,
    mutate_media_hash,
    mutate_remove_field,
    mutate_source_url,
    mutate_timestamp,
)

__all__ = [
    "TamperDetector",
    "TamperEngine",
    "TamperSuiteRunner",
    "TamperDiffEngine",
    "get_all_tamper_scenarios",
    "mutate_caption",
    "mutate_timestamp",
    "mutate_media_hash",
    "mutate_remove_field",
    "mutate_source_url",
]
