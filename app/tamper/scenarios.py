"""
app/tamper/scenarios.py - Automated 5-Scenario Tamper Attack Matrix.

Implements Requirement R6:
1. Caption modification: Injects malicious text / payload into caption.
2. Timestamp modification: Shifts post timestamp by +3 hours / replaces ISO timestamp.
3. Media hash modification: Alters image SHA-256 cryptographic digest.
4. Removed metadata field: Deletes mandatory metadata field (e.g., 'author').
5. Altered source URL: Modifies post origin URL to a fraudulent/malicious domain.
"""

import copy
from typing import Any, Dict, List, NamedTuple, Tuple


class TamperScenarioDefinition(NamedTuple):
    scenario_id: str
    scenario_name: str
    description: str
    attack_vector: str
    mutated_dict: Dict[str, Any]


def mutate_caption(original: Dict[str, Any], injection: str = " [TAMPERED_MALICIOUS_INJECTION]") -> Dict[str, Any]:
    """Scenario 1: Modifies post caption text."""
    t = copy.deepcopy(original)
    current_caption = t.get("caption", "")
    t["caption"] = current_caption + injection
    return t


def mutate_timestamp(original: Dict[str, Any], new_timestamp: str = "2026-09-01T15:00:00Z") -> Dict[str, Any]:
    """Scenario 2: Modifies post timestamp."""
    t = copy.deepcopy(original)
    t["post_timestamp"] = new_timestamp
    return t


def mutate_media_hash(original: Dict[str, Any], fake_hash: str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff") -> Dict[str, Any]:
    """Scenario 3: Modifies media SHA-256 cryptographic digest."""
    t = copy.deepcopy(original)
    t["media_sha256"] = fake_hash
    return t


def mutate_remove_field(original: Dict[str, Any], field_name: str = "author") -> Dict[str, Any]:
    """Scenario 4: Removes mandatory metadata field."""
    t = copy.deepcopy(original)
    t.pop(field_name, None)
    return t


def mutate_source_url(original: Dict[str, Any], fake_url: str = "https://malicious-tampered-site.org/alice/fake_post/789102") -> Dict[str, Any]:
    """Scenario 5: Alters post source URL."""
    t = copy.deepcopy(original)
    t["source_url"] = fake_url
    return t


def get_all_tamper_scenarios(original: Dict[str, Any]) -> List[TamperScenarioDefinition]:
    """
    Generates all 5 standard tamper attack scenarios against the given canonical baseline.
    """
    return [
        TamperScenarioDefinition(
            scenario_id="SCENARIO_1_MODIFIED_CAPTION",
            scenario_name="Modified post caption/text",
            description="Appended malicious injection to caption field",
            attack_vector="Modified Caption",
            mutated_dict=mutate_caption(original),
        ),
        TamperScenarioDefinition(
            scenario_id="SCENARIO_2_MODIFIED_TIMESTAMP",
            scenario_name="Modified post timestamp (+3 hours)",
            description="Replaced post timestamp with altered UTC timestamp",
            attack_vector="Modified Timestamp",
            mutated_dict=mutate_timestamp(original),
        ),
        TamperScenarioDefinition(
            scenario_id="SCENARIO_3_MODIFIED_MEDIA_HASH",
            scenario_name="Modified image/media SHA-256 digest",
            description="Replaced media SHA-256 with invalid cryptographic digest",
            attack_vector="Modified Media Hash",
            mutated_dict=mutate_media_hash(original),
        ),
        TamperScenarioDefinition(
            scenario_id="SCENARIO_4_REMOVED_FIELD",
            scenario_name="Removed mandatory field 'author'",
            description="Deleted required 'author' key from canonical metadata",
            attack_vector="Removed Metadata Field",
            mutated_dict=mutate_remove_field(original, "author"),
        ),
        TamperScenarioDefinition(
            scenario_id="SCENARIO_5_ALTERED_SOURCE_URL",
            scenario_name="Altered source URL to fake domain",
            description="Changed provenance source URL to spoofed destination domain",
            attack_vector="Altered Source URL",
            mutated_dict=mutate_source_url(original),
        ),
    ]
