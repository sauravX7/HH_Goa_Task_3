"""
app/tamper/differ.py - Deep field-level diff generator between original canonical metadata and tampered mutations.

Implements Requirement R6:
- Compares original and mutated dictionaries across all fields.
- Identifies modified, added, and deleted fields.
- Formats impact descriptions and structures diff records for UI presentation and reporting.
"""

from typing import Any, Dict, List, Optional
from app.models import FieldDiff


class TamperDiffEngine:
    """
    Computes field-level differences between original canonical dict and tampered dict.
    """

    @staticmethod
    def compute_diffs(original: Dict[str, Any], tampered: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Calculates field-by-field differences returning standardized dictionaries:
        - field_name: Name of the key
        - original_value: Value in original dict or '<NOT_PRESENT>'
        - tampered_value: Value in tampered dict or '<MISSING>'
        - impact_description: Human-readable explanation of the change
        """
        diffs: List[Dict[str, Any]] = []
        all_keys = sorted(set(original.keys()) | set(tampered.keys()))

        for k in all_keys:
            orig_val = original.get(k)
            tamp_val = tampered.get(k)

            if k not in original:
                diffs.append({
                    "field_name": k,
                    "original_value": "<NOT_PRESENT>",
                    "tampered_value": tamp_val,
                    "impact_description": "Field was unexpectedly added",
                })
            elif k not in tampered:
                diffs.append({
                    "field_name": k,
                    "original_value": orig_val,
                    "tampered_value": "<MISSING>",
                    "impact_description": "Mandatory canonical field was deleted",
                })
            elif orig_val != tamp_val:
                diffs.append({
                    "field_name": k,
                    "original_value": orig_val,
                    "tampered_value": tamp_val,
                    "impact_description": f"Field value modified from '{orig_val}' to '{tamp_val}'",
                })

        return diffs

    @classmethod
    def compute_field_diff_models(cls, original: Dict[str, Any], tampered: Dict[str, Any]) -> List[FieldDiff]:
        """Returns diffs as Pydantic FieldDiff instances."""
        raw_diffs = cls.compute_diffs(original, tampered)
        return [FieldDiff(**d) for d in raw_diffs]
