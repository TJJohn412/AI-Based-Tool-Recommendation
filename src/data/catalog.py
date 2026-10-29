from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG_PATH = PROJECT_ROOT / "dataset" / "tool_reference_catalog.csv"

DEFAULT_REFERENCE_CATALOG: dict[str, dict[str, dict[str, Any]]] = {
    "Steel": {
        "Turning": {"speed_m_min": 110.0, "feed_mm_rev": 0.25, "doc_mm": 2.5, "tool_type": "Indexable turning insert", "tool_material": "Carbide insert"},
        "Milling": {"speed_m_min": 120.0, "feed_mm_rev": 0.22, "doc_mm": 2.2, "tool_type": "End mill", "tool_material": "Carbide end mill"},
        "Drilling": {"speed_m_min": 35.0, "feed_mm_rev": 0.18, "doc_mm": 2.0, "tool_type": "Twist drill", "tool_material": "Cobalt twist drill"},
    },
    "Stainless Steel": {
        "Turning": {"speed_m_min": 80.0, "feed_mm_rev": 0.18, "doc_mm": 1.8, "tool_type": "Indexable turning insert", "tool_material": "Coated carbide"},
        "Milling": {"speed_m_min": 90.0, "feed_mm_rev": 0.16, "doc_mm": 1.5, "tool_type": "End mill", "tool_material": "Coated carbide end mill"},
        "Drilling": {"speed_m_min": 22.0, "feed_mm_rev": 0.12, "doc_mm": 1.4, "tool_type": "Twist drill", "tool_material": "Solid carbide drill"},
    },
    "Aluminium": {
        "Turning": {"speed_m_min": 260.0, "feed_mm_rev": 0.35, "doc_mm": 3.5, "tool_type": "Indexable turning insert", "tool_material": "Polished carbide"},
        "Milling": {"speed_m_min": 220.0, "feed_mm_rev": 0.30, "doc_mm": 3.0, "tool_type": "End mill", "tool_material": "High-speed steel or carbide"},
        "Drilling": {"speed_m_min": 110.0, "feed_mm_rev": 0.24, "doc_mm": 2.4, "tool_type": "Twist drill", "tool_material": "HSS drill"},
    },
    "Titanium": {
        "Turning": {"speed_m_min": 55.0, "feed_mm_rev": 0.12, "doc_mm": 1.2, "tool_type": "Indexable turning insert", "tool_material": "Ceramic or carbide"},
        "Milling": {"speed_m_min": 45.0, "feed_mm_rev": 0.10, "doc_mm": 1.0, "tool_type": "End mill", "tool_material": "Carbide end mill"},
        "Drilling": {"speed_m_min": 18.0, "feed_mm_rev": 0.08, "doc_mm": 0.9, "tool_type": "Twist drill", "tool_material": "Solid carbide"},
    },
}


def _normalize_catalog_value(value: str | None) -> float | int | str | None:
    if value is None:
        return None
    text = value.strip()
    if text == "":
        return None
    try:
        if "." in text or "e" in text.lower():
            return float(text)
        return int(text)
    except ValueError:
        return text


def parse_reference_catalog_text(csv_text: str) -> dict[str, dict[str, dict[str, Any]]]:
    """Parse a CSV catalog from a string payload."""
    catalog: dict[str, dict[str, dict[str, Any]]] = {}
    if not csv_text or not csv_text.strip():
        return catalog

    import io

    reader = csv.DictReader(io.StringIO(csv_text))
    for row in reader:
        material = (row.get("material_category") or "").strip()
        operation = (row.get("operation") or "").strip()
        if not material or not operation:
            continue
        entry = {
            "speed_m_min": _normalize_catalog_value(row.get("speed_m_min")),
            "feed_mm_rev": _normalize_catalog_value(row.get("feed_mm_rev")),
            "doc_mm": _normalize_catalog_value(row.get("doc_mm")),
            "tool_type": row.get("tool_type") or "General-purpose cutting tool",
            "tool_material": row.get("tool_material") or "General-purpose tool material",
            "source": row.get("source") or "Imported reference catalog",
        }
        catalog.setdefault(material, {})[operation] = entry
    return catalog


def load_reference_catalog(csv_path: str | Path = DEFAULT_CATALOG_PATH, catalog_override: dict[str, dict[str, dict[str, Any]]] | None = None) -> dict[str, dict[str, dict[str, Any]]]:
    """Load a catalog from CSV if available; otherwise fall back to built-in defaults."""
    if catalog_override:
        return catalog_override

    catalog_path = Path(csv_path)
    if not catalog_path.exists():
        return DEFAULT_REFERENCE_CATALOG

    catalog: dict[str, dict[str, dict[str, Any]]] = {}
    with catalog_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            material = (row.get("material_category") or "").strip()
            operation = (row.get("operation") or "").strip()
            if not material or not operation:
                continue
            operation_key = operation.strip()
            entry = {
                "speed_m_min": _normalize_catalog_value(row.get("speed_m_min")),
                "feed_mm_rev": _normalize_catalog_value(row.get("feed_mm_rev")),
                "doc_mm": _normalize_catalog_value(row.get("doc_mm")),
                "tool_type": row.get("tool_type") or "General-purpose cutting tool",
                "tool_material": row.get("tool_material") or "General-purpose tool material",
                "source": row.get("source") or "Imported reference catalog",
            }
            catalog.setdefault(material, {})[operation_key] = entry

    return catalog or DEFAULT_REFERENCE_CATALOG


def get_reference_limits(material_category: str, operation: str, csv_path: str | Path = DEFAULT_CATALOG_PATH, catalog_override: dict[str, dict[str, dict[str, Any]]] | None = None) -> dict[str, Any] | None:
    """Look up a material-specific reference limit for a given operation."""
    catalog = load_reference_catalog(csv_path, catalog_override=catalog_override)
    material_rules = catalog.get(material_category)
    if not material_rules:
        return None

    operation_key = operation
    if operation in {"Turning", "Facing", "Parting", "Grooving", "Chamfering", "Threading", "Boring"}:
        operation_key = "Turning"
    elif operation in {"Milling", "Slot Milling", "End Milling", "Side Milling"}:
        operation_key = "Milling"
    elif operation in {"Drilling", "Tapping"}:
        operation_key = "Drilling"

    return material_rules.get(operation_key)
