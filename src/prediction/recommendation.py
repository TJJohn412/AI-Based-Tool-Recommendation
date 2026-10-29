from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from src.data.catalog import get_reference_limits
from src.data.schema import (
    ACCURACY_REQUIREMENTS,
    DIMENSION_FIELDS,
    HARDNESS_LEVELS,
    MACHINE_TYPES,
    MATERIAL_CATEGORIES,
    OPERATIONS,
    PREFERENCES,
    STAGES,
    SURFACE_FINISHES,
    TOLERANCES,
    WORKPIECE_SHAPES,
    get_grade_options,
)
from src.features.preprocess import payload_to_feature_frame

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "tool_recommendation_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "tool_recommendation_metadata.json"

MACHINE_COMPATIBILITY: dict[str, set[str]] = {
    "Manual Lathe": {"Turning", "Facing", "Parting", "Grooving", "Chamfering", "Threading"},
    "CNC Lathe": {"Turning", "Facing", "Boring", "Threading", "Grooving", "Parting", "Chamfering"},
    "Vertical Milling Machine": {"Milling", "Slot Milling", "End Milling", "Side Milling", "Facing", "Chamfering"},
    "Horizontal Milling Machine": {"Milling", "Slot Milling", "End Milling", "Side Milling", "Facing", "Chamfering"},
    "VMC": {"Milling", "Slot Milling", "End Milling", "Side Milling", "Facing", "Chamfering"},
    "HMC": {"Milling", "Slot Milling", "End Milling", "Side Milling", "Facing", "Chamfering"},
    "Drill Machine": {"Drilling", "Boring", "Tapping", "Chamfering"},
}

MATERIAL_ENGINEERING_RULES: dict[str, dict[str, dict[str, float]]] = {
    "Steel": {"Turning": {"speed_m_min": 110.0, "feed_mm_rev": 0.25, "doc_mm": 2.5}, "Milling": {"speed_m_min": 120.0, "feed_mm_rev": 0.22, "doc_mm": 2.2}, "Drilling": {"speed_m_min": 35.0, "feed_mm_rev": 0.18, "doc_mm": 2.0}},
    "Stainless Steel": {"Turning": {"speed_m_min": 80.0, "feed_mm_rev": 0.18, "doc_mm": 1.8}, "Milling": {"speed_m_min": 90.0, "feed_mm_rev": 0.16, "doc_mm": 1.5}, "Drilling": {"speed_m_min": 22.0, "feed_mm_rev": 0.12, "doc_mm": 1.4}},
    "Cast Iron": {"Turning": {"speed_m_min": 100.0, "feed_mm_rev": 0.22, "doc_mm": 2.2}, "Milling": {"speed_m_min": 95.0, "feed_mm_rev": 0.18, "doc_mm": 2.0}, "Drilling": {"speed_m_min": 28.0, "feed_mm_rev": 0.16, "doc_mm": 1.6}},
    "Aluminium": {"Turning": {"speed_m_min": 260.0, "feed_mm_rev": 0.35, "doc_mm": 3.5}, "Milling": {"speed_m_min": 220.0, "feed_mm_rev": 0.30, "doc_mm": 3.0}, "Drilling": {"speed_m_min": 110.0, "feed_mm_rev": 0.24, "doc_mm": 2.4}},
    "Copper": {"Turning": {"speed_m_min": 200.0, "feed_mm_rev": 0.28, "doc_mm": 2.8}, "Milling": {"speed_m_min": 180.0, "feed_mm_rev": 0.24, "doc_mm": 2.5}, "Drilling": {"speed_m_min": 90.0, "feed_mm_rev": 0.20, "doc_mm": 2.0}},
    "Brass": {"Turning": {"speed_m_min": 220.0, "feed_mm_rev": 0.30, "doc_mm": 3.2}, "Milling": {"speed_m_min": 200.0, "feed_mm_rev": 0.26, "doc_mm": 2.8}, "Drilling": {"speed_m_min": 100.0, "feed_mm_rev": 0.22, "doc_mm": 2.2}},
    "Bronze": {"Turning": {"speed_m_min": 120.0, "feed_mm_rev": 0.22, "doc_mm": 2.4}, "Milling": {"speed_m_min": 110.0, "feed_mm_rev": 0.20, "doc_mm": 2.1}, "Drilling": {"speed_m_min": 45.0, "feed_mm_rev": 0.18, "doc_mm": 1.8}},
    "Titanium": {"Turning": {"speed_m_min": 55.0, "feed_mm_rev": 0.12, "doc_mm": 1.2}, "Milling": {"speed_m_min": 45.0, "feed_mm_rev": 0.10, "doc_mm": 1.0}, "Drilling": {"speed_m_min": 18.0, "feed_mm_rev": 0.08, "doc_mm": 0.9}},
    "Nickel Alloy": {"Turning": {"speed_m_min": 42.0, "feed_mm_rev": 0.10, "doc_mm": 1.0}, "Milling": {"speed_m_min": 35.0, "feed_mm_rev": 0.08, "doc_mm": 0.9}, "Drilling": {"speed_m_min": 14.0, "feed_mm_rev": 0.07, "doc_mm": 0.8}},
    "Plastic": {"Turning": {"speed_m_min": 160.0, "feed_mm_rev": 0.32, "doc_mm": 3.0}, "Milling": {"speed_m_min": 180.0, "feed_mm_rev": 0.30, "doc_mm": 2.8}, "Drilling": {"speed_m_min": 80.0, "feed_mm_rev": 0.22, "doc_mm": 2.0}},
    "Composite": {"Turning": {"speed_m_min": 120.0, "feed_mm_rev": 0.22, "doc_mm": 2.2}, "Milling": {"speed_m_min": 130.0, "feed_mm_rev": 0.20, "doc_mm": 1.8}, "Drilling": {"speed_m_min": 50.0, "feed_mm_rev": 0.15, "doc_mm": 1.4}},
}


def validate_input_form(payload: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Validate supplied inputs and return user-facing errors and warnings."""
    errors: list[str] = []
    warnings: list[str] = []

    if payload.get("material_category") not in MATERIAL_CATEGORIES:
        errors.append("Select a valid material category.")

    if payload.get("material_category") and payload.get("material_grade"):
        valid_grades = get_grade_options(payload["material_category"])
        if payload["material_grade"] not in valid_grades:
            errors.append("Selected material grade does not match the material category.")

    if payload.get("machine_type") not in MACHINE_TYPES:
        errors.append("Select a valid machine type.")

    if payload.get("machining_operation") not in OPERATIONS:
        errors.append("Select a valid machining operation.")

    if payload.get("machining_stage") not in STAGES:
        errors.append("Select a valid machining stage.")

    if payload.get("surface_finish") not in SURFACE_FINISHES:
        errors.append("Select a valid surface finish requirement.")

    if payload.get("tolerance") not in TOLERANCES:
        errors.append("Select a valid tolerance value.")

    if payload.get("accuracy_requirement") not in ACCURACY_REQUIREMENTS:
        errors.append("Select a valid accuracy requirement.")

    if payload.get("workpiece_shape") not in WORKPIECE_SHAPES:
        errors.append("Select a valid workpiece shape.")

    shape = payload.get("workpiece_shape")
    required_dims = DIMENSION_FIELDS.get(shape, [])
    for dim_name in required_dims:
        value = payload.get(dim_name)
        if value is None or value <= 0:
            errors.append(f"{dim_name.replace('_mm', ' (mm)')} is required for {shape} and must be greater than 0.")

    for field_name in [
        "machine_power_kw",
        "max_rpm",
        "feed_rate_value",
        "spindle_speed_rpm",
        "cutting_speed_m_min",
        "depth_of_cut_mm",
        "width_of_cut_mm",
    ]:
        value = payload.get(field_name)
        if value is not None and value <= 0:
            errors.append(f"{field_name.replace('_', ' ')} must be greater than 0.")

    if payload.get("machine_type") and payload.get("machining_operation"):
        allowed_ops = MACHINE_COMPATIBILITY.get(payload["machine_type"], set())
        if payload["machining_operation"] not in allowed_ops:
            warnings.append(
                f"{payload['machining_operation']} is not a normal match for {payload['machine_type']}; "
                "the recommendation is a heuristic only and should be reviewed by an engineer."
            )

    if payload.get("hardness_level") not in HARDNESS_LEVELS:
        warnings.append("Hardness level is missing or invalid; the recommendation may be conservative.")

    if payload.get("hardness_level") == "Very Hard" and payload.get("machine_rigidity") == "Low":
        warnings.append("Very hard material on a low-rigidity machine may require reduced DOC and feed to avoid chatter or tool failure.")

    if payload.get("max_rpm") is not None and payload.get("spindle_speed_rpm") is not None:
        if float(payload["spindle_speed_rpm"]) > float(payload["max_rpm"]):
            errors.append("Selected spindle speed exceeds the entered machine maximum RPM.")

    if payload.get("machine_power_kw") is not None and payload.get("machine_power_kw") <= 0:
        errors.append("Machine power must be positive.")

    if payload.get("cutting_speed_m_min") and payload.get("spindle_speed_rpm") and payload.get("diameter_mm"):
        if payload["diameter_mm"] > 0:
            calculated = (3.14159 * payload["diameter_mm"] * payload["spindle_speed_rpm"]) / 1000.0
            if abs(calculated - payload["cutting_speed_m_min"]) > 35:
                warnings.append(
                    "The entered cutting speed and spindle speed do not match the expected diameter-based relationship; "
                    "check the units before trusting the recommendation."
                )

    if payload.get("material_category") in {"Aluminium", "Plastic", "Composite"} and payload.get("machine_rigidity") == "Low":
        warnings.append("Low rigidity on light alloys or composites may increase tool deflection; lower DOC or use a more stable setup.")

    if payload.get("machining_operation") in {"Threading", "Tapping"} and payload.get("machine_type") not in {"CNC Lathe", "Drill Machine", "VMC", "HMC"}:
        warnings.append("Threading and tapping should generally be performed on a machine with stable spindle control and adequate threading support.")

    if payload.get("machining_operation") in {"Milling", "Slot Milling", "End Milling", "Side Milling"} and payload.get("step_over_mm") is not None and payload.get("step_over_mm") <= 0:
        warnings.append("Step-over value is zero or invalid for milling; the recommended toolpath should be reviewed.")

    material_rules = MATERIAL_ENGINEERING_RULES.get(payload.get("material_category"), {})
    operation = payload.get("machining_operation")
    if material_rules and operation:
        operation_key = operation
        if operation in {"Turning", "Facing", "Parting", "Grooving", "Chamfering", "Threading", "Boring"}:
            operation_key = "Turning"
        elif operation in {"Milling", "Slot Milling", "End Milling", "Side Milling"}:
            operation_key = "Milling"
        elif operation in {"Drilling", "Tapping"}:
            operation_key = "Drilling"

        limits = material_rules.get(operation_key)
        if limits:
            if payload.get("cutting_speed_m_min") is not None and float(payload["cutting_speed_m_min"]) > limits["speed_m_min"] * 1.35:
                warnings.append(f"Selected cutting speed is above the material-specific rule-of-thumb limit for {payload['material_category']} {operation}.")
            if payload.get("feed_rate_value") is not None and float(payload["feed_rate_value"]) > limits["feed_mm_rev"] * 1.45:
                warnings.append(f"Selected feed rate is above the material-specific guideline for {payload['material_category']} {operation}.")
            if payload.get("depth_of_cut_mm") is not None and float(payload["depth_of_cut_mm"]) > limits["doc_mm"] * 1.6:
                warnings.append(f"Selected depth of cut is above the conservative rule-based limit for {payload['material_category']} {operation}.")

    return errors, warnings


def _material_tool_family(material_category: str, hardness_level: str) -> str:
    if material_category in {"Steel", "Stainless Steel", "Titanium", "Nickel Alloy", "Cast Iron"}:
        return "Carbide / coated carbide insert" if hardness_level in {"Hard", "Very Hard"} else "Cermet / indexable cutting tool"
    if material_category in {"Aluminium", "Copper", "Brass", "Plastic", "Composite"}:
        return "HSS or polished carbide cutter"
    return "General-purpose cutting tool"


def _rule_based_tool_material(material_category: str, hardness_level: str) -> str:
    if material_category in {"Steel", "Stainless Steel", "Cast Iron", "Titanium", "Nickel Alloy"}:
        if hardness_level in {"Hard", "Very Hard"}:
            return "Coated carbide or ceramic insert"
        return "Cermet or standard carbide insert"
    if material_category in {"Aluminium", "Copper", "Brass", "Plastic", "Composite"}:
        return "HSS, polished carbide, or polycrystalline diamond where validated"
    return "General-purpose tool material"


def _operation_tool_type(operation: str, material_category: str) -> str:
    mapping: dict[str, str] = {
        "Turning": "Indexable turning insert",
        "Facing": "Facing insert",
        "Drilling": "Twist drill",
        "Milling": "End mill",
        "Slot Milling": "Slotting cutter",
        "End Milling": "End mill",
        "Side Milling": "Side milling cutter",
        "Boring": "Boring bar",
        "Threading": "Threading insert or thread mill",
        "Grooving": "Grooving insert",
        "Parting": "Parting tool",
        "Chamfering": "Chamfer cutter",
        "Tapping": "Machine tap",
    }
    if operation == "Drilling" and material_category in {"Aluminium", "Plastic"}:
        return "High-speed steel twist drill"
    return mapping.get(operation, "General-purpose cutting tool")


def _cutting_speed_range(operation: str, material_category: str, hardness_level: str) -> str:
    base_ranges: dict[str, dict[str, tuple[int, int]]] = {
        "Steel": {"Turning": (90, 180), "Facing": (100, 170), "Drilling": (25, 60), "Milling": (80, 150), "Boring": (60, 120), "Threading": (20, 50), "Grooving": (70, 130), "Parting": (60, 110), "Chamfering": (50, 90), "Tapping": (10, 25)},
        "Stainless Steel": {"Turning": (60, 140), "Facing": (70, 130), "Drilling": (15, 35), "Milling": (50, 110), "Boring": (40, 90), "Threading": (12, 30), "Grooving": (45, 90), "Parting": (40, 80), "Chamfering": (30, 70), "Tapping": (8, 20)},
        "Cast Iron": {"Turning": (80, 160), "Facing": (90, 170), "Drilling": (20, 55), "Milling": (70, 140), "Boring": (55, 100), "Threading": (15, 35), "Grooving": (55, 110), "Parting": (50, 100), "Chamfering": (40, 80), "Tapping": (10, 25)},
        "Aluminium": {"Turning": (200, 500), "Facing": (180, 420), "Drilling": (60, 200), "Milling": (150, 350), "Boring": (120, 300), "Threading": (30, 90), "Grooving": (120, 250), "Parting": (100, 250), "Chamfering": (100, 220), "Tapping": (20, 60)},
        "Copper": {"Turning": (180, 350), "Facing": (160, 300), "Drilling": (40, 120), "Milling": (120, 260), "Boring": (100, 220), "Threading": (20, 80), "Grooving": (100, 230), "Parting": (80, 200), "Chamfering": (80, 180), "Tapping": (15, 45)},
        "Brass": {"Turning": (180, 400), "Facing": (170, 360), "Drilling": (50, 140), "Milling": (170, 320), "Boring": (110, 240), "Threading": (25, 85), "Grooving": (110, 260), "Parting": (90, 220), "Chamfering": (90, 210), "Tapping": (18, 50)},
        "Bronze": {"Turning": (110, 220), "Facing": (100, 200), "Drilling": (30, 80), "Milling": (90, 180), "Boring": (70, 150), "Threading": (18, 45), "Grooving": (70, 160), "Parting": (60, 140), "Chamfering": (60, 130), "Tapping": (13, 35)},
        "Titanium": {"Turning": (40, 100), "Facing": (45, 100), "Drilling": (12, 30), "Milling": (30, 70), "Boring": (25, 60), "Threading": (8, 25), "Grooving": (25, 60), "Parting": (20, 50), "Chamfering": (25, 60), "Tapping": (6, 18)},
        "Nickel Alloy": {"Turning": (30, 80), "Facing": (35, 90), "Drilling": (10, 25), "Milling": (25, 60), "Boring": (20, 50), "Threading": (10, 25), "Grooving": (20, 50), "Parting": (20, 45), "Chamfering": (20, 50), "Tapping": (6, 18)},
        "Plastic": {"Turning": (120, 300), "Facing": (120, 260), "Drilling": (40, 150), "Milling": (150, 350), "Boring": (50, 180), "Threading": (20, 70), "Grooving": (80, 200), "Parting": (70, 180), "Chamfering": (70, 180), "Tapping": (15, 60)},
        "Composite": {"Turning": (80, 220), "Facing": (90, 200), "Drilling": (25, 100), "Milling": (100, 240), "Boring": (60, 150), "Threading": (15, 50), "Grooving": (60, 160), "Parting": (50, 140), "Chamfering": (55, 150), "Tapping": (12, 35)},
    }

    if material_category not in base_ranges:
        return "Use supplier data for the selected material family."

    low, high = base_ranges[material_category].get(operation, (20, 100))
    if hardness_level in {"Hard", "Very Hard"}:
        low = max(8, int(low * 0.7))
        high = max(low + 10, int(high * 0.8))
    elif hardness_level == "Soft":
        low = int(low * 1.1)
        high = int(high * 1.2)
    return f"{low}-{high} m/min (synthetic engineering estimate)"


def _feed_rate_range(operation: str, material_category: str, hardness_level: str) -> str:
    material_adjustments: dict[str, tuple[float, float]] = {
        "Steel": (0.10, 0.45),
        "Stainless Steel": (0.08, 0.35),
        "Cast Iron": (0.12, 0.55),
        "Aluminium": (0.15, 0.80),
        "Copper": (0.12, 0.60),
        "Brass": (0.15, 0.75),
        "Bronze": (0.10, 0.50),
        "Titanium": (0.05, 0.25),
        "Nickel Alloy": (0.04, 0.20),
        "Plastic": (0.10, 0.80),
        "Composite": (0.08, 0.60),
    }
    low, high = material_adjustments.get(material_category, (0.10, 0.40))
    if operation in {"Milling", "End Milling", "Side Milling", "Slot Milling"}:
        low *= 1.2
        high *= 1.5
    if hardness_level in {"Hard", "Very Hard"}:
        low *= 0.75
        high *= 0.85
    return f"{low:.2f}-{high:.2f} mm/rev (synthetic)"


def _depth_of_cut_range(operation: str, material_category: str, hardness_level: str) -> str:
    if operation in {"Turning", "Facing", "Parting", "Grooving", "Chamfering"}:
        if hardness_level in {"Hard", "Very Hard"}:
            return "0.5-2.0 mm"
        return "1.0-4.0 mm"
    if operation in {"Milling", "End Milling", "Side Milling", "Slot Milling"}:
        if material_category in {"Aluminium", "Plastic", "Composite"}:
            return "0.5-3.5 mm"
        if hardness_level in {"Hard", "Very Hard"}:
            return "0.5-2.5 mm"
        return "1.0-4.0 mm"
    if operation in {"Drilling", "Boring", "Tapping"}:
        return "0.5-3.0 mm (depth depends on hole geometry)"
    return "Use machine and tool manufacturer guidance."


def _load_model_assets() -> tuple[Any | None, dict[str, Any] | None, list[str]]:
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        return None, None, []

    try:
        model = joblib.load(MODEL_PATH)
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        return model, metadata, metadata.get("feature_names", [])
    except Exception:
        return None, None, []


def _predict_with_model(payload: dict[str, Any]) -> tuple[str | None, str, float]:
    model, metadata, feature_names = _load_model_assets()
    if model is None or not feature_names:
        return None, "Engineering-rule-based recommendation", 0.0

    try:
        feature_frame = payload_to_feature_frame(payload, feature_names)
        prediction = model.predict(feature_frame)[0]
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(feature_frame)[0]
            return prediction, "Hybrid ML + engineering-rule recommendation", float(max(probs))
        return prediction, "Hybrid ML + engineering-rule recommendation", 0.55
    except Exception:
        return None, "Engineering-rule-based recommendation", 0.0


def _rule_consistency_score(payload: dict[str, Any], rule_limits: dict[str, float]) -> tuple[str, float]:
    if not rule_limits:
        return "No rule baseline", 0.5

    speed = float(payload.get("cutting_speed_m_min") or 0)
    feed = float(payload.get("feed_rate_value") or 0)
    doc = float(payload.get("depth_of_cut_mm") or 0)
    if speed <= 0 or feed <= 0 or doc <= 0:
        return "Needs user values", 0.55

    checks = [
        speed <= rule_limits.get("speed_m_min", float("inf")) * 1.35,
        feed <= rule_limits.get("feed_mm_rev", float("inf")) * 1.45,
        doc <= rule_limits.get("doc_mm", float("inf")) * 1.6,
    ]
    score = sum(float(item) for item in checks) / len(checks)
    if score >= 0.8:
        return "Consistent with rule window", score
    if score >= 0.5:
        return "Borderline but usable", score
    return "Outside rule window", score


def _confidence_label(score: float) -> str:
    if score >= 0.8:
        return "High confidence"
    if score >= 0.6:
        return "Moderate confidence"
    if score >= 0.4:
        return "Low confidence"
    return "Very low confidence"


def recommend_tool(payload: dict[str, Any], catalog_override: dict[str, dict[str, dict[str, Any]]] | None = None) -> dict[str, Any]:
    """Return a structured recommendation using engineering heuristics and synthetic rules."""
    material_category = payload.get("material_category")
    operation = payload.get("machining_operation")
    stage = payload.get("machining_stage")
    hardness_level = payload.get("hardness_level", "Medium")
    machine_type = payload.get("machine_type")
    tool_family = _material_tool_family(material_category or "Steel", hardness_level)
    tool_type = _operation_tool_type(operation or "Turning", material_category or "Steel")
    ml_prediction, source_type, model_confidence = _predict_with_model(payload)
    material_rule_summary = MATERIAL_ENGINEERING_RULES.get(material_category or "Steel", {})
    operation_key = operation or "Turning"
    if operation_key in {"Turning", "Facing", "Parting", "Grooving", "Chamfering", "Threading", "Boring"}:
        operation_key = "Turning"
    elif operation_key in {"Milling", "Slot Milling", "End Milling", "Side Milling"}:
        operation_key = "Milling"
    elif operation_key in {"Drilling", "Tapping"}:
        operation_key = "Drilling"

    reference_limits = get_reference_limits(material_category or "Steel", operation or "Turning", catalog_override=catalog_override)
    if reference_limits:
        material_rule_summary = {operation_key: {
            "speed_m_min": float(reference_limits.get("speed_m_min", 0.0)),
            "feed_mm_rev": float(reference_limits.get("feed_mm_rev", 0.0)),
            "doc_mm": float(reference_limits.get("doc_mm", 0.0)),
        }}
        tool_type = reference_limits.get("tool_type", tool_type)
        tool_family = reference_limits.get("tool_material", tool_family)

    rule_limits = material_rule_summary.get(operation_key, {})
    rule_status, rule_confidence = _rule_consistency_score(payload, rule_limits)
    machine_compatible = machine_type and operation in MACHINE_COMPATIBILITY.get(machine_type, set())
    compatibility_score = 0.85 if machine_compatible else 0.45
    confidence_score = min(1.0, max(0.0, (model_confidence if model_confidence else 0.55) * 0.6 + rule_confidence * 0.3 + compatibility_score * 0.1))
    confidence_label = _confidence_label(confidence_score)
    agreement_status = "Strong agreement" if rule_status == "Consistent with rule window" and model_confidence >= 0.5 else "Check recommendation"
    if source_type == "Engineering-rule-based recommendation":
        agreement_status = "Rule-based fallback"
    rule_text = (
        f"Rule-based safety window: {rule_limits.get('speed_m_min', 0)} m/min, {rule_limits.get('feed_mm_rev', 0)} mm/rev, {rule_limits.get('doc_mm', 0)} mm DOC."
        if rule_limits
        else "No material-specific rule window available for this combination."
    )
    if reference_limits:
        catalog_source = reference_limits.get("source") or "Imported reference catalog"
        data_source = f"{catalog_source} and engineering heuristics; this is a decision-support prototype, not a manufacturer specification."
    else:
        data_source = "Engineering rules based on synthetic dataset and machining heuristics; this is a decision-support prototype, not a manufacturer specification."

    result = {
        "recommended_tool_type": f"{tool_type} ({tool_family})",
        "ml_prediction": ml_prediction,
        "tool_material_or_grade": _rule_based_tool_material(material_category or "Steel", hardness_level),
        "suitable_operation": operation,
        "recommended_cutting_speed_range": _cutting_speed_range(operation or "Turning", material_category or "Steel", hardness_level),
        "recommended_feed_rate_range": _feed_rate_range(operation or "Turning", material_category or "Steel", hardness_level),
        "suggested_depth_of_cut_range": _depth_of_cut_range(operation or "Turning", material_category or "Steel", hardness_level),
        "machine_compatibility": "Compatible" if machine_compatible else "Check compatibility",
        "coolant_recommendation": _coolant_recommendation(material_category or "Steel"),
        "tool_life_cost_trade_off": _trade_off_summary(material_category or "Steel", operation or "Turning", stage or "Roughing"),
        "explanation": _build_explanation(material_category, hardness_level, operation, machine_type),
        "validation_warnings": [],
        "material_rule_summary": rule_text,
        "data_source": data_source,
        "source_type": source_type,
        "model_confidence": round(model_confidence, 3) if model_confidence else 0.0,
        "confidence_score": round(confidence_score, 3),
        "confidence_label": confidence_label,
        "agreement_status": agreement_status,
        "rule_status": rule_status,
        "reference_catalog_used": bool(reference_limits),
    }
    return result


def _coolant_recommendation(material_category: str) -> str:
    if material_category in {"Steel", "Stainless Steel", "Titanium", "Nickel Alloy"}:
        return "Flood coolant or high-pressure coolant recommended."
    if material_category in {"Aluminium", "Copper", "Brass", "Plastic", "Composite"}:
        return "Soluble coolant, mist, or dry machining may be suitable depending on chip control and part finish."
    if material_category == "Cast Iron":
        return "Dry cutting or light mist is often acceptable."
    return "Use coolant compatible with the work material and machine tool inventory."


def _trade_off_summary(material_category: str, operation: str, stage: str) -> str:
    if material_category in {"Steel", "Stainless Steel", "Titanium", "Nickel Alloy"}:
        return "Higher tool life and stability usually win over raw productivity when the material is hard; lower feed may be needed at roughing stage."
    if material_category in {"Aluminium", "Plastic"}:
        return "Higher productivity is usually achievable, but chip evacuation and tool wear control remain important."
    return "Balanced feed and speed are typically preferred to maintain finish without unnecessary tool wear."


def _build_explanation(material_category: str, hardness_level: str, operation: str, machine_type: str) -> str:
    if material_category is None:
        material_category = "Steel"
    material_phrase = f"{material_category} with {hardness_level.lower()} hardness"
    machine_phrase = f"on a {machine_type}" if machine_type else "for the selected machine"
    return (
        f"The recommendation is based on the selected {material_phrase}, the {operation.lower()} process, and the machine context {machine_phrase}. "
        "The rule set prefers harder, more wear-resistant tool materials for harder workpieces and more aggressive cutting conditions for lighter materials. "
        "This output is intended for prototype decision support and should still be verified against machine limits and manufacturer guidance."
    )


def build_recommendation_report(payload: dict[str, Any], recommendation: dict[str, Any]) -> str:
    """Create a plain-text report for export or audit use."""
    confidence = recommendation.get('confidence_label', 'No confidence score available')
    confidence_value = recommendation.get('confidence_score', 0)
    agreement = recommendation.get('agreement_status', 'No status available')
    model_confidence = recommendation.get('model_confidence', 0)
    lines = [
        "SMART MACHINING ASSISTANT - Recommendation Report",
        "=" * 48,
        f"Material: {payload.get('material_category')} / {payload.get('material_grade')}",
        f"Hardness: {payload.get('hardness_level') or 'Not supplied'}",
        f"Workpiece shape: {payload.get('workpiece_shape')}",
        f"Operation: {payload.get('machining_operation')} ({payload.get('machining_stage')})",
        f"Machine: {payload.get('machine_type')} | Rigidity: {payload.get('machine_rigidity')}",
        "",
        f"Recommended tool type: {recommendation.get('recommended_tool_type')}",
        f"ML prediction: {recommendation.get('ml_prediction') or 'Not available'}",
        f"Recommendation confidence: {confidence} ({confidence_value:.2f})",
        f"Model confidence: {model_confidence:.2f}",
        f"Agreement status: {agreement}",
        f"Reference catalog status: {'Uploaded supplier catalog' if recommendation.get('reference_catalog_used') else 'Built-in synthetic rules only'}",
        f"Cutting speed range: {recommendation.get('recommended_cutting_speed_range')}",
        f"Feed-rate range: {recommendation.get('recommended_feed_rate_range')}",
        f"Depth-of-cut range: {recommendation.get('suggested_depth_of_cut_range')}",
        f"Machine compatibility: {recommendation.get('machine_compatibility')}",
        f"Coolant: {recommendation.get('coolant_recommendation')}",
        f"Trade-off note: {recommendation.get('tool_life_cost_trade_off')}",
        f"Material rule summary: {recommendation.get('material_rule_summary', 'Not provided')}",
        "",
        "Why this recommendation was made:",
        recommendation.get('explanation', 'No explanation available.'),
        "",
        "Data source:",
        recommendation.get('data_source', 'Not provided.'),
    ]
    return "\n".join(lines)


def describe_preferences(preferences: list[str]) -> str:
    """Explain how user preferences affect the recommendation."""
    if not preferences:
        return "No explicit user preferences were supplied; the recommendation is balanced between cost, life, and productivity."
    return "User preferences are used as a weighting signal for the recommendation, preferring tool choices that better match the selected objective while keeping machine capability and tool wear in check."
