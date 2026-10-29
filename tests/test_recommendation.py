from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.preprocess import payload_to_feature_frame
from src.prediction.recommendation import build_recommendation_report, recommend_tool, validate_input_form


def test_validate_input_form_accepts_basic_valid_payload() -> None:
    payload = {
        "material_category": "Steel",
        "material_grade": "EN8",
        "hardness_level": "Medium",
        "machine_type": "CNC Lathe",
        "machining_operation": "Turning",
        "machining_stage": "Finishing",
        "surface_finish": "Fine",
        "tolerance": "±0.1 mm",
        "accuracy_requirement": "Medium",
        "workpiece_shape": "Round Bar",
        "diameter_mm": 80,
        "length_mm": 200,
        "machine_power_kw": 15,
        "max_rpm": 5000,
        "feed_rate_value": 0.2,
        "feed_rate_unit": "mm/rev",
        "spindle_speed_rpm": 1500,
        "cutting_speed_m_min": 120,
        "depth_of_cut_mm": 2,
        "width_of_cut_mm": 10,
        "step_over_mm": 1,
        "machine_rigidity": "Medium",
    }
    errors, warnings = validate_input_form(payload)
    assert errors == []


def test_validate_input_form_rejects_invalid_dimensions() -> None:
    payload = {
        "material_category": "Steel",
        "material_grade": "EN8",
        "hardness_level": "Medium",
        "machine_type": "CNC Lathe",
        "machining_operation": "Turning",
        "machining_stage": "Finishing",
        "surface_finish": "Fine",
        "tolerance": "±0.1 mm",
        "accuracy_requirement": "Medium",
        "workpiece_shape": "Round Bar",
        "diameter_mm": 0,
        "length_mm": 200,
    }
    errors, warnings = validate_input_form(payload)
    assert any("diameter" in error.lower() for error in errors)


def test_validate_input_form_rejects_excess_spindle_speed() -> None:
    payload = {
        "material_category": "Steel",
        "material_grade": "EN8",
        "hardness_level": "Medium",
        "machine_type": "CNC Lathe",
        "machining_operation": "Turning",
        "machining_stage": "Finishing",
        "surface_finish": "Fine",
        "tolerance": "±0.1 mm",
        "accuracy_requirement": "Medium",
        "workpiece_shape": "Round Bar",
        "diameter_mm": 80,
        "length_mm": 200,
        "max_rpm": 1200,
        "spindle_speed_rpm": 1500,
        "machine_rigidity": "Medium",
    }
    errors, warnings = validate_input_form(payload)
    assert any("exceeds" in error.lower() for error in errors)


def test_payload_to_feature_frame_matches_expected_columns() -> None:
    payload = {
        "material_category": "Steel",
        "material_grade": "EN8",
        "hardness_level": "Medium",
        "hardness_value": 180,
        "hardness_unit": "HB",
        "workpiece_shape": "Round Bar",
        "length_mm": 200,
        "diameter_mm": 80,
        "machining_operation": "Turning",
        "machining_stage": "Finishing",
        "machine_type": "CNC Lathe",
        "machine_power_kw": 15,
        "max_rpm": 5000,
        "machine_rigidity": "Medium",
        "feed_rate_value": 0.2,
        "feed_rate_unit": "mm/rev",
        "spindle_speed_rpm": 1500,
        "cutting_speed_m_min": 120,
        "depth_of_cut_mm": 2,
        "width_of_cut_mm": 10,
        "step_over_mm": 1,
        "surface_finish": "Fine",
        "tolerance": "±0.1 mm",
        "accuracy_requirement": "Medium",
        "preference_lower_tool_cost": False,
        "preference_longer_tool_life": True,
        "preference_better_surface_finish": True,
        "preference_higher_productivity": False,
        "preference_lower_machining_time": False,
    }
    frame = payload_to_feature_frame(payload)
    assert not frame.empty
    assert "material_category_Steel" in frame.columns


def test_build_recommendation_report_contains_expected_sections() -> None:
    payload = {
        "material_category": "Steel",
        "material_grade": "EN8",
        "hardness_level": "Medium",
        "machining_operation": "Turning",
        "machining_stage": "Finishing",
        "machine_type": "CNC Lathe",
        "workpiece_shape": "Round Bar",
    }
    recommendation = {
        "recommended_tool_type": "Indexable turning insert",
        "ml_prediction": "CNMG 120408",
        "recommended_cutting_speed_range": "90-180 m/min",
        "recommended_feed_rate_range": "0.10-0.45 mm/rev",
        "suggested_depth_of_cut_range": "1.0-4.0 mm",
        "machine_compatibility": "Compatible",
        "coolant_recommendation": "Flood coolant recommended.",
        "tool_life_cost_trade_off": "Balanced trade-off",
        "explanation": "The recommendation is justified.",
        "data_source": "Synthetic engineering rules",
        "material_rule_summary": "Rule-based safety window: 110.0 m/min, 0.25 mm/rev, 2.5 mm DOC.",
    }
    report = build_recommendation_report(payload, recommendation)
    assert "SMART MACHINING ASSISTANT" in report
    assert "Recommended tool type" in report
    assert "ML prediction" in report


def test_recommend_tool_returns_expected_fields() -> None:
    result = recommend_tool({
        "material_category": "Steel",
        "material_grade": "AISI 4140",
        "hardness_level": "Hard",
        "machining_operation": "Milling",
        "machining_stage": "Roughing",
        "machine_type": "VMC",
        "cutting_speed_m_min": 120,
        "feed_rate_value": 0.22,
        "depth_of_cut_mm": 2.0,
    })
    assert "recommended_tool_type" in result
    assert "ml_prediction" in result
    assert "confidence_score" in result
    assert "confidence_label" in result
    assert "agreement_status" in result
    assert "cutting_speed" not in result["recommended_tool_type"].lower()
    assert "Suitable machining operation" not in result["recommended_tool_type"]
    assert result["machine_compatibility"] in {"Compatible", "Check compatibility"}


def test_recommend_tool_uses_reference_catalog_when_available() -> None:
    result = recommend_tool({
        "material_category": "Steel",
        "material_grade": "EN8",
        "hardness_level": "Medium",
        "machining_operation": "Turning",
        "machining_stage": "Finishing",
        "machine_type": "CNC Lathe",
        "cutting_speed_m_min": 100,
        "feed_rate_value": 0.20,
        "depth_of_cut_mm": 2.0,
    })
    assert result["reference_catalog_used"] is True
    assert "Imported reference catalog" in result["data_source"] or "reference catalog" in result["data_source"].lower()


def test_recommend_tool_accepts_catalog_override() -> None:
    override = {
        "Steel": {
            "Turning": {
                "speed_m_min": 120.0,
                "feed_mm_rev": 0.30,
                "doc_mm": 3.0,
                "tool_type": "Custom turning insert",
                "tool_material": "Custom carbide",
                "source": "Uploaded supplier catalog",
            }
        }
    }
    result = recommend_tool({
        "material_category": "Steel",
        "material_grade": "EN8",
        "hardness_level": "Medium",
        "machining_operation": "Turning",
        "machining_stage": "Finishing",
        "machine_type": "CNC Lathe",
        "cutting_speed_m_min": 115,
        "feed_rate_value": 0.25,
        "depth_of_cut_mm": 2.5,
    }, catalog_override=override)
    assert result["reference_catalog_used"] is True
    assert "Custom turning insert" in result["recommended_tool_type"]
    assert "Uploaded supplier catalog" in result["data_source"] or "supplier catalog" in result["data_source"].lower()
