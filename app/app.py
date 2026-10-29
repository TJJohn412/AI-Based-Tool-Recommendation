from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.data.catalog import parse_reference_catalog_text
from src.data.schema import (
    ACCURACY_REQUIREMENTS,
    HARDNESS_LEVELS,
    HARDNESS_UNITS,
    MACHINE_RIGIDITY,
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
from src.prediction.recommendation import (
    build_recommendation_report,
    describe_preferences,
    recommend_tool,
    validate_input_form,
)


def _extract_numeric_range(range_text: str) -> tuple[float | None, float | None]:
    if not range_text or "-" not in range_text:
        return None, None
    try:
        numbers = [float(part.strip().split()[0]) for part in range_text.split("-")[:2]]
        if len(numbers) == 2:
            return numbers[0], numbers[1]
    except Exception:
        return None, None
    return None, None


st.set_page_config(page_title="SMART MACHINING ASSISTANT", page_icon="🛠️", layout="wide")

with open(Path(__file__).with_name("style.css"), encoding="utf-8") as css_file:
    st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)

st.title("SMART MACHINING ASSISTANT")
st.caption("AI-based decision-support prototype for machining-tool recommendation")

uploaded_catalog = st.sidebar.file_uploader("Optional supplier reference catalog (CSV)", type=["csv"], help="Upload a CSV with columns: material_category, operation, speed_m_min, feed_mm_rev, doc_mm, tool_type, tool_material, source")
catalog_override = None
if uploaded_catalog is not None:
    try:
        catalog_override = parse_reference_catalog_text(uploaded_catalog.read().decode("utf-8"))
        if catalog_override:
            st.sidebar.success("Supplier reference catalog loaded successfully.")
        else:
            st.sidebar.warning("The uploaded CSV could not be parsed; ignoring it.")
    except Exception:
        st.sidebar.warning("The uploaded CSV could not be parsed; ignoring it.")

with st.form("machining_assistant_form"):
    st.subheader("1. Workpiece Information")
    col1, col2 = st.columns(2)

    with col1:
        material_category = st.selectbox("Material category", MATERIAL_CATEGORIES, index=0)
        grade_options = get_grade_options(material_category)
        material_grade = st.selectbox("Material grade", grade_options if grade_options else ["N/A"], index=0)

        hardness_mode = st.selectbox("Hardness input mode", ["Level", "Numeric value"], index=0)
        if hardness_mode == "Level":
            hardness_level = st.selectbox("Hardness level", HARDNESS_LEVELS, index=1)
            hardness_value = None
            hardness_unit = None
        else:
            hardness_level = st.selectbox("Hardness level", HARDNESS_LEVELS, index=1)
            hardness_value = st.number_input("Hardness value", min_value=0.0, step=1.0, value=180.0)
            hardness_unit = st.selectbox("Hardness unit", HARDNESS_UNITS, index=0)

        workpiece_shape = st.selectbox("Workpiece shape", WORKPIECE_SHAPES, index=0)

    with col2:
        shape_fields = {
            "Round Bar": ["diameter_mm", "length_mm"],
            "Square Bar": ["width_mm", "length_mm"],
            "Plate": ["length_mm", "width_mm", "thickness_mm"],
            "Pipe": ["diameter_mm", "length_mm", "thickness_mm"],
            "Block": ["length_mm", "width_mm", "height_mm"],
            "Forged Part": ["length_mm", "width_mm", "thickness_mm"],
            "Casting": ["length_mm", "width_mm", "thickness_mm"],
        }
        field_data = {}
        for field in shape_fields.get(workpiece_shape, []):
            label = field.replace("_mm", " (mm)")
            field_data[field] = st.number_input(label, min_value=0.0, step=0.5, value=50.0)

    st.subheader("2. Machining Operation")
    operation_col, stage_col = st.columns(2)
    with operation_col:
        machining_operation = st.selectbox("Operation", OPERATIONS, index=0)
    with stage_col:
        machining_stage = st.selectbox("Machining stage", STAGES, index=0)

    st.subheader("3. Machine Details")
    machine_col, power_col, rpm_col = st.columns(3)
    with machine_col:
        machine_type = st.selectbox("Machine type", MACHINE_TYPES, index=1)
    with power_col:
        machine_power_kw = st.number_input("Machine power (kW)", min_value=0.0, step=0.5, value=15.0)
    with rpm_col:
        max_rpm = st.number_input("Maximum spindle speed (RPM)", min_value=0, step=100, value=6000)
    machine_rigidity = st.selectbox("Machine rigidity", MACHINE_RIGIDITY, index=1)

    st.subheader("4. Cutting Parameters")
    cut_col1, cut_col2, cut_col3 = st.columns(3)
    with cut_col1:
        feed_rate_value = st.number_input("Feed rate", min_value=0.0, step=0.01, value=0.25)
        feed_rate_unit = st.selectbox("Feed-rate unit", ["mm/rev", "mm/min"], index=0)
        spindle_speed_rpm = st.number_input("Spindle speed (RPM)", min_value=0, step=10, value=1200)
    with cut_col2:
        cutting_speed_m_min = st.number_input("Cutting speed (m/min)", min_value=0.0, step=1.0, value=120.0)
        depth_of_cut_mm = st.number_input("Depth of cut (mm)", min_value=0.0, step=0.1, value=2.0)
        width_of_cut_mm = st.number_input("Width of cut (mm)", min_value=0.0, step=0.1, value=10.0)
    with cut_col3:
        step_over_mm = st.number_input("Step-over (mm)", min_value=0.0, step=0.1, value=2.0)

    st.subheader("5. Surface Requirements")
    surf_col1, surf_col2, surf_col3 = st.columns(3)
    with surf_col1:
        surface_finish = st.selectbox("Required surface finish", SURFACE_FINISHES, index=1)
    with surf_col2:
        tolerance = st.selectbox("Tolerance", TOLERANCES, index=1)
    with surf_col3:
        accuracy_requirement = st.selectbox("Accuracy requirement", ACCURACY_REQUIREMENTS, index=1)

    st.subheader("6. User Preferences")
    preferences = st.multiselect(
        "Prioritize the following preferences:",
        PREFERENCES,
        default=["Longer tool life", "Higher productivity"],
    )

    submitted = st.form_submit_button("RECOMMEND TOOL")

if submitted:
    payload = {
        "material_category": material_category,
        "material_grade": material_grade,
        "hardness_level": hardness_level,
        "hardness_value": hardness_value,
        "hardness_unit": hardness_unit,
        "workpiece_shape": workpiece_shape,
        **field_data,
        "machining_operation": machining_operation,
        "machining_stage": machining_stage,
        "machine_type": machine_type,
        "machine_power_kw": machine_power_kw,
        "max_rpm": max_rpm,
        "machine_rigidity": machine_rigidity,
        "feed_rate_value": feed_rate_value,
        "feed_rate_unit": feed_rate_unit,
        "spindle_speed_rpm": spindle_speed_rpm,
        "cutting_speed_m_min": cutting_speed_m_min,
        "depth_of_cut_mm": depth_of_cut_mm,
        "width_of_cut_mm": width_of_cut_mm,
        "step_over_mm": step_over_mm,
        "surface_finish": surface_finish,
        "tolerance": tolerance,
        "accuracy_requirement": accuracy_requirement,
        "preferences": preferences,
    }

    errors, warnings = validate_input_form(payload)

    if errors:
        st.error("Input validation failed:")
        for error in errors:
            st.markdown(f"- {error}")
        st.info("Please correct the invalid values before continuing.")
    else:
        recommendation = recommend_tool(payload, catalog_override=catalog_override)

        if warnings:
            st.warning("Validation notes and safety considerations:")
            for warning in warnings:
                st.markdown(f"- {warning}")

        st.subheader("Recommendation")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**Recommended cutting-tool type:** {recommendation['recommended_tool_type']}")
            st.markdown(f"**ML model prediction:** {recommendation['ml_prediction'] if recommendation['ml_prediction'] else 'Not available'}")
            st.markdown(f"**Tool material / grade:** {recommendation['tool_material_or_grade']}")
            st.markdown(f"**Suitable machining operation:** {recommendation['suitable_operation']}")
            st.markdown(f"**Recommended cutting-speed range:** {recommendation['recommended_cutting_speed_range']}")
            st.markdown(f"**Recommended feed-rate range:** {recommendation['recommended_feed_rate_range']}")
            st.markdown(f"**Suggested depth-of-cut range:** {recommendation['suggested_depth_of_cut_range']}")
        with col_b:
            st.markdown(f"**Machine compatibility:** {recommendation['machine_compatibility']}")
            st.markdown(f"**Recommendation confidence:** {recommendation.get('confidence_label', 'Unknown')} ({recommendation.get('confidence_score', 0):.2f})")
            st.markdown(f"**Model confidence:** {recommendation.get('model_confidence', 0.0):.2f}")
            st.markdown(f"**Agreement status:** {recommendation.get('agreement_status', 'No status available')}")
            st.markdown(f"**Coolant recommendation:** {recommendation['coolant_recommendation']}")
            st.markdown(f"**Tool-life / productivity / cost trade-off:** {recommendation['tool_life_cost_trade_off']}")
            st.markdown(f"**Source type:** {recommendation['source_type']}")
            st.caption(f"{recommendation['data_source']}")

        st.markdown("### Engineering rule check")
        st.info(recommendation.get("material_rule_summary", "No material-specific rule summary available."))
        if recommendation.get("reference_catalog_used"):
            st.caption("Reference catalog in use: supplier/imported limit data is being applied.")
        if recommendation.get("rule_status"):
            st.caption(f"Rule status: {recommendation['rule_status']}")

        st.markdown("### Decision review")
        review_df = pd.DataFrame(
            {
                "Signal": [
                    "Model output",
                    "Engineering rule window",
                    "Reference catalog",
                    "Agreement status",
                    "Confidence",
                ],
                "Value": [
                    recommendation.get("ml_prediction") or "Not available",
                    recommendation.get("material_rule_summary") or "No material-specific rule summary available.",
                    "Uploaded supplier catalog" if recommendation.get("reference_catalog_used") else "Built-in synthetic rules only",
                    recommendation.get("agreement_status") or "No status available",
                    f"{recommendation.get('confidence_label', 'Unknown')} ({recommendation.get('confidence_score', 0):.2f})",
                ],
            }
        )
        st.dataframe(review_df, use_container_width=True, hide_index=True)

        st.markdown("### Recommendation comparison")
        speed_low, speed_high = _extract_numeric_range(recommendation["recommended_cutting_speed_range"])
        feed_low, feed_high = _extract_numeric_range(recommendation["recommended_feed_rate_range"])
        doc_low, doc_high = _extract_numeric_range(recommendation["suggested_depth_of_cut_range"])

        recommended_values = {
            "Cutting speed": (speed_low + speed_high) / 2 if speed_low and speed_high else 0,
            "Feed rate": (feed_low + feed_high) / 2 if feed_low and feed_high else 0,
            "Depth of cut": (doc_low + doc_high) / 2 if doc_low and doc_high else 0,
        }
        selected_values = {
            "Cutting speed": float(cutting_speed_m_min) if cutting_speed_m_min is not None else 0,
            "Feed rate": float(feed_rate_value) if feed_rate_value is not None else 0,
            "Depth of cut": float(depth_of_cut_mm) if depth_of_cut_mm is not None else 0,
        }

        comparison_fig = go.Figure()
        comparison_fig.add_trace(
            go.Bar(name="Selected input", x=list(selected_values.keys()), y=list(selected_values.values()), marker_color="#123b72")
        )
        comparison_fig.add_trace(
            go.Bar(name="Recommended target", x=list(recommended_values.keys()), y=list(recommended_values.values()), marker_color="#4a90e2")
        )
        comparison_fig.update_layout(
            barmode="group",
            title="Selected cutting values versus recommended target range",
            template="plotly_white",
            height=420,
            legend_title_text="Series",
        )
        st.plotly_chart(comparison_fig, use_container_width=True)

        st.markdown("### Why this recommendation was made")
        st.write(recommendation["explanation"])
        st.write(describe_preferences(preferences))

        report_text = build_recommendation_report(payload, recommendation)
        st.download_button(
            label="Download recommendation report",
            data=report_text,
            file_name="smart_machining_assistant_report.txt",
            mime="text/plain",
        )

        st.markdown("### Data quality and limitations")
        st.info(
            "This result is a synthetic engineering decision-support prototype. It is not a replacement for qualified machining judgment, manufacturer catalog data, or validated shop-floor process planning."
        )
