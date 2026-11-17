"""
Page 1: Overview & Macro Policy State
"""

import streamlit as st
import pandas as pd
from database.db_connection import (
    get_all_policies,
    get_policy_objectives_with_indicators,
    get_available_filters,
    query_education_records,
)
from utils.helpers import (
    apply_custom_css,
    render_academic_disclaimer,
    render_kpi,
    format_percentage,
)
from utils.visualizations import (
    plot_indicator_trend_with_target,
    plot_state_comparison_bar,
)

apply_custom_css()

st.markdown('<div class="main-header">Policy & Data Landscape Overview</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Macro indicators, coverage statistics, and baseline policy configuration</div>',
    unsafe_allow_html=True,
)

render_academic_disclaimer()

# Fetch active policy and database coverage
policies = get_all_policies()
if not policies:
    st.warning("No policy documents found. Please visit Page 2 (Policy Understanding) to upload a document.")
    st.stop()

selected_policy_name = st.selectbox(
    "Selected Policy Document",
    options=[p["title"] for p in policies],
    index=0,
)
selected_policy = next(p for p in policies if p["title"] == selected_policy_name)
objectives = get_policy_objectives_with_indicators(selected_policy["id"])
filter_meta = get_available_filters()
records_df = query_education_records()

# Overview Statistics Cards
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    render_kpi("Policy", selected_policy.get("code", "NEP 2020"), f"Sector: {selected_policy.get('sector', 'School')}")
with col2:
    states_count = len(filter_meta.get("states", []))
    render_kpi("Available States", states_count, "State-level entities")
with col3:
    districts_count = len(filter_meta.get("districts", []))
    render_kpi("Available Districts", districts_count, "District administrative units")
with col4:
    y_min = filter_meta.get("min_year")
    y_max = filter_meta.get("max_year")
    y_str = f"{y_min} - {y_max}" if y_min and y_max else "N/A"
    render_kpi("Data Years", y_str, "Time-series horizon")
with col5:
    indicators_tracked = len({o["indicator_name"] for o in objectives if o.get("indicator_name")})
    render_kpi("Indicators Tracked", indicators_tracked, "Policy metrics")

st.markdown("---")

# Policy Summary Card
st.markdown("### Selected Policy Baseline")
with st.expander("📌 View Policy Description & Scope", expanded=True):
    st.markdown(f"**Policy Title:** {selected_policy['title']}")
    st.markdown(f"**Sector:** {selected_policy.get('sector', 'School Education')}")
    st.markdown(f"**Year of Enactment:** {selected_policy.get('publication_year', 2020)}")
    if selected_policy.get("summary"):
        st.info(selected_policy["summary"])

# Tracked Objectives & Indicators Table
st.markdown("### Core Objectives & Measurable Indicators")
if objectives:
    obj_table = []
    for o in objectives:
        obj_table.append({
            "Objective": o.get("objective_title"),
            "Indicator": o.get("indicator_name"),
            "Target Group": o.get("target_group"),
            "Target Value": f"{o.get('target_value', 'N/A')} {o.get('target_unit', '')}" if o.get("target_value") is not None else "Descriptive Target",
            "Target Year": o.get("target_year") or "N/A",
            "Section Ref": o.get("section_reference", "General"),
        })
    df_obj = pd.DataFrame(obj_table)
    st.dataframe(df_obj, use_container_width=True, hide_index=True)
else:
    st.info("No policy objectives mapped yet.")

st.markdown("---")

# Data Quality & Coverage Status
st.markdown("### Education Data Quality Status")
if not records_df.empty:
    q1, q2, q3 = st.columns(3)
    with q1:
        st.metric("Total Ingested Records", len(records_df))
    with q2:
        missing_count = records_df.isna().sum().sum()
        missing_pct = (missing_count / (records_df.shape[0] * records_df.shape[1])) * 100
        st.metric("Data Completeness", f"{100 - missing_pct:.1f}%", f"{missing_count} missing cells")
    with q3:
        st.metric("Official Primary Source", "UDISE+ & Ministry of Education", "Verified Public Data")

    # Sample visual preview
    st.markdown("#### National Trajectory Preview: Gross Enrolment Ratio (GER)")
    ger_annual = records_df.groupby("year")["ger"].mean().reset_index()
    fig = plot_indicator_trend_with_target(
        df=ger_annual,
        year_col="year",
        val_col="ger",
        target_val=100.0,
        target_year=2030,
        indicator_title="National Average GER",
        unit="%",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No education records loaded into database yet. Visit Page 3 (Data Collection) to ingest datasets.")
