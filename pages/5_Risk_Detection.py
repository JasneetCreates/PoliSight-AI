"""
Page 5: Regional Risk Detection
Module 4: Multi-criteria performance gap analysis, anomaly detection, and regional vulnerability tiers.
"""

import streamlit as st
import pandas as pd
import numpy as np

from database.db_connection import (
    query_education_records,
    get_available_filters,
    get_latest_risk_analysis,
    save_risk_records_batch,
)
from modules.risk_detector import PolicyRiskDetector
from utils.helpers import (
    apply_custom_css,
    render_academic_disclaimer,
    render_kpi,
)
from utils.visualizations import plot_risk_distribution

apply_custom_css()

st.markdown('<div class="main-header">Module 4: Regional Risk Detection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Multi-criteria performance gap analysis, Isolation Forest anomaly scoring, and regional vulnerability tiers</div>',
    unsafe_allow_html=True,
)

render_academic_disclaimer()

records = query_education_records()
if records.empty:
    st.info("No education data in database. Ingest UDISE+ data on Page 3.")
    st.stop()

detector = PolicyRiskDetector()

# Controls
st.markdown("### Risk Detection Engine Configuration")
c_type1, c_type2 = st.columns([1, 2])
with c_type1:
    region_granularity = st.selectbox("Regional Level", options=["State-Level Analysis", "District-Level Analysis"])
    reg_type = "state" if "State" in region_granularity else "district"

with c_type2:
    st.caption("Benchmark thresholds calibrated against NEP 2020 objectives:")

col1, col2, col3, col4 = st.columns(4)
with col1:
    target_ger_threshold = st.slider("GER Benchmark (%)", 75.0, 100.0, 90.0)
with col2:
    dropout_threshold = st.slider("Secondary Dropout Alert (%)", 5.0, 25.0, 12.0)
with col3:
    ptr_alert_threshold = st.slider("PTR Statutory Ceiling", 20.0, 45.0, 30.0)
with col4:
    internet_threshold = st.slider("Internet Baseline (%)", 30.0, 90.0, 50.0)

# Run Risk Engine
with st.spinner("Computing regional performance gaps and Isolation Forest anomalies..."):
    risk_df = detector.evaluate_regional_risks(
        df=records,
        region_type=reg_type,
        target_ger=target_ger_threshold,
        dropout_ceiling=dropout_threshold,
        ptr_ceiling=ptr_alert_threshold,
        internet_benchmark=internet_threshold,
        save_to_db=False,
    )

if not risk_df.empty:
    col_k1, col_k2, col_k3 = st.columns(3)
    with col_k1:
        high_count = (risk_df["risk_level"] == "High Risk").sum()
        render_kpi("High Risk Regions", high_count, "Immediate policy priority", border_color="#EF4444")
    with col_k2:
        mod_count = (risk_df["risk_level"] == "Moderate Risk").sum()
        render_kpi("Moderate Risk Regions", mod_count, "Targeted intervention needed", border_color="#F59E0B")
    with col_k3:
        low_count = (risk_df["risk_level"] == "Low Risk").sum()
        render_kpi("Low Risk Regions", low_count, "On track with benchmarks", border_color="#10B981")

    # Chart & Filter Table
    c_chart, c_table = st.columns([1, 2])
    with c_chart:
        st.plotly_chart(plot_risk_distribution(risk_df), use_container_width=True)

    with c_table:
        st.markdown("#### Regional Vulnerability Roster")
        tier_filter = st.multiselect(
            "Filter Tiers",
            options=["High Risk", "Moderate Risk", "Low Risk"],
            default=["High Risk", "Moderate Risk"],
        )
        display_sub = risk_df[risk_df["risk_level"].isin(tier_filter)] if tier_filter else risk_df

        st.dataframe(
            display_sub[["region_name", "risk_level", "risk_score", "anomaly_score", "gap_summary", "explanation"]],
            use_container_width=True,
            hide_index=True,
        )

    # Save risk findings to DB button
    if st.button("💾 Persist Risk Assessment Findings to Database", type="primary"):
        save_risk_records_batch(risk_df.to_dict(orient="records"))
        st.success(f"Successfully logged {len(risk_df)} regional risk assessments to database!")
else:
    st.warning("No records available to evaluate.")
