"""
Page 4: Regional Risk Detection
Module 4: Multi-criteria performance gap analysis, Isolation Forest anomaly detection, and vulnerability tiers.
"""

import sys
from pathlib import Path

# Guarantee project root is in sys.path on Streamlit Cloud
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd

from database.db_connection import query_education_records
from modules.risk_detector import PolicyRiskDetector
from utils.helpers import (
    apply_custom_css,
    render_kpi,
    render_academic_disclaimer,
)
from utils.visualizations import plot_risk_distribution

apply_custom_css()

st.markdown('<div class="main-header">⚠️ Module 4: Regional Risk Detection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Identifies vulnerable states and districts lagging behind policy expectations using Isolation Forest anomaly detection.</div>',
    unsafe_allow_html=True,
)

records = query_education_records()
if records.empty:
    st.info("No education data in database. Please ingest UDISE+ data in Module 2.")
    st.stop()

detector = PolicyRiskDetector()

# Optional Benchmark Adjustments (Collapsed by default so it's not messy)
with st.expander("⚙️ Adjust Benchmark Alert Thresholds (Optional)", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        target_ger_threshold = st.slider("GER Benchmark (%)", 75.0, 100.0, 90.0)
    with c2:
        dropout_threshold = st.slider("Dropout Alert Ceiling (%)", 5.0, 25.0, 12.0)
    with c3:
        ptr_alert_threshold = st.slider("PTR Statutory Limit", 20.0, 45.0, 30.0)
    with c4:
        internet_threshold = st.slider("Internet Baseline (%)", 30.0, 90.0, 50.0)

# Run risk evaluation
risk_df = detector.evaluate_regional_risks(
    df=records,
    region_type="state",
    target_ger=target_ger_threshold if 'target_ger_threshold' in locals() else 90.0,
    dropout_ceiling=dropout_threshold if 'dropout_threshold' in locals() else 12.0,
    ptr_ceiling=ptr_alert_threshold if 'ptr_alert_threshold' in locals() else 30.0,
    internet_benchmark=internet_threshold if 'internet_threshold' in locals() else 50.0,
    save_to_db=False,
)

if not risk_df.empty:
    # KPI Row
    col_k1, col_k2, col_k3 = st.columns(3)
    with col_k1:
        high_cnt = (risk_df["risk_level"] == "High Risk").sum()
        render_kpi("High Risk States", f"{high_cnt} Regions", "Immediate policy priority", border_color="#DC2626")
    with col_k2:
        mod_cnt = (risk_df["risk_level"] == "Moderate Risk").sum()
        render_kpi("Moderate Risk States", f"{mod_cnt} Regions", "Targeted support needed", border_color="#D97706")
    with col_k3:
        low_cnt = (risk_df["risk_level"] == "Low Risk").sum()
        render_kpi("Low Risk States", f"{low_cnt} Regions", "On track with benchmarks", border_color="#059669")

    st.markdown("<br>", unsafe_allow_html=True)

    c_chart, c_table = st.columns([1, 1.8])
    with c_chart:
        fig_donut = plot_risk_distribution(risk_df)
        fig_donut.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_donut, use_container_width=True)

    with c_table:
        st.markdown("##### 📋 Regional Vulnerability Roster")
        tier_choice = st.radio("Filter Tier", ["All", "High Risk", "Moderate Risk", "Low Risk"], horizontal=True)
        sub_df = risk_df if tier_choice == "All" else risk_df[risk_df["risk_level"] == tier_choice]

        disp_table = sub_df[["region_name", "risk_level", "risk_score", "explanation"]].rename(columns={
            "region_name": "State",
            "risk_level": "Risk Tier",
            "risk_score": "Score (0-100)",
            "explanation": "Primary Identified Alerts",
        })
        st.dataframe(disp_table, use_container_width=True, hide_index=True)

render_academic_disclaimer()
