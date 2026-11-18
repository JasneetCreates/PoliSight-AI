"""
Page 5: Recommendation Engine
Module 5: Generates evidence-based, prioritized policy recommendations linked to empirical gaps.
"""

import streamlit as st
import pandas as pd

from database.db_connection import query_education_records
from modules.risk_detector import PolicyRiskDetector
from modules.recommendation_engine import PolicyRecommendationEngine
from utils.helpers import (
    apply_custom_css,
    render_kpi,
    render_academic_disclaimer,
)

apply_custom_css()

st.markdown('<div class="main-header">💡 Module 5: Evidence-Based Recommendations</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Data-driven policy interventions and targeted diagnostic investigation pathways for flagged regions.</div>',
    unsafe_allow_html=True,
)

records = query_education_records()
if records.empty:
    st.info("No education records in database. Please ingest UDISE+ data in Module 2.")
    st.stop()

detector = PolicyRiskDetector()
rec_engine = PolicyRecommendationEngine()

risk_df = detector.evaluate_regional_risks(df=records, region_type="state", save_to_db=False)
generated_recs = rec_engine.generate_recommendations(risk_df=risk_df, education_df=records, save_to_db=False)
df_recs = pd.DataFrame(generated_recs)

# KPI Row
c1, c2, c3, c4 = st.columns(4)
crit_cnt = len(df_recs[df_recs["priority_level"] == "Critical"]) if not df_recs.empty else 0
high_cnt = len(df_recs[df_recs["priority_level"] == "High"]) if not df_recs.empty else 0
med_cnt = len(df_recs[df_recs["priority_level"] == "Medium"]) if not df_recs.empty else 0

with c1:
    render_kpi("Critical Interventions", f"{crit_cnt} Items", "Immediate priority", border_color="#DC2626")
with c2:
    render_kpi("High Priority Guidance", f"{high_cnt} Items", "Urgent reform needed", border_color="#D97706")
with c3:
    render_kpi("Medium Priority", f"{med_cnt} Items", "Infrastructure upgrades", border_color="#2563EB")
with c4:
    render_kpi("Total Action Points", f"{len(df_recs)} Actions", "Evidence-based", border_color="#059669")

st.markdown("<br>", unsafe_allow_html=True)

# Clean Filter Row
f1, f2 = st.columns([1, 2])
with f1:
    filter_prio = st.selectbox("Filter by Priority", options=["All", "Critical", "High", "Medium"])
with f2:
    states_list = sorted(df_recs["affected_region"].unique().tolist()) if not df_recs.empty else []
    filter_state = st.selectbox("Filter by State", options=["All States"] + states_list)

display_recs = df_recs.copy()
if not display_recs.empty:
    if filter_prio != "All":
        display_recs = display_recs[display_recs["priority_level"] == filter_prio]
    if filter_state != "All States":
        display_recs = display_recs[display_recs["affected_region"] == filter_state]

st.markdown(f"##### Showing {len(display_recs)} Prioritized Policy Guidance Items")

badge_colors = {
    "Critical": "badge-red",
    "High": "badge-amber",
    "Medium": "badge-blue",
}

for _, rec in display_recs.iterrows():
    prio = rec["priority_level"]
    cls_name = badge_colors.get(prio, "badge-blue")

    with st.expander(f"📌 [{prio}] {rec['title']} — {rec['affected_region']}", expanded=(prio == "Critical")):
        st.markdown(f"<span class='badge-pill {cls_name}'>{prio} Priority</span> **Indicator:** `{rec['related_indicator']}`", unsafe_allow_html=True)
        st.markdown(f"**Identified Gap:** {rec['identified_problem']}")
        st.markdown(f"**Empirical Evidence:** `{rec['supporting_data']}`")
        st.markdown(f"**Policy Rationale:** {rec['reason']}")
        st.info(f"🔍 **Recommended Diagnostic Investigation:**\n{rec['suggested_investigation']}")

render_academic_disclaimer()
