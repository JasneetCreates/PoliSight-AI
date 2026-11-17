"""
Page 6: Recommendation Engine
Module 5: Generates evidence-based, prioritized policy recommendations linked to empirical gaps.
"""

import streamlit as st
import pandas as pd

from database.db_connection import (
    get_all_recommendations,
    save_recommendations_batch,
    query_education_records,
)
from modules.risk_detector import PolicyRiskDetector
from modules.recommendation_engine import PolicyRecommendationEngine
from utils.helpers import (
    apply_custom_css,
    render_academic_disclaimer,
    render_kpi,
)

apply_custom_css()

st.markdown('<div class="main-header">Module 5: Evidence-Based Recommendations</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Data-driven policy interventions, priority assignment, and rigorous diagnostic investigation pathways</div>',
    unsafe_allow_html=True,
)

render_academic_disclaimer()

records = query_education_records()
if records.empty:
    st.info("No education records in database. Please ingest UDISE+ data on Page 3.")
    st.stop()

detector = PolicyRiskDetector()
rec_engine = PolicyRecommendationEngine()

# First compute risk analysis to generate fresh evidence-based recommendations
with st.spinner("Synthesizing empirical findings and generating recommendations..."):
    risk_df = detector.evaluate_regional_risks(df=records, region_type="state", save_to_db=False)
    generated_recs = rec_engine.generate_recommendations(risk_df=risk_df, education_df=records, save_to_db=False)

df_recs = pd.DataFrame(generated_recs)

# KPI Summary
c1, c2, c3, c4 = st.columns(4)
crit_cnt = len(df_recs[df_recs["priority_level"] == "Critical"]) if not df_recs.empty else 0
high_cnt = len(df_recs[df_recs["priority_level"] == "High"]) if not df_recs.empty else 0
med_cnt = len(df_recs[df_recs["priority_level"] == "Medium"]) if not df_recs.empty else 0
tot_cnt = len(df_recs)

c1.metric("Critical Action Items", crit_cnt)
c2.metric("High Priority Guidance", high_cnt)
c3.metric("Medium Priority Upgrades", med_cnt)
c4.metric("Total Interventions", tot_cnt)

st.markdown("---")

# Priority Filter
c_f1, c_f2 = st.columns([1, 2])
with c_f1:
    filter_prio = st.selectbox("Filter Recommendations by Priority", options=["All", "Critical", "High", "Medium", "Low"])
with c_f2:
    filter_region = st.selectbox("Filter by Affected Region", options=["All Regions"] + sorted(df_recs["affected_region"].unique().tolist()) if not df_recs.empty else ["All Regions"])

display_df = df_recs.copy()
if not display_df.empty:
    if filter_prio != "All":
        display_df = display_df[display_df["priority_level"] == filter_prio]
    if filter_region != "All Regions":
        display_df = display_df[display_df["affected_region"] == filter_region]

st.markdown(f"### Prioritized Policy Action Dossier ({len(display_df)} Recommendations)")

for _, rec in display_df.iterrows():
    prio = rec["priority_level"]
    badge_cls = "risk-badge-high" if prio in ["Critical", "High"] else "risk-badge-mod" if prio == "Medium" else "risk-badge-low"

    with st.expander(f"📌 [{prio}] {rec['title']} - {rec['affected_region']}", expanded=(prio == "Critical")):
        st.markdown(f"<span class='{badge_cls}'>{prio} Priority</span> | **Indicator:** `{rec['related_indicator']}`", unsafe_allow_html=True)
        st.markdown(f"**Identified Policy Gap:** {rec['identified_problem']}")
        st.markdown(f"**Empirical Baseline Evidence:** {rec['supporting_data']}")
        st.markdown(f"**Policy Rationale:** {rec['reason']}")
        st.info(f"🔍 **Recommended Diagnostic Investigation:**\n{rec['suggested_investigation']}")

if st.button("💾 Persist Recommendations to System Database", type="primary"):
    save_recommendations_batch(generated_recs)
    st.success(f"Successfully saved {len(generated_recs)} prioritized recommendations to database!")
