"""
Page 7: Report Generation & Exports
Allows users to download comprehensive PDF evaluation dossiers and CSV analytical datasets.
"""

import streamlit as st
import pandas as pd
from database.db_connection import (
    get_all_policies,
    get_policy_objectives_with_indicators,
    query_education_records,
    get_available_filters,
    get_latest_risk_analysis,
    get_all_recommendations,
)
from modules.report_generator import PolicyReportGenerator
from utils.helpers import apply_custom_css, render_academic_disclaimer

apply_custom_css()

st.markdown('<div class="main-header">Policy Analysis Reports & Export</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Download formal evaluation dossiers, executive summaries, and empirical datasets</div>',
    unsafe_allow_html=True,
)

render_academic_disclaimer()

policies = get_all_policies()
if not policies:
    st.warning("No policies registered yet.")
    st.stop()

active_policy = policies[0]
objectives = get_policy_objectives_with_indicators(active_policy["id"])
records = query_education_records()
filter_meta = get_available_filters()
risk_df = get_latest_risk_analysis()
recs_df = get_all_recommendations()

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📄 Comprehensive PDF Policy Dossier")
    st.write(
        "Generate a publication-grade PDF report containing policy framework objectives, statutory targets, "
        "data quality audits, regional risk rankings, and prioritized recommendations."
    )

    report_gen = PolicyReportGenerator()
    
    # Prepare data for PDF
    risk_list = risk_df.to_dict(orient="records") if not risk_df.empty else []
    rec_list = recs_df.to_dict(orient="records") if not recs_df.empty else []

    try:
        pdf_bytes = report_gen.generate_pdf_report(
            policy_meta=active_policy,
            objectives=objectives,
            data_summary=filter_meta,
            risk_summary=risk_list,
            recommendations=rec_list,
        )

        st.download_button(
            label="📥 Download Policy Evaluation PDF Report",
            data=pdf_bytes,
            file_name=f"Policy_Evaluation_Report_{active_policy.get('code', 'NEP2020')}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True,
        )
    except Exception as e:
        st.error(f"Error generating PDF dossier: {e}")

with col2:
    st.markdown("### 📊 Raw & Analytical CSV Exports")
    st.write("Export structured tables for academic econometric analysis and reproducibility.")

    # Export records
    if not records.empty:
        csv_records = records.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Education Records (CSV)",
            data=csv_records,
            file_name="education_records_udise.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Export objectives
    if objectives:
        df_obj = pd.DataFrame(objectives)
        csv_obj = df_obj.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Extracted Policy Objectives (CSV)",
            data=csv_obj,
            file_name="policy_objectives_extracted.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Export recommendations
    if not recs_df.empty:
        csv_recs = recs_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Policy Recommendations (CSV)",
            data=csv_recs,
            file_name="policy_recommendations.csv",
            mime="text/csv",
            use_container_width=True,
        )
