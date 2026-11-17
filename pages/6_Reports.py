"""
Page 6: Report Generation & Exports
Module 6: Allows users to download comprehensive PDF evaluation dossiers and CSV analytical datasets.
"""

import streamlit as st
import pandas as pd

from database.db_connection import (
    get_all_policies,
    get_policy_objectives_with_indicators,
    query_education_records,
    get_available_filters,
)
from modules.risk_detector import PolicyRiskDetector
from modules.recommendation_engine import PolicyRecommendationEngine
from modules.report_generator import PolicyReportGenerator
from utils.helpers import (
    apply_custom_css,
    render_academic_disclaimer,
)

apply_custom_css()

st.markdown('<div class="main-header">📑 Module 6: Reports & Exports</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Download formal evaluation dossiers for policymakers and export raw reproducible datasets for research.</div>',
    unsafe_allow_html=True,
)

policies = get_all_policies()
active_policy = policies[0] if policies else {"title": "NEP 2020", "sector": "School Education", "publication_year": 2020}
objectives = get_policy_objectives_with_indicators(active_policy.get("id", 1))
records = query_education_records()
filter_meta = get_available_filters()

detector = PolicyRiskDetector()
rec_engine = PolicyRecommendationEngine()

risk_df = detector.evaluate_regional_risks(df=records, region_type="state", save_to_db=False)
recs = rec_engine.generate_recommendations(risk_df=risk_df, education_df=records, save_to_db=False)

c_pdf, c_csv = st.columns(2)

with c_pdf:
    st.markdown("##### 📄 Formal Evaluation Dossier (PDF)")
    st.write(
        "Generates a publication-grade, 2-page academic evaluation dossier formatted with "
        "ReportLab, including statutory targets, data quality audits, regional risk rankings, and diagnostic pathways."
    )

    report_gen = PolicyReportGenerator()
    try:
        pdf_bytes = report_gen.generate_pdf_report(
            policy_meta=active_policy,
            objectives=objectives,
            data_summary=filter_meta,
            risk_summary=risk_df.to_dict(orient="records"),
            recommendations=recs,
        )

        st.download_button(
            label="📥 Download Policy Evaluation Dossier (PDF)",
            data=pdf_bytes,
            file_name=f"Policy_Evaluation_Report_{active_policy.get('code', 'NEP2020')}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True,
        )
        st.success("PDF Dossier compiled and ready for download!")
    except Exception as e:
        st.error(f"Error compiling PDF: {e}")

with c_csv:
    st.markdown("##### 📊 Reproducible Research Datasets (CSV)")
    st.write("Export structured tables for peer review, econometric analysis, and reproducibility.")

    if not records.empty:
        csv_records = records.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Cleaned Education Records (CSV)",
            data=csv_records,
            file_name="education_records_udise.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if objectives:
        csv_obj = pd.DataFrame(objectives).to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Extracted Policy Objectives (CSV)",
            data=csv_obj,
            file_name="policy_objectives_extracted.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if recs:
        csv_recs = pd.DataFrame(recs).to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Prioritized Recommendations (CSV)",
            data=csv_recs,
            file_name="policy_recommendations.csv",
            mime="text/csv",
            use_container_width=True,
        )

render_academic_disclaimer()
