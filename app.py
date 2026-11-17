"""
Policy-to-Impact Intelligence System
An NLP and Machine Learning-Based Framework for Education Policy Evaluation and Decision Support in India.
Main Streamlit Application Entrypoint.
"""

import os
from pathlib import Path
import streamlit as st

# Configure Streamlit page settings
st.set_page_config(
    page_title="Policy-to-Impact Intelligence System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

from database.db_connection import (
    init_db,
    get_all_policies,
    insert_policy,
    insert_policy_objective,
    insert_policy_indicator,
    get_policy_objectives_with_indicators,
    get_available_filters,
)
from utils.helpers import (
    apply_custom_css,
    render_academic_disclaimer,
    render_kpi,
)
from modules.policy_extractor import PolicyExtractor
from modules.nlp_processor import PolicyNLPProcessor

# Apply custom academic styling
apply_custom_css()

# Initialize Database
try:
    init_db()
except Exception as e:
    st.error(f"Database initialization error: {e}")


def seed_nep_2020_if_needed():
    """Ensure NEP 2020 School Education baseline exists in the database on initial startup."""
    policies = get_all_policies()
    if not any(p["code"] == "NEP_2020" for p in policies):
        nep_file = Path(__file__).parent / "data" / "policies" / "NEP_2020_School_Education.txt"
        if nep_file.exists():
            extractor = PolicyExtractor()
            nlp = PolicyNLPProcessor()

            with open(nep_file, "r", encoding="utf-8") as f:
                content = f.read()

            extracted = extractor.extract_from_file(nep_file, "NEP_2020_School_Education.txt")
            analysis = nlp.analyze_policy_document(extracted)

            policy_id = insert_policy(
                title="National Education Policy 2020",
                code="NEP_2020",
                sector=analysis.get("sector", "School Education"),
                publication_year=2020,
                document_name="NEP_2020_School_Education.txt",
                document_path=str(nep_file),
                document_text=content,
                summary=analysis.get("summary", ""),
            )

            # Insert parsed objectives & indicators
            for obj in analysis.get("objectives", []):
                obj_id = insert_policy_objective(
                    policy_id=policy_id,
                    title=obj.get("objective", "Policy Objective"),
                    section_reference=obj.get("section_reference", ""),
                    page_number=obj.get("page_number", 1),
                    target_group=obj.get("target_group", ""),
                    expected_outcome=obj.get("expected_outcome", ""),
                    timeline_year=obj.get("target_year"),
                    status="approved",
                )
                insert_policy_indicator(
                    objective_id=obj_id,
                    indicator_name=obj.get("indicator", ""),
                    indicator_code=obj.get("indicator_code", ""),
                    target_value=obj.get("target_value"),
                    target_unit=obj.get("target_unit", "%"),
                    target_year=obj.get("target_year"),
                    is_numerical_target=obj.get("is_numerical_target", 0),
                    source_reference=obj.get("source", ""),
                )


# Seed initial baseline
seed_nep_2020_if_needed()

# Sidebar Navigation Header
st.sidebar.title("🎓 Policy-to-Impact")
st.sidebar.caption("AI/ML Education Policy Evaluation System")
st.sidebar.markdown("---")

# Fetch current metadata for KPIs
policies = get_all_policies()
active_policy = policies[0] if policies else None
objectives = get_policy_objectives_with_indicators(active_policy["id"]) if active_policy else []
filter_meta = get_available_filters()

# Main Header
st.markdown('<div class="main-header">Policy-to-Impact Intelligence System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">An NLP and Machine Learning-Based Framework for Education Policy Evaluation and Decision Support in India</div>',
    unsafe_allow_html=True,
)

# Academic disclaimer
render_academic_disclaimer()

# Overview KPI row
col1, col2, col3, col4 = st.columns(4)

with col1:
    p_title = active_policy["title"] if active_policy else "None"
    render_kpi("Active Policy Document", p_title, f"Published: {active_policy.get('publication_year', 2020) if active_policy else 'N/A'}")

with col2:
    total_obj = len({o["objective_id"] for o in objectives})
    render_kpi("Extracted Objectives", total_obj, "NLP-parsed goals")

with col3:
    num_targets = sum(1 for o in objectives if o.get("is_numerical_target") == 1)
    render_kpi("Numerical Policy Targets", num_targets, "Explicit statutory targets")

with col4:
    tot_recs = filter_meta.get("total_records", 0)
    render_kpi("Education Records in DB", tot_recs, "UDISE+ & official metrics")

st.markdown("### System Architecture & Analytical Workflow")

# Workflow Architecture diagram
st.markdown(
    """
```mermaid
flowchart LR
    A[Policy Document<br/>PDF/DOCX/TXT] --> B[Module 1: NLP Policy<br/>Understanding]
    B --> C[Module 2: Data Collection<br/>& Preprocessing]
    C --> D[Module 3: Impact Analysis<br/>Regression/Clustering]
    D --> E[Module 4: Regional<br/>Risk Detection]
    E --> F[Module 5: Recommendation<br/>Engine]
    F --> G[Interactive Dashboard<br/>& Policy Reports]
```
    """
)

st.markdown("---")

# Feature Highlights Grid
st.markdown("### Framework Modules & Capabilities")
c1, c2, c3 = st.columns(3)

with c1:
    st.info("#### 1. Policy Understanding (NLP)")
    st.write(
        "Upload Indian policy documents (NEP 2020). Automatically extracts objectives, target populations, "
        "measurable indicators, and explicit numerical targets while preserving provenance."
    )
    if st.button("Go to Policy Understanding", key="btn_nav_policy"):
        st.switch_page("pages/2_Policy_Understanding.py")

with c2:
    st.info("#### 2. Data Collection & Preprocessing")
    st.write(
        "Integrates official UDISE+ and Ministry of Education datasets. Handles missing values, standardizes "
        "state/district names, validates data integrity, and generates data quality reports."
    )
    if st.button("Go to Data Collection", key="btn_nav_data"):
        st.switch_page("pages/3_Data_Collection.py")

with c3:
    st.info("#### 3. Impact Analysis (ML)")
    st.write(
        "Linear Regression trends, Random Forest association analysis, K-Means regional clustering, "
        "and time-series forecasting against policy target milestones."
    )
    if st.button("Go to Impact Analysis", key="btn_nav_impact"):
        st.switch_page("pages/4_Impact_Analysis.py")

c4, c5, c6 = st.columns(3)

with c4:
    st.info("#### 4. Regional Risk Detection")
    st.write(
        "Multi-criteria anomaly detection (Isolation Forest) and target-gap scoring to classify states and "
        "districts into High, Moderate, and Low risk tiers with explainability."
    )
    if st.button("Go to Risk Detection", key="btn_nav_risk"):
        st.switch_page("pages/5_Risk_Detection.py")

with c5:
    st.info("#### 5. Evidence-Based Recommendations")
    st.write(
        "Generates actionable, prioritized policy guidance for flagged regions. Explains underlying data "
        "correlations without unfounded causal claims."
    )
    if st.button("Go to Recommendations", key="btn_nav_rec"):
        st.switch_page("pages/6_Recommendations.py")

with c6:
    st.info("#### 6. Downloadable Reports")
    st.write(
        "Export comprehensive policy evaluation dossiers in PDF format (ReportLab) and analytical datasets "
        "in CSV format for academic publication and governance review."
    )
    if st.button("Go to Reports", key="btn_nav_rep"):
        st.switch_page("pages/7_Reports.py")
