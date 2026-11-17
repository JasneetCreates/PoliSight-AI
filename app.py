"""
Policy-to-Impact Intelligence System (PoliSight-AI)
An NLP and Machine Learning-Based Framework for Education Policy Evaluation and Decision Support in India.
Main Application Entrypoint & Executive Overview.
"""

from pathlib import Path
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="PoliSight-AI | Education Policy Intelligence",
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
    query_education_records,
)
from modules.policy_extractor import PolicyExtractor
from modules.nlp_processor import PolicyNLPProcessor
from modules.data_collector import EducationDataCollector
from utils.helpers import (
    apply_custom_css,
    render_kpi,
    render_academic_disclaimer,
)
from utils.visualizations import plot_indicator_trend_with_target

# Apply minimalist, clean UI styling
apply_custom_css()

# Initialize Database and Seed Baseline if empty
init_db()


def auto_seed_baseline():
    """Ensure sample policy and benchmark records exist so the app is instantly usable."""
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

    records = query_education_records()
    if records.empty:
        collector = EducationDataCollector()
        collector.generate_and_save_udise_sample()
        sample_path = Path(__file__).parent / "data" / "sample" / "udise_education_dataset.csv"
        if sample_path.exists():
            df_sample = pd.read_csv(sample_path)
            collector.ingest_and_store(
                df=df_sample,
                source_name="UDISE+ Benchmark Sample",
                source_type="UDISE+",
            )


auto_seed_baseline()

# Sidebar Navigation Header
st.sidebar.markdown("### 🎓 PoliSight-AI")
st.sidebar.caption("Education Policy Intelligence Framework")
st.sidebar.markdown("---")

# Header Section
st.markdown('<div class="main-header">Policy-to-Impact Intelligence System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">An NLP and Machine Learning framework connecting National Education Policy (NEP 2020) goals with real-world UDISE+ school data.</div>',
    unsafe_allow_html=True,
)

# Fetch current metadata for KPIs
policies = get_all_policies()
active_policy = policies[0] if policies else {"title": "NEP 2020", "publication_year": 2020}
objectives = get_policy_objectives_with_indicators(active_policy.get("id", 1))
filter_meta = get_available_filters()
records_df = query_education_records()

# Clean KPI Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    render_kpi("Active Policy", "NEP 2020", "School Education (Govt. of India)", border_color="#2563EB")
with col2:
    total_obj = len({o["objective_id"] for o in objectives})
    render_kpi("Tracked Goals", f"{total_obj} Objectives", "NLP-extracted targets", border_color="#059669")
with col3:
    states_count = len(filter_meta.get("states", []))
    render_kpi("Geographic Coverage", f"{states_count} States", "19 States • 59 Districts", border_color="#7C3AED")
with col4:
    y_str = f"{filter_meta.get('min_year', 2018)} - {filter_meta.get('max_year', 2024)}"
    render_kpi("Time Horizon", y_str, f"{filter_meta.get('total_records', 0)} UDISE+ records", border_color="#D97706")

st.markdown("<br>", unsafe_allow_html=True)

# Main Grid: National Indicator Trajectory
c_chart, c_guide = st.columns([1.6, 1])

with c_chart:
    st.markdown("##### 📈 National Trajectory: Gross Enrolment Ratio (GER)")
    if not records_df.empty:
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
        fig.update_layout(height=340, margin=dict(l=30, r=20, t=30, b=30))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Loading baseline education records...")

with c_guide:
    st.markdown("##### 🧭 How the System Works")
    st.markdown(
        """
        1. **📄 Policy Understanding (NLP):** Reads official policy documents and identifies targets (*100% GER by 2030, PTR under 30:1*).
        2. **📊 Data Preprocessing:** Standardizes school data (UDISE+) across all states and districts.
        3. **📈 Impact Analysis (ML):** Uses Regression and Clustering to predict progress toward 2030.
        4. **⚠️ Risk Detection:** Identifies which regions have large performance gaps using Isolation Forest.
        5. **💡 Recommendations:** Generates actionable, prioritized policy guidance for officials.
        """
    )
    st.caption("👈 **Use the left sidebar** to navigate through each module step-by-step.")

# Subtle academic methodology note at bottom
render_academic_disclaimer()
