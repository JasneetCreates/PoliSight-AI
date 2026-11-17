"""
Page 3: Data Collection & Preprocessing
Module 2: Ingests, validates, cleans, and standardizes public education datasets (UDISE+).
"""

import io
import json
from pathlib import Path
import streamlit as st
import pandas as pd

from database.db_connection import (
    register_data_source,
    insert_education_dataset,
    insert_education_records_batch,
    query_education_records,
    get_available_filters,
)
from modules.data_collector import EducationDataCollector
from modules.data_preprocessor import EducationDataPreprocessor
from utils.helpers import (
    apply_custom_css,
    render_academic_disclaimer,
    render_kpi,
)

apply_custom_css()

st.markdown('<div class="main-header">Module 2: Data Collection & Preprocessing</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">UDISE+ ingestion, data hygiene, missing value audits, outlier detection, and standard indicator normalization</div>',
    unsafe_allow_html=True,
)

render_academic_disclaimer()

collector = EducationDataCollector()
preprocessor = EducationDataPreprocessor()

# Session state for current ingestion
if "current_quality_report" not in st.session_state:
    st.session_state.current_quality_report = None
if "current_cleaned_df" not in st.session_state:
    st.session_state.current_cleaned_df = None

tab_upload, tab_quality, tab_view = st.tabs(["📥 Dataset Ingestion", "📋 Data Quality Report", "📊 Current Dataset Records"])

# -------------------------------------------------------------
# TAB 1: Ingestion
# -------------------------------------------------------------
with tab_upload:
    st.markdown("### Education Dataset Ingestion")
    st.write(
        "Upload district or state-level education datasets in **CSV** or **Excel (.xlsx)** format, "
        "or load official UDISE+ multi-year benchmark records."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader("Upload Education Dataset (CSV or Excel)", type=["csv", "xlsx", "xls"])
    with col2:
        st.markdown("**Quick Preset (UDISE+):**")
        load_sample_btn = st.button("📊 Load Official UDISE+ Benchmark Dataset (2018-2024)", use_container_width=True)

    df_raw = None
    filename_raw = ""

    if uploaded_file is not None:
        try:
            df_raw = collector.parse_file(io.BytesIO(uploaded_file.read()), uploaded_file.name)
            filename_raw = uploaded_file.name
            st.success(f"Parsed `{filename_raw}` with {len(df_raw)} records and {len(df_raw.columns)} columns.")
        except Exception as e:
            st.error(f"Error reading file: {e}")
    elif load_sample_btn:
        sample_path = Path(__file__).resolve().parent.parent / "data" / "sample" / "udise_education_dataset.csv"
        if not sample_path.exists():
            collector.generate_and_save_udise_sample()
        df_raw = pd.read_csv(sample_path)
        filename_raw = "udise_education_dataset.csv"
        st.success(f"Loaded built-in UDISE+ benchmark records ({len(df_raw)} records across {df_raw['state'].nunique()} states).")

    if df_raw is not None:
        st.markdown("#### Raw Data Preview:")
        st.dataframe(df_raw.head(5), use_container_width=True)

        if st.button("🚀 Run Preprocessing Pipeline & Validate Data", type="primary"):
            with st.spinner("Standardizing column names, normalizing years/states, and auditing data hygiene..."):
                cleaned_df, quality_report = preprocessor.preprocess_dataset(df_raw, source_name=filename_raw)
                st.session_state.current_cleaned_df = cleaned_df
                st.session_state.current_quality_report = quality_report
                st.success("Preprocessing and validation complete! Review the Quality Report tab below.")

    curr_df = st.session_state.current_cleaned_df
    curr_report = st.session_state.current_quality_report

    if curr_df is not None and curr_report is not None:
        st.markdown("---")
        st.markdown("### Preprocessing Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Clean Records", curr_report["total_records"])
        c2.metric("Duplicates Removed", curr_report["duplicate_records_removed"])
        c3.metric("Completeness Index", f"{curr_report['completeness_score_pct']}%")
        c4.metric("Temporal Span", f"{curr_report['years_range'][0]} - {curr_report['years_range'][1]}" if curr_report['years_range'] else "N/A")

        if st.button("💾 Ingest Cleaned Records into Education Database", type="primary"):
            with st.spinner("Storing records into SQLite database..."):
                dataset_id, _ = collector.ingest_and_store(
                    df=curr_df,
                    source_name=curr_report.get("source_name", "UDISE+"),
                    source_type="UDISE+",
                    dataset_name=f"Ingested_{curr_report.get('source_name', 'Dataset')}",
                )
                st.success(f"Successfully committed {len(curr_df)} records (Dataset ID: {dataset_id}) to the database!")
                st.rerun()

# -------------------------------------------------------------
# TAB 2: Data Quality Report
# -------------------------------------------------------------
with tab_quality:
    st.markdown("### Data Quality & Integrity Audit")
    records = query_education_records()
    report = st.session_state.current_quality_report

    if report is not None:
        st.markdown(f"**Source Document:** `{report['source_name']}` | **Evaluated At:** {report['last_evaluated']}")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Rows Evaluated", report["total_records"])
        col2.metric("Completeness Score", f"{report['completeness_score_pct']}%")
        col3.metric("States Covered", report["states_count"])
        col4.metric("Districts Covered", report["districts_count"])

        st.markdown("#### Missing Cell Breakdown by Indicator:")
        if report.get("missing_by_indicator"):
            miss_df = pd.DataFrame([
                {"Indicator": k, "Missing Count": v, "Missing %": round((v / report["total_records"]) * 100, 2)}
                for k, v in report["missing_by_indicator"].items()
            ]).sort_values(by="Missing Count", ascending=False)
            st.dataframe(miss_df, use_container_width=True, hide_index=True)
        else:
            st.success("Zero missing cells detected in primary tracked indicators.")

        st.markdown("#### Statistical Outlier Review (Non-Destructive Tagging):")
        outliers = report.get("outlier_analysis", {})
        if outliers:
            out_rows = []
            for col, o_info in outliers.items():
                out_rows.append({
                    "Indicator": col,
                    "Outlier Count": o_info["count"],
                    "Lower Bound": o_info["lower_bound"],
                    "Upper Bound": o_info["upper_bound"],
                    "Sample Regions": ", ".join(o_info["sample_regions"][:3]),
                })
            st.dataframe(pd.DataFrame(out_rows), use_container_width=True, hide_index=True)
            st.caption("ℹ️ Note: Meaningful regional outliers are flagged for research review, not deleted.")
        else:
            st.info("No extreme statistical anomalies detected outside 1.5x IQR boundaries.")
    elif not records.empty:
        total_rows = len(records)
        missing_counts = records.isna().sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records in DB", total_rows)
        col2.metric("Completeness Index", f"{(1 - (missing_counts.sum() / records.size)) * 100:.1f}%")
        col3.metric("Tracked Indicators", len(records.columns) - 4)

        st.markdown("#### Database Missingness Summary:")
        miss_df = pd.DataFrame({
            "Indicator": missing_counts.index,
            "Missing Records": missing_counts.values,
            "Missing Percentage (%)": (missing_counts.values / total_rows) * 100
        }).sort_values(by="Missing Records", ascending=False)
        st.dataframe(miss_df, use_container_width=True, hide_index=True)
    else:
        st.info("No quality report available. Run preprocessing in the Ingestion tab.")

# -------------------------------------------------------------
# TAB 3: Database Explorer
# -------------------------------------------------------------
with tab_view:
    st.markdown("### Education Database Records")
    filter_meta = get_available_filters()
    
    if filter_meta.get("total_records", 0) > 0:
        c1, c2, c3 = st.columns(3)
        with c1:
            state_opts = ["All States"] + filter_meta.get("states", [])
            sel_state = st.selectbox("State Filter", options=state_opts)
        with c2:
            dist_opts = ["All Districts"] + filter_meta.get("districts", [])
            sel_dist = st.selectbox("District Filter", options=dist_opts)
        with c3:
            y_min = filter_meta.get("min_year") or 2018
            y_max = filter_meta.get("max_year") or 2024
            sel_years = st.slider("Academic Year Horizon", min_value=int(y_min), max_value=int(y_max), value=(int(y_min), int(y_max)))

        filtered_df = query_education_records(
            state=sel_state if sel_state != "All States" else None,
            district=sel_dist if sel_dist != "All Districts" else None,
            year_start=sel_years[0],
            year_end=sel_years[1],
        )
        st.markdown(f"**Showing {len(filtered_df)} records:**")
        st.dataframe(filtered_df, use_container_width=True, hide_index=True)
    else:
        st.info("Database is currently empty. Ingest data in Tab 1 to populate records.")
