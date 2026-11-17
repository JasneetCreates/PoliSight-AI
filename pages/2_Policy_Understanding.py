"""
Page 2: Policy Understanding (NLP Extraction & Review)
Processes policy documents (PDF, DOCX, TXT), extracts objectives, indicators, and targets,
and enables user review and database persistence.
"""

import io
import json
from pathlib import Path
import streamlit as st
import pandas as pd

from database.db_connection import (
    get_all_policies,
    insert_policy,
    insert_policy_objective,
    insert_policy_indicator,
    get_policy_objectives_with_indicators,
    update_policy_objective,
    update_policy_indicator,
    delete_policy_objective,
)
from modules.policy_extractor import PolicyExtractor
from modules.nlp_processor import PolicyNLPProcessor
from utils.helpers import apply_custom_css, render_academic_disclaimer

apply_custom_css()

st.markdown('<div class="main-header">Module 1: Policy Understanding (NLP)</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Automated extraction of policy objectives, target populations, measurable indicators, and statutory targets</div>',
    unsafe_allow_html=True,
)

render_academic_disclaimer()

# Initialize NLP & Extractor instances
extractor = PolicyExtractor()
nlp = PolicyNLPProcessor()

# Session state initialization for extracted policy
if "current_extracted_policy" not in st.session_state:
    st.session_state.current_extracted_policy = None

# Tabbed Interface
tab_upload, tab_review, tab_database = st.tabs(["📤 Upload & Extract", "✏️ Review & Approve Extracted Data", "📚 Stored Policies"])

# -------------------------------------------------------------
# TAB 1: Upload & Extract
# -------------------------------------------------------------
with tab_upload:
    st.markdown("### Upload Policy Document")
    st.write("Upload official education policy documents in **PDF**, **DOCX**, or **TXT** format.")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Choose a document file",
            type=["pdf", "docx", "txt"],
            help="Upload NEP 2020 or any other education policy document",
        )

    with col2:
        st.markdown("**Quick Preset:**")
        load_sample = st.button("📄 Load Official NEP 2020 Document", use_container_width=True)

    file_to_process = None
    filename_to_process = ""

    if uploaded_file is not None:
        file_to_process = io.BytesIO(uploaded_file.read())
        filename_to_process = uploaded_file.name
    elif load_sample:
        sample_path = Path(__file__).resolve().parent.parent / "data" / "policies" / "NEP_2020_School_Education.txt"
        if sample_path.exists():
            with open(sample_path, "rb") as f:
                file_to_process = io.BytesIO(f.read())
            filename_to_process = "NEP_2020_School_Education.txt"
            st.success("Loaded built-in NEP 2020 School Education document!")

    if file_to_process:
        with st.spinner("Extracting text and running NLP pipeline..."):
            try:
                extracted = extractor.extract_from_file(file_to_process, filename_to_process)
                analysis = nlp.analyze_policy_document(extracted)
                
                # Store in session state
                st.session_state.current_extracted_policy = {
                    "filename": filename_to_process,
                    "extracted": extracted,
                    "analysis": analysis,
                }
                st.success(f"Successfully processed '{filename_to_process}'! Found {analysis['total_objectives_found']} key objectives/indicators.")
            except Exception as e:
                st.error(f"Error processing document: {e}")

    # Display results if available
    curr = st.session_state.current_extracted_policy
    if curr:
        st.markdown("---")
        st.markdown(f"### Extraction Results for: `{curr['filename']}`")

        c1, c2, c3, c4 = st.columns(4)
        analysis_data = curr["analysis"]
        with c1:
            st.metric("Detected Policy", analysis_data.get("policy_name", "N/A"))
        with c2:
            st.metric("Sector", analysis_data.get("sector", "N/A"))
        with c3:
            st.metric("Enactment Year", analysis_data.get("publication_year", 2020))
        with c4:
            st.metric("Objectives Extracted", analysis_data.get("total_objectives_found", 0))

        # Text and Sections Explorer
        with st.expander("📖 View Document Sections & Text Snippets", expanded=False):
            sec_tabs = st.tabs(["Cleaned Document Text", "Detected Sections"])
            with sec_tabs[0]:
                st.text_area("Extracted Clean Text", curr["extracted"]["cleaned_text"][:4000] + ("..." if len(curr["extracted"]["cleaned_text"]) > 4000 else ""), height=250)
            with sec_tabs[1]:
                sections_list = curr["extracted"].get("sections", [])
                for s in sections_list[:10]:
                    st.markdown(f"**{s['title']}** (Page {s.get('page', 1)})")
                    st.caption(s["text"][:200] + "...")

        # Structured JSON Viewer
        st.markdown("#### Structured Policy Records (JSON)")
        st.json(analysis_data.get("objectives", [])[:5])

        # Action button to save to DB
        if st.button("💾 Approve & Save Policy to Database", type="primary"):
            try:
                policy_id = insert_policy(
                    title=analysis_data.get("policy_name", "Policy Document"),
                    code=curr["filename"].replace(".", "_").upper()[:20],
                    sector=analysis_data.get("sector", "School Education"),
                    publication_year=analysis_data.get("publication_year", 2020),
                    document_name=curr["filename"],
                    document_text=curr["extracted"]["cleaned_text"],
                    summary=analysis_data.get("summary", ""),
                )

                for obj in analysis_data.get("objectives", []):
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
                st.success(f"Saved policy and {len(analysis_data.get('objectives', []))} objectives to database!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to save policy: {e}")

# -------------------------------------------------------------
# TAB 2: Review & Approve Extracted Data
# -------------------------------------------------------------
with tab_review:
    st.markdown("### Verification and Human-in-the-Loop Review")
    st.write(
        "Academic and policy integrity requires that extracted policy objectives and numerical targets "
        "are editable and verifiable before running ML impact analysis."
    )

    policies = get_all_policies()
    if policies:
        selected_p = st.selectbox(
            "Select Policy to Review",
            options=policies,
            format_func=lambda x: f"{x['title']} ({x.get('sector', 'School')})",
            key="sb_review_policy",
        )

        records = get_policy_objectives_with_indicators(selected_p["id"])
        if records:
            st.markdown(f"Found **{len(records)}** mapped objectives and indicators:")

            for r in records:
                with st.expander(f"🎯 {r['objective_title']} | Indicator: {r.get('indicator_name', 'None')}", expanded=False):
                    with st.form(key=f"form_obj_{r['objective_id']}_{r.get('indicator_id')}"):
                        c_o1, c_o2 = st.columns(2)
                        with c_o1:
                            new_title = st.text_input("Objective Title", value=r["objective_title"])
                            new_target_group = st.text_input("Target Group", value=r.get("target_group") or "")
                            new_outcome = st.text_area("Expected Outcome", value=r.get("expected_outcome") or "", height=70)
                        with c_o2:
                            new_ind_name = st.text_input("Measurable Indicator", value=r.get("indicator_name") or "")
                            new_target_val = st.number_input(
                                "Target Value",
                                value=float(r["target_value"]) if r.get("target_value") is not None else 0.0,
                                step=1.0,
                            )
                            c_u1, c_u2 = st.columns(2)
                            with c_u1:
                                new_unit = st.selectbox(
                                    "Unit",
                                    options=["%", "ratio", "count", "score"],
                                    index=0 if r.get("target_unit") == "%" else 1,
                                )
                            with c_u2:
                                new_year = st.number_input(
                                    "Target Year",
                                    value=int(r["target_year"]) if r.get("target_year") else 2030,
                                    step=1,
                                )
                            is_num = st.checkbox("Explicit Numerical Target (Stated in Policy)", value=bool(r.get("is_numerical_target", 0)))

                        col_btn1, col_btn2 = st.columns([1, 4])
                        with col_btn1:
                            submitted = st.form_submit_button("Update Record")
                        
                        if submitted:
                            update_policy_objective(
                                objective_id=r["objective_id"],
                                title=new_title,
                                target_group=new_target_group,
                                expected_outcome=new_outcome,
                                timeline_year=new_year,
                            )
                            if r.get("indicator_id"):
                                update_policy_indicator(
                                    indicator_id=r["indicator_id"],
                                    indicator_name=new_ind_name,
                                    target_value=new_target_val if is_num else None,
                                    target_unit=new_unit,
                                    target_year=new_year if is_num else None,
                                    is_numerical_target=1 if is_num else 0,
                                )
                            st.success("Record updated successfully!")
                            st.rerun()
        else:
            st.info("No objectives mapped to this policy yet.")
    else:
        st.info("No policy documents in the database. Upload one in Tab 1.")

# -------------------------------------------------------------
# TAB 3: Stored Policies
# -------------------------------------------------------------
with tab_database:
    st.markdown("### Registered Policy Documents")
    all_p = get_all_policies()
    if all_p:
        df_p = pd.DataFrame(all_p)[["id", "title", "sector", "publication_year", "document_name", "created_at"]]
        st.dataframe(df_p, use_container_width=True, hide_index=True)
    else:
        st.info("No policies stored yet.")
