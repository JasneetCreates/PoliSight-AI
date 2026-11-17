"""
Page 1: Policy Understanding (NLP Extraction & Review)
Module 1: Automated extraction of policy objectives, target populations, measurable indicators, and statutory targets.
"""

import io
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
)
from modules.policy_extractor import PolicyExtractor
from modules.nlp_processor import PolicyNLPProcessor
from utils.helpers import apply_custom_css, render_academic_disclaimer

apply_custom_css()

st.markdown('<div class="main-header">📄 Module 1: Policy Understanding (NLP)</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Automatically extracts objectives, target populations, and statutory numerical targets from policy documents.</div>',
    unsafe_allow_html=True,
)

extractor = PolicyExtractor()
nlp = PolicyNLPProcessor()

policies = get_all_policies()
active_policy = policies[0] if policies else None

# Upload & Quick Load Section
col_upload, col_status = st.columns([1.5, 1])

with col_upload:
    st.markdown("##### 📤 Upload Policy Document")
    uploaded_file = st.file_uploader(
        "Upload a Policy Document (PDF, DOCX, or TXT)",
        type=["pdf", "docx", "txt"],
        help="Upload NEP 2020 or any other education policy",
        label_visibility="collapsed",
    )

with col_status:
    st.markdown("##### 📌 Current Active Policy")
    if active_policy:
        st.markdown(
            f"""
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; padding:0.85rem; border-radius:8px;">
                <div style="font-weight:700; color:#1E3A8A; font-size:1rem;">{active_policy['title']}</div>
                <div style="font-size:0.8rem; color:#64748B; margin-top:0.2rem;">Sector: <b>{active_policy.get('sector', 'School')}</b> • Year: <b>{active_policy.get('publication_year', 2020)}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    btn_reload = st.button("🔄 Reset to Official NEP 2020 Document", use_container_width=True)

# Process uploaded file if provided
if uploaded_file is not None or btn_reload:
    if uploaded_file:
        file_to_process = io.BytesIO(uploaded_file.read())
        filename = uploaded_file.name
    else:
        sample_path = Path(__file__).resolve().parent.parent / "data" / "policies" / "NEP_2020_School_Education.txt"
        with open(sample_path, "rb") as f:
            file_to_process = io.BytesIO(f.read())
        filename = "NEP_2020_School_Education.txt"

    with st.spinner("Extracting text and identifying policy objectives via spaCy NLP..."):
        try:
            extracted = extractor.extract_from_file(file_to_process, filename)
            analysis = nlp.analyze_policy_document(extracted)

            policy_id = insert_policy(
                title=analysis.get("policy_name", "Policy Document"),
                code=filename.replace(".", "_").upper()[:20],
                sector=analysis.get("sector", "School Education"),
                publication_year=analysis.get("publication_year", 2020),
                document_name=filename,
                document_text=extracted["cleaned_text"],
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
            st.success(f"Successfully processed and updated {filename}!")
            st.rerun()
        except Exception as e:
            st.error(f"Error extracting document: {e}")

st.markdown("---")

# Refresh active policy objectives
if active_policy:
    objectives = get_policy_objectives_with_indicators(active_policy["id"])
else:
    objectives = []

tab_goals, tab_edit, tab_text = st.tabs([
    "🎯 Extracted Policy Targets",
    "✏️ Review & Edit Goals",
    "📖 Document Text Preview",
])

with tab_goals:
    if objectives:
        display_data = []
        for o in objectives:
            t_val = o.get("target_value")
            t_unit = o.get("target_unit", "")
            target_str = f"{t_val:.0f}{t_unit}" if t_val is not None else "Descriptive Goal"
            is_num_badge = "Statutory Target" if o.get("is_numerical_target") == 1 else "Policy Goal"

            display_data.append({
                "Policy Objective": o.get("objective_title", "General Objective"),
                "Measurable Indicator": o.get("indicator_name", "N/A"),
                "Target Value": target_str,
                "Target Year": o.get("target_year") or "2030",
                "Target Population": o.get("target_group", "School Students"),
                "Type": is_num_badge,
            })

        df_display = pd.DataFrame(display_data)
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        st.caption("ℹ️ These NLP-extracted indicators are automatically mapped to UDISE+ school data in Module 3.")
    else:
        st.info("No policy objectives mapped yet.")

with tab_edit:
    st.markdown("##### Human-in-the-Loop Review")
    st.write("Academic integrity allows researchers to verify, adjust, or refine extracted policy targets.")

    if objectives:
        sel_obj_id = st.selectbox(
            "Select Objective to Review / Modify",
            options=[o["objective_id"] for o in objectives],
            format_func=lambda oid: next((f"{o['objective_title']} ({o.get('indicator_name', '')})" for o in objectives if o["objective_id"] == oid), str(oid)),
        )
        rec = next(o for o in objectives if o["objective_id"] == sel_obj_id)

        with st.form(key=f"edit_form_{sel_obj_id}"):
            c_e1, c_e2 = st.columns(2)
            with c_e1:
                e_title = st.text_input("Objective Title", value=rec["objective_title"])
                e_group = st.text_input("Target Group", value=rec.get("target_group", "School Students"))
                e_outcome = st.text_area("Expected Outcome", value=rec.get("expected_outcome", ""), height=70)
            with c_e2:
                e_ind = st.text_input("Indicator Name", value=rec.get("indicator_name", ""))
                e_val = st.number_input("Target Value", value=float(rec.get("target_value") or 100.0), step=1.0)
                e_yr = st.number_input("Target Year", value=int(rec.get("target_year") or 2030), step=1)

            if st.form_submit_button("Save Changes to Database", type="primary"):
                update_policy_objective(
                    objective_id=rec["objective_id"],
                    title=e_title,
                    target_group=e_group,
                    expected_outcome=e_outcome,
                    timeline_year=int(e_yr),
                )
                if rec.get("indicator_id"):
                    update_policy_indicator(
                        indicator_id=rec["indicator_id"],
                        indicator_name=e_ind,
                        target_value=float(e_val),
                        target_unit=rec.get("target_unit", "%"),
                        target_year=int(e_yr),
                        is_numerical_target=1,
                    )
                st.success("Target updated successfully!")
                st.rerun()

with tab_text:
    st.markdown("##### Cleaned Document Excerpt (NEP 2020 Part I: School Education)")
    if active_policy and active_policy.get("document_text"):
        st.text_area(
            "Official Policy Text",
            value=active_policy["document_text"],
            height=300,
            disabled=True,
            label_visibility="collapsed",
        )

render_academic_disclaimer()
