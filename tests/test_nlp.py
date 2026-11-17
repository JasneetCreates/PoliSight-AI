"""
Unit and Integration Tests for NLP Policy Understanding (Phase 1 & 2).
"""

import io
import os
import pytest
from pathlib import Path

from modules.policy_extractor import PolicyExtractor
from modules.nlp_processor import PolicyNLPProcessor
from database.db_connection import (
    init_db,
    insert_policy,
    insert_policy_objective,
    insert_policy_indicator,
    get_policy_objectives_with_indicators,
    get_all_policies,
)

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_POLICY_PATH = BASE_DIR / "data" / "policies" / "NEP_2020_School_Education.txt"
SAMPLE_POLICY_PDF_PATH = BASE_DIR / "data" / "policies" / "NEP_2020_School_Education.pdf"


@pytest.fixture(scope="module")
def setup_database():
    """Initialize fresh test database."""
    init_db(force_recreate=True)
    yield
    # Teardown if needed


def test_extractor_from_pdf():
    """Test PolicyExtractor reading from PDF file (Test Case 1)."""
    extractor = PolicyExtractor()
    assert SAMPLE_POLICY_PDF_PATH.exists(), "Sample NEP policy PDF must exist"

    result = extractor.extract_from_file(SAMPLE_POLICY_PDF_PATH, "NEP_2020_School_Education.pdf")
    assert result["document_name"] == "NEP_2020_School_Education.pdf"
    assert result["file_type"] == ".pdf"
    assert len(result["raw_text"]) > 500
    assert result["total_pages"] >= 1
    assert "Gross Enrolment Ratio" in result["cleaned_text"]

    nlp = PolicyNLPProcessor()
    analysis = nlp.analyze_policy_document(result)
    assert analysis["policy_name"] == "National Education Policy 2020"
    assert any("ger" in o["indicator_code"] for o in analysis["objectives"])


def test_extractor_from_txt():
    """Test PolicyExtractor reading from plain text."""
    extractor = PolicyExtractor()
    assert SAMPLE_POLICY_PATH.exists(), "Sample NEP policy file must exist"

    result = extractor.extract_from_file(SAMPLE_POLICY_PATH, "NEP_2020_School_Education.txt")
    assert result["document_name"] == "NEP_2020_School_Education.txt"
    assert len(result["raw_text"]) > 500
    assert len(result["sections"]) >= 5
    assert any("EARLY CHILDHOOD" in s["title"].upper() for s in result["sections"])
    assert any("DROPOUT" in s["title"].upper() for s in result["sections"])


def test_extractor_from_bytes_io():
    """Test PolicyExtractor handling memory BytesIO object (simulating Streamlit upload)."""
    extractor = PolicyExtractor()
    dummy_text = "CHAPTER 1: UNIVERSAL ACCESS\nThe target is to achieve 100% Gross Enrolment Ratio by 2030."
    file_bytes = io.BytesIO(dummy_text.encode("utf-8"))

    result = extractor.extract_from_file(file_bytes, "test_upload.txt")
    assert result["document_name"] == "test_upload.txt"
    assert "100% Gross Enrolment Ratio" in result["cleaned_text"]


def test_nlp_policy_metadata_detection():
    """Test detection of policy title, sector, and year."""
    extractor = PolicyExtractor()
    nlp = PolicyNLPProcessor()

    extracted = extractor.extract_from_file(SAMPLE_POLICY_PATH, "NEP_2020_School_Education.txt")
    analysis = nlp.analyze_policy_document(extracted)

    assert "NEP" in analysis["policy_name"] or "National Education Policy" in analysis["policy_name"]
    assert analysis["sector"] == "School Education"
    assert analysis["publication_year"] == 2020


def test_nlp_objective_and_indicator_extraction():
    """Test that NLP accurately identifies core NEP 2020 objectives and indicators."""
    extractor = PolicyExtractor()
    nlp = PolicyNLPProcessor()

    extracted = extractor.extract_from_file(SAMPLE_POLICY_PATH, "NEP_2020_School_Education.txt")
    analysis = nlp.analyze_policy_document(extracted)
    objectives = analysis["objectives"]

    assert len(objectives) >= 4

    # Check for GER indicator
    ger_objs = [o for o in objectives if "ger" in o.get("indicator_code", "")]
    assert len(ger_objs) > 0
    ger = ger_objs[0]
    assert ger["is_numerical_target"] == 1
    assert ger["target_value"] == 100.0
    assert ger["target_year"] == 2030
    assert "School Students" in ger["target_group"] or "Children" in ger["target_group"]

    # Check for PTR indicator
    ptr_objs = [o for o in objectives if "pupil_teacher_ratio" in o.get("indicator_code", "") or "ptr" in o.get("indicator_code", "")]
    assert len(ptr_objs) > 0
    ptr = ptr_objs[0]
    assert ptr["is_numerical_target"] == 1
    assert ptr["target_value"] in [30.0, 25.0]


def test_nlp_no_fake_numerical_targets():
    """Test that general descriptive text does NOT invent a numerical target."""
    extractor = PolicyExtractor()
    nlp = PolicyNLPProcessor()

    dummy_doc = """
    CHAPTER 4: CURRICULUM AND PEDAGOGY
    The curricular and pedagogical structure of school education will be reconfigured into a 5+3+3+4 design.
    Learning should be holistic, integrated, enjoyable, and engaging.
    """
    file_bytes = io.BytesIO(dummy_doc.encode("utf-8"))
    extracted = extractor.extract_from_file(file_bytes, "curriculum.txt")
    analysis = nlp.analyze_policy_document(extracted)

    for obj in analysis["objectives"]:
        if obj["indicator_code"] not in ["ger", "pupil_teacher_ratio"]:
            # If no numerical target was explicitly written in the dummy doc, it should not invent one
            assert obj.get("is_numerical_target") == 0 or obj.get("target_value") is None


def test_database_persistence(setup_database):
    """Test saving extracted policy and objectives into the SQLite database."""
    extractor = PolicyExtractor()
    nlp = PolicyNLPProcessor()

    extracted = extractor.extract_from_file(SAMPLE_POLICY_PATH, "NEP_2020_School_Education.txt")
    analysis = nlp.analyze_policy_document(extracted)

    policy_id = insert_policy(
        title=analysis["policy_name"],
        code="TEST_NEP_2020",
        sector=analysis["sector"],
        publication_year=analysis["publication_year"],
        document_name="NEP_2020_School_Education.txt",
        document_text=extracted["cleaned_text"],
        summary=analysis["summary"],
    )
    assert policy_id > 0

    for obj in analysis["objectives"]:
        obj_id = insert_policy_objective(
            policy_id=policy_id,
            title=obj["objective"],
            section_reference=obj.get("section_reference"),
            page_number=obj.get("page_number", 1),
            target_group=obj.get("target_group"),
            expected_outcome=obj.get("expected_outcome"),
            timeline_year=obj.get("target_year"),
        )
        assert obj_id > 0

        ind_id = insert_policy_indicator(
            objective_id=obj_id,
            indicator_name=obj["indicator"],
            indicator_code=obj.get("indicator_code"),
            target_value=obj.get("target_value"),
            target_unit=obj.get("target_unit", "%"),
            target_year=obj.get("target_year"),
            is_numerical_target=obj.get("is_numerical_target", 0),
            source_reference=obj.get("source"),
        )
        assert ind_id > 0

    records = get_policy_objectives_with_indicators(policy_id)
    assert len(records) >= len(analysis["objectives"])
    assert any(r["target_value"] == 100.0 for r in records)
