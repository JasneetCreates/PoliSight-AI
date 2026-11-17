"""
Test for Module 5 / Phase 9: Report Generation (ReportLab PDF).
"""

from pathlib import Path
import pypdf
from modules.report_generator import PolicyReportGenerator

BASE_DIR = Path(__file__).resolve().parent.parent


def test_pdf_report_generation():
    """Test generating and verifying formal academic policy evaluation PDF."""
    generator = PolicyReportGenerator()

    policy_meta = {
        "title": "National Education Policy 2020",
        "sector": "School Education",
        "publication_year": 2020,
        "summary": "Universal access, foundational learning, and infrastructure parity.",
    }

    objectives = [
        {
            "objective_title": "Universal Access to School Education",
            "indicator_name": "Gross Enrolment Ratio",
            "target_value": 100.0,
            "target_unit": "%",
            "target_year": 2030,
            "is_numerical_target": 1,
        },
        {
            "objective_title": "Teacher Rationalization",
            "indicator_name": "Pupil Teacher Ratio",
            "target_value": 30.0,
            "target_unit": "ratio",
            "target_year": 2025,
            "is_numerical_target": 1,
        },
    ]

    risk_summary = [
        {
            "region_name": "Bihar",
            "risk_level": "High Risk",
            "risk_score": 42.5,
            "gap_summary": "GER: 76.5%, Dropout: 19.8%, PTR: 38.4:1",
            "explanation": "Secondary dropout of 19.8% exceeds alert ceiling; PTR of 38.4:1 exceeds 30:1 norm.",
        }
    ]

    recommendations = [
        {
            "title": "Secondary School Transition Taskforce (Bihar)",
            "affected_region": "Bihar",
            "priority_level": "Critical",
            "supporting_data": "Observed Dropout: 19.8%",
            "suggested_investigation": "Audit spatial secondary school distribution within 5km radius.",
        }
    ]

    pdf_bytes = generator.generate_pdf_report(
        policy_meta=policy_meta,
        objectives=objectives,
        data_summary={"states_count": 19, "total_records": 413},
        risk_summary=risk_summary,
        recommendations=recommendations,
    )

    assert len(pdf_bytes) > 1000

    # Validate with pypdf
    import io
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    assert len(reader.pages) >= 1
    extracted_text = reader.pages[0].extract_text()
    assert "Policy-to-Impact Intelligence System" in extracted_text
    assert "National Education Policy 2020" in extracted_text
