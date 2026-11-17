"""
Unit and Integration Tests for Module 4 (Risk Detection) and Module 5 (Recommendation Engine).
"""

import pytest
import pandas as pd
from pathlib import Path

from models.anomaly_detection import PolicyAnomalyDetector
from modules.risk_detector import PolicyRiskDetector
from modules.recommendation_engine import PolicyRecommendationEngine
from database.db_connection import (
    init_db,
    save_risk_records_batch,
    get_latest_risk_analysis,
    save_recommendations_batch,
    get_all_recommendations,
)

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_CSV = BASE_DIR / "data" / "sample" / "udise_education_dataset.csv"


@pytest.fixture(scope="module")
def sample_dataset():
    """Load sample UDISE dataset."""
    assert SAMPLE_CSV.exists(), "Sample CSV must exist for tests"
    return pd.read_csv(SAMPLE_CSV)


@pytest.fixture(scope="module")
def setup_db():
    """Ensure database schema is ready."""
    init_db()
    yield


def test_anomaly_detector(sample_dataset):
    """Test Isolation Forest outlier detection across educational vectors."""
    detector = PolicyAnomalyDetector(contamination=0.15, random_state=42)
    latest_df = sample_dataset[sample_dataset["year"] == sample_dataset["year"].max()]

    result_df = detector.fit_and_detect(
        df=latest_df,
        feature_cols=["ger", "dropout_rate", "pupil_teacher_ratio"],
        region_col="state",
    )

    assert "is_anomaly" in result_df.columns
    assert "anomaly_score" in result_df.columns
    assert result_df["anomaly_score"].between(0.0, 1.0).all()
    assert result_df["is_anomaly"].dtype == bool


def test_risk_detector(sample_dataset):
    """Test multi-criteria risk scoring and regional classification."""
    detector = PolicyRiskDetector()
    risk_df = detector.evaluate_regional_risks(
        df=sample_dataset,
        region_type="state",
        target_ger=90.0,
        dropout_ceiling=12.0,
        ptr_ceiling=30.0,
    )

    assert not risk_df.empty
    assert len(risk_df) == sample_dataset["state"].nunique()

    # Check risk categories
    valid_tiers = {"High Risk", "Moderate Risk", "Low Risk"}
    assert set(risk_df["risk_level"].unique()).issubset(valid_tiers)

    # Check explanations and gap summaries
    for _, row in risk_df.iterrows():
        assert len(row["explanation"]) > 10
        assert "GER" in row["gap_summary"]
        assert row["risk_score"] >= 0.0

    # Underperforming states should have high risk score
    bihar_row = risk_df[risk_df["region_name"] == "Bihar"]
    if not bihar_row.empty:
        assert bihar_row["risk_level"].values[0] in ["High Risk", "Moderate Risk"]

    kerala_row = risk_df[risk_df["region_name"] == "Kerala"]
    if not kerala_row.empty:
        assert kerala_row["risk_level"].values[0] == "Low Risk"


def test_recommendation_engine(sample_dataset):
    """Test evidence-based recommendation synthesis and required fields."""
    detector = PolicyRiskDetector()
    rec_engine = PolicyRecommendationEngine()

    risk_df = detector.evaluate_regional_risks(df=sample_dataset, region_type="state")
    recs = rec_engine.generate_recommendations(risk_df=risk_df, education_df=sample_dataset)

    assert len(recs) > 0

    required_keys = [
        "title", "identified_problem", "affected_region", "related_indicator",
        "supporting_data", "reason", "priority_level", "suggested_investigation", "is_approved"
    ]

    for rec in recs:
        for k in required_keys:
            assert k in rec, f"Key '{k}' missing from recommendation"
        assert rec["priority_level"] in ["Critical", "High", "Medium", "Low"]
        # Academic integrity check: contains diagnostic investigation guidance
        assert "investigat" in rec["suggested_investigation"].lower() or "audit" in rec["suggested_investigation"].lower() or "review" in rec["suggested_investigation"].lower()


def test_db_persistence_risk_and_recs(setup_db, sample_dataset):
    """Test saving risk records and recommendations into database tables."""
    detector = PolicyRiskDetector()
    rec_engine = PolicyRecommendationEngine()

    risk_df = detector.evaluate_regional_risks(df=sample_dataset, region_type="state")
    recs = rec_engine.generate_recommendations(risk_df=risk_df, education_df=sample_dataset)

    # Persist risk analysis
    saved_risk_count = save_risk_records_batch(risk_df.to_dict(orient="records"))
    assert saved_risk_count == len(risk_df)
    db_risks = get_latest_risk_analysis()
    assert len(db_risks) >= len(risk_df)

    # Persist recommendations
    saved_rec_count = save_recommendations_batch(recs)
    assert saved_rec_count == len(recs)
    db_recs = get_all_recommendations()
    assert len(db_recs) >= len(recs)
