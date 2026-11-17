"""
Unit and Integration Tests for Module 3: Machine Learning Impact Analysis (Phase 4 & 5).
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from modules.indicator_matcher import IndicatorMatcher
from modules.impact_analyzer import ImpactAnalyzer
from models.regression_model import PolicyLinearRegressionModel
from models.random_forest_model import PolicyRandomForestModel
from models.clustering_model import PolicyRegionalClusterModel
from models.forecasting_model import PolicyTimeSeriesForecastingModel
from database.db_connection import init_db, get_model_results

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_CSV = BASE_DIR / "data" / "sample" / "udise_education_dataset.csv"


@pytest.fixture(scope="module")
def sample_dataset():
    """Load sample UDISE dataset."""
    assert SAMPLE_CSV.exists(), "Sample CSV must exist for ML tests"
    return pd.read_csv(SAMPLE_CSV)


@pytest.fixture(scope="module")
def setup_db():
    """Ensure database schema is ready."""
    init_db()
    yield


def test_indicator_matcher():
    """Test indicator ontology matching and directional gap math."""
    matcher = IndicatorMatcher()

    # Test matching text
    obj1 = {"objective_title": "Universal Access to School Education", "indicator_name": "Gross Enrolment Ratio"}
    assert matcher.match_objective_to_indicator(obj1) == "ger"

    obj2 = {"objective_title": "Teacher Rationalization", "indicator_name": "Pupil Teacher Ratio"}
    assert matcher.match_objective_to_indicator(obj2) == "pupil_teacher_ratio"

    # Test directional target gap
    # GER: Higher is better -> target 100%, actual 90% -> gap = +10% (deficit)
    assert matcher.calculate_target_gap(actual_val=90.0, target_val=100.0, indicator_code="ger") == 10.0
    # Dropout: Lower is better -> target 0%, actual 12% -> gap = +12% (excess)
    assert matcher.calculate_target_gap(actual_val=12.0, target_val=0.0, indicator_code="dropout_rate") == 12.0
    # PTR: Lower is better -> target 30:1, actual 35:1 -> gap = +5.0 (excess)
    assert matcher.calculate_target_gap(actual_val=35.0, target_val=30.0, indicator_code="pupil_teacher_ratio") == 5.0

    # Percentage change formula
    assert matcher.calculate_percentage_change(80.0, 88.0) == 10.0
    assert matcher.calculate_percentage_change(20.0, 15.0) == -25.0


def test_linear_regression_model(sample_dataset):
    """Test OLS regression, chronological split, slope, and 2030 projection."""
    yearly = sample_dataset.groupby("year")["ger"].mean().reset_index()
    model = PolicyLinearRegressionModel()

    result = model.fit_and_evaluate(
        df=yearly,
        year_col="year",
        val_col="ger",
        target_year=2030,
        target_val=100.0,
    )

    metrics = result["metrics"]
    assert "slope_annual_velocity" in metrics
    assert metrics["slope_annual_velocity"] > 0  # Should show upward progress
    assert "r2_score" in metrics
    assert "rmse" in metrics
    assert metrics["projected_target_year"] == 2030
    assert metrics["projected_value"] > 80.0
    assert len(result["trajectory"]) >= len(yearly)


def test_random_forest_model(sample_dataset):
    """Test Random Forest multivariate evaluation and causal disclaimer."""
    rf = PolicyRandomForestModel(n_estimators=50, random_state=42)
    result = rf.fit_and_evaluate(
        df=sample_dataset,
        target_col="ger",
    )

    metrics = result["metrics"]
    assert metrics["total_samples"] == len(sample_dataset)
    assert "mean_r2" in metrics

    # Check feature importances
    fi = result["feature_importances"]
    assert len(fi) >= 4
    total_imp = sum(f["importance"] for f in fi)
    assert abs(total_imp - 1.0) < 0.05

    # Check academic disclaimer
    assert "causal" in result["causal_disclaimer"].lower()


def test_clustering_model(sample_dataset):
    """Test K-Means clustering optimization and cohort assignment."""
    cluster_model = PolicyRegionalClusterModel(random_state=42)
    result = cluster_model.find_optimal_clusters(
        df=sample_dataset,
        feature_cols=["ger", "dropout_rate", "pupil_teacher_ratio"],
        region_col="state",
        k_min=2,
        k_max=4,
    )

    assert result["optimal_k"] in [2, 3, 4]
    assert result["best_silhouette_score"] > 0
    assert len(result["evaluations"]) >= 2

    # Check assigned cohorts
    clusters = result["regional_clusters"]
    assert len(clusters) == sample_dataset["state"].nunique()
    assert all("performance_cohort" in c for c in clusters)


def test_time_series_forecasting(sample_dataset):
    """Test time-series pre/post policy velocity and future forecast bounds."""
    forecaster = PolicyTimeSeriesForecastingModel()
    result = forecaster.analyze_trajectory(
        df=sample_dataset,
        year_col="year",
        val_col="ger",
        policy_year=2020,
        forecast_horizon_years=5,
    )

    metrics = result["metrics"]
    assert "average_annual_drift" in metrics
    assert "pre_policy_annual_velocity" in metrics
    assert "post_policy_annual_velocity" in metrics

    traj = result["trajectory"]
    forecast_points = [p for p in traj if p["type"] == "Forecast"]
    assert len(forecast_points) == 5
    for p in forecast_points:
        assert p["lower_bound"] <= p["projected"] <= p["upper_bound"]


def test_impact_analyzer_db_integration(setup_db, sample_dataset):
    """Test ImpactAnalyzer logging runs into the SQLite database."""
    analyzer = ImpactAnalyzer()

    # Run regression with DB logging
    res = analyzer.run_linear_regression_analysis(
        df=sample_dataset,
        indicator_code="ger",
        target_year=2030,
        target_val=100.0,
        save_to_db=True,
    )
    assert res is not None

    db_results = get_model_results()
    assert not db_results.empty
    assert any("Linear Regression" in str(name) for name in db_results["model_type"].values)
