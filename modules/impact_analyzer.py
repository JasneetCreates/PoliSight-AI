"""
Impact Analyzer.
Central analytical coordinator orchestrating indicator matching, statistical algorithms,
and machine learning models (Linear Regression, Random Forest, Clustering, Time-Series).
Module 3.
"""

import json
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from database.db_connection import (
    save_model_result,
    get_connection,
)
from modules.indicator_matcher import IndicatorMatcher
from models.regression_model import PolicyLinearRegressionModel
from models.random_forest_model import PolicyRandomForestModel
from models.clustering_model import PolicyRegionalClusterModel
from models.forecasting_model import PolicyTimeSeriesForecastingModel


class ImpactAnalyzer:
    """Orchestrates comprehensive econometric and ML impact evaluation."""

    def __init__(self):
        self.matcher = IndicatorMatcher()
        self.reg_model = PolicyLinearRegressionModel()
        self.rf_model = PolicyRandomForestModel()
        self.cluster_model = PolicyRegionalClusterModel()
        self.forecast_model = PolicyTimeSeriesForecastingModel()

    def run_linear_regression_analysis(
        self,
        df: pd.DataFrame,
        indicator_code: str,
        target_year: int = 2030,
        target_val: Optional[float] = None,
        save_to_db: bool = True,
    ) -> Dict[str, Any]:
        """Execute linear regression trend analysis and milestone projections."""
        if indicator_code not in df.columns:
            raise ValueError(f"Indicator '{indicator_code}' not available in dataset.")

        yearly = df.groupby("year")[indicator_code].mean().reset_index()
        result = self.reg_model.fit_and_evaluate(
            df=yearly,
            year_col="year",
            val_col=indicator_code,
            target_year=target_year,
            target_val=target_val,
        )

        if save_to_db:
            min_yr, max_yr = int(df["year"].min()), int(df["year"].max())
            save_model_result(
                model_name=f"Linear Regression - {indicator_code}",
                model_type="Linear Regression",
                target_variable=indicator_code,
                features_used="year",
                train_year_range=f"{min_yr}-{max_yr}",
                metrics_json=json.dumps(result["metrics"]),
                model_params_json=json.dumps({"target_year": target_year, "target_val": target_val}),
            )

        return result

    def run_random_forest_analysis(
        self,
        df: pd.DataFrame,
        target_col: str = "ger",
        feature_cols: Optional[List[str]] = None,
        save_to_db: bool = True,
    ) -> Dict[str, Any]:
        """Execute Random Forest feature importance evaluation."""
        result = self.rf_model.fit_and_evaluate(
            df=df,
            target_col=target_col,
            feature_cols=feature_cols,
        )

        if save_to_db:
            min_yr, max_yr = int(df["year"].min()), int(df["year"].max())
            save_model_result(
                model_name=f"Random Forest - {target_col}",
                model_type="Random Forest Regressor",
                target_variable=target_col,
                features_used=", ".join(result["features_used"]),
                train_year_range=f"{min_yr}-{max_yr}",
                metrics_json=json.dumps(result["metrics"]),
                model_params_json=json.dumps({"n_estimators": 100}),
            )

        return result

    def run_clustering_analysis(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        region_col: str = "state",
        save_to_db: bool = True,
    ) -> Dict[str, Any]:
        """Execute K-Means regional clustering with Silhouette score optimization."""
        default_features = ["ger", "dropout_rate", "pupil_teacher_ratio"]
        feats = [f for f in (feature_cols or default_features) if f in df.columns]

        result = self.cluster_model.find_optimal_clusters(
            df=df,
            feature_cols=feats,
            region_col=region_col,
        )

        if save_to_db:
            min_yr, max_yr = int(df["year"].min()), int(df["year"].max())
            save_model_result(
                model_name=f"K-Means Clustering ({region_col})",
                model_type="K-Means",
                target_variable="Regional Clusters",
                features_used=", ".join(feats),
                train_year_range=f"{min_yr}-{max_yr}",
                metrics_json=json.dumps({
                    "optimal_k": result["optimal_k"],
                    "silhouette_score": result["best_silhouette_score"],
                }),
                model_params_json=json.dumps({"evaluations": result["evaluations"]}),
            )

        return result

    def run_time_series_forecast(
        self,
        df: pd.DataFrame,
        indicator_code: str,
        policy_year: int = 2020,
        horizon_years: int = 6,
        save_to_db: bool = True,
    ) -> Dict[str, Any]:
        """Execute longitudinal time-series analysis and forecasting."""
        result = self.forecast_model.analyze_trajectory(
            df=df,
            year_col="year",
            val_col=indicator_code,
            policy_year=policy_year,
            forecast_horizon_years=horizon_years,
        )

        if save_to_db:
            min_yr, max_yr = int(df["year"].min()), int(df["year"].max())
            save_model_result(
                model_name=f"Time-Series Forecasting - {indicator_code}",
                model_type="Time-Series Drift",
                target_variable=indicator_code,
                features_used="longitudinal_years",
                train_year_range=f"{min_yr}-{max_yr}",
                metrics_json=json.dumps(result["metrics"]),
            )

        return result
