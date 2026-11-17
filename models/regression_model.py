"""
Linear Regression Model for Time-Series Indicator Velocity & Policy Milestone Projections.
Module 3: Algorithm 1.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class PolicyLinearRegressionModel:
    """
    Fits Ordinary Least Squares Linear Regression over longitudinal academic years.
    Evaluates annual velocity, goodness-of-fit, and projects indicator milestone values.
    """

    def __init__(self):
        self.model = LinearRegression()
        self.is_fitted = False
        self.metrics: Dict[str, float] = {}

    def fit_and_evaluate(
        self,
        df: pd.DataFrame,
        year_col: str,
        val_col: str,
        target_year: int = 2030,
        target_val: Optional[float] = None,
        test_size: float = 0.25,
    ) -> Dict[str, Any]:
        """
        Fit model using chronological splitting to prevent temporal data leakage.
        Returns metrics, slope, projections, and fitted trajectory.
        """
        clean_df = df[[year_col, val_col]].dropna().sort_values(by=year_col)
        if len(clean_df) < 3:
            raise ValueError(f"Insufficient longitudinal observations ({len(clean_df)} records) for regression analysis.")

        X = clean_df[[year_col]].values
        y = clean_df[val_col].values

        # Chronological train/test split
        split_idx = max(2, int(len(X) * (1 - test_size)))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Train on historical horizon
        self.model.fit(X_train, y_train)
        self.is_fitted = True

        # Evaluate on test partition if test exists; otherwise train
        eval_X = X_test if len(X_test) > 0 else X_train
        eval_y = y_test if len(y_test) > 0 else y_train
        preds_eval = self.model.predict(eval_X)

        mae = float(mean_absolute_error(eval_y, preds_eval))
        mse = float(mean_squared_error(eval_y, preds_eval))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(eval_y, preds_eval)) if len(eval_y) > 1 else float(self.model.score(X_train, y_train))

        # Refit on all available data for future projection
        self.model.fit(X, y)
        all_preds = self.model.predict(X)
        overall_r2 = float(r2_score(y, all_preds))

        slope = float(self.model.coef_[0])
        intercept = float(self.model.intercept_)

        # Project to future policy target year
        proj_val = float(self.model.predict([[target_year]])[0])

        gap = None
        target_status = "N/A"
        if target_val is not None:
            gap = round(target_val - proj_val, 2)
            target_status = "Achievable on Current Trajectory" if gap <= 0 else f"Deficit of {gap:.1f} pts"

        # Generate future projection points from latest year to target_year
        latest_year = int(clean_df[year_col].max())
        future_years = list(range(latest_year + 1, target_year + 1))
        future_preds = self.model.predict([[yr] for yr in future_years]).tolist() if future_years else []

        self.metrics = {
            "mae": round(mae, 3),
            "mse": round(mse, 3),
            "rmse": round(rmse, 3),
            "r2_score": round(overall_r2, 3),
            "slope_annual_velocity": round(slope, 3),
            "intercept": round(intercept, 3),
            "projected_target_year": target_year,
            "projected_value": round(proj_val, 2),
            "target_value": target_val,
            "projected_gap": gap,
            "target_status": target_status,
        }

        # Format historical and projected points
        fitted_points = [
            {"year": int(yr), "actual": float(act), "fitted": round(float(fit), 2), "type": "Historical Observation"}
            for yr, act, fit in zip(clean_df[year_col], y, all_preds)
        ]

        projection_points = [
            {"year": int(yr), "actual": None, "fitted": round(float(fit), 2), "type": "Model Projection"}
            for yr, fit in zip(future_years, future_preds)
        ]

        return {
            "metrics": self.metrics,
            "trajectory": fitted_points + projection_points,
            "summary_formula": f"{val_col} = {slope:+.3f} * Year {intercept:+.2f}",
        }
