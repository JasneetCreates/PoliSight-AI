"""
Time-Series Analysis & Trajectory Forecasting.
Module 3: Algorithm 4.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


class PolicyTimeSeriesForecastingModel:
    """
    Time-series analysis measuring annual fluctuations, pre-policy vs post-policy momentum,
    and estimating forward indicator paths with confidence bounds.
    """

    def __init__(self):
        pass

    def analyze_trajectory(
        self,
        df: pd.DataFrame,
        year_col: str,
        val_col: str,
        policy_year: int = 2020,
        forecast_horizon_years: int = 6,
    ) -> Dict[str, Any]:
        """
        Analyze multi-year historical changes and forecast future trajectory.
        Compares pre-policy velocity with post-policy velocity.
        """
        yearly = df.groupby(year_col)[val_col].mean().dropna().sort_values()
        years = yearly.index.values.astype(int)
        vals = yearly.values.astype(float)

        if len(years) < 3:
            raise ValueError(f"Insufficient longitudinal points ({len(years)}) for time-series forecasting.")

        # Annual fluctuations (Year-over-Year differences)
        yoy_changes = [
            {"year": int(years[i]), "value": round(float(vals[i]), 2), "yoy_change": round(float(vals[i] - vals[i - 1]), 2)}
            for i in range(1, len(years))
        ]

        # Pre-policy (<= 2020) vs Post-policy (> 2020) empirical velocity
        pre_mask = years <= policy_year
        post_mask = years >= policy_year

        pre_velocity = None
        post_velocity = None

        if np.sum(pre_mask) >= 2:
            pre_x = years[pre_mask]
            pre_y = vals[pre_mask]
            pre_velocity = round(float((pre_y[-1] - pre_y[0]) / (pre_x[-1] - pre_x[0])), 3)

        if np.sum(post_mask) >= 2:
            post_x = years[post_mask]
            post_y = vals[post_mask]
            post_velocity = round(float((post_y[-1] - post_y[0]) / (post_x[-1] - post_x[0])), 3)

        # Average historical annual drift
        annual_slopes = np.diff(vals) / np.diff(years)
        mean_drift = float(np.mean(annual_slopes))
        std_drift = float(np.std(annual_slopes)) if len(annual_slopes) > 1 else 0.5

        # In-sample validation using chronological holdout
        split_idx = max(2, len(years) - 2)
        train_x, test_x = years[:split_idx], years[split_idx:]
        train_y, test_y = vals[:split_idx], vals[split_idx:]

        # Simple drift projection from holdout point
        holdout_preds = train_y[-1] + (test_x - train_x[-1]) * mean_drift
        mae = float(mean_absolute_error(test_y, holdout_preds)) if len(test_y) > 0 else 0.0
        rmse = float(np.sqrt(mean_squared_error(test_y, holdout_preds))) if len(test_y) > 0 else 0.0

        # Future forecast points
        latest_year = int(years[-1])
        latest_val = float(vals[-1])
        future_years = [latest_year + i for i in range(1, forecast_horizon_years + 1)]

        forecast_points = []
        for i, f_yr in enumerate(future_years, 1):
            projected = latest_val + (i * mean_drift)
            # 90% confidence interval estimation based on historical drift variance
            margin = 1.645 * std_drift * np.sqrt(i)
            forecast_points.append({
                "year": int(f_yr),
                "actual": None,
                "projected": round(float(projected), 2),
                "lower_bound": round(float(projected - margin), 2),
                "upper_bound": round(float(projected + margin), 2),
                "type": "Forecast",
            })

        historical_points = [
            {"year": int(yr), "actual": round(float(v), 2), "projected": round(float(v), 2), "lower_bound": None, "upper_bound": None, "type": "Historical"}
            for yr, v in zip(years, vals)
        ]

        return {
            "metrics": {
                "validation_mae": round(mae, 3),
                "validation_rmse": round(rmse, 3),
                "average_annual_drift": round(mean_drift, 3),
                "drift_volatility_std": round(std_drift, 3),
                "pre_policy_annual_velocity": pre_velocity,
                "post_policy_annual_velocity": post_velocity,
                "policy_enactment_year": policy_year,
            },
            "yoy_fluctuations": yoy_changes,
            "trajectory": historical_points + forecast_points,
        }
