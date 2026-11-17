"""
Regional Risk Detection Engine.
Module 4.
Evaluates multi-criteria performance gaps, trend velocity, and Isolation Forest anomalies
to classify states and districts into High, Moderate, and Low risk tiers with explainability.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from database.db_connection import save_risk_records_batch
from models.anomaly_detection import PolicyAnomalyDetector
from modules.indicator_matcher import IndicatorMatcher


class PolicyRiskDetector:
    """
    Evaluates education indicator gaps, temporal trajectories, and multivariate anomalies
    to categorize regional risks and generate explainable risk alerts.
    """

    def __init__(self):
        self.anomaly_detector = PolicyAnomalyDetector()
        self.matcher = IndicatorMatcher()

    def evaluate_regional_risks(
        self,
        df: pd.DataFrame,
        region_type: str = "state",  # 'state' or 'district'
        target_ger: float = 90.0,
        dropout_ceiling: float = 12.0,
        ptr_ceiling: float = 30.0,
        internet_benchmark: float = 60.0,
        save_to_db: bool = False,
    ) -> pd.DataFrame:
        """
        Run multi-criteria risk classification on latest available records.
        """
        if df.empty:
            return pd.DataFrame()

        region_col = "state" if region_type == "state" else "district"
        if region_col not in df.columns:
            region_col = "state"

        recent_year = int(df["year"].max())
        base_year = int(df["year"].min())

        # Filter latest observations
        latest_df = df[df["year"] == recent_year].groupby(region_col).mean(numeric_only=True).reset_index()

        # Run anomaly detection across core indicator vector
        feature_cols = [c for c in ["ger", "dropout_rate", "pupil_teacher_ratio", "internet_availability"] if c in latest_df.columns]
        if len(feature_cols) >= 2:
            latest_df = self.anomaly_detector.fit_and_detect(latest_df, feature_cols=feature_cols, region_col=region_col)
        else:
            latest_df["anomaly_score"] = 0.0
            latest_df["is_anomaly"] = False

        # Calculate historical velocity where possible
        velocity_map = {}
        if recent_year > base_year:
            base_df = df[df["year"] == base_year].groupby(region_col).mean(numeric_only=True).reset_index()
            for _, b_row in base_df.iterrows():
                r_name = b_row[region_col]
                l_row = latest_df[latest_df[region_col] == r_name]
                if not l_row.empty:
                    # Annual GER change
                    ger_diff = l_row["ger"].values[0] - b_row.get("ger", l_row["ger"].values[0])
                    velocity_map[r_name] = ger_diff / (recent_year - base_year)

        risk_records = []
        for _, row in latest_df.iterrows():
            reg_name = row[region_col]
            ger_val = row.get("ger", 85.0)
            drop_val = row.get("dropout_rate", 10.0)
            ptr_val = row.get("pupil_teacher_ratio", 25.0)
            net_val = row.get("internet_availability", 50.0)
            anom_score = row.get("anomaly_score", 0.0)
            ger_vel = velocity_map.get(reg_name, 0.5)

            # Gaps
            ger_gap = max(0.0, target_ger - (ger_val if pd.notnull(ger_val) else 80.0))
            drop_gap = max(0.0, (drop_val if pd.notnull(drop_val) else 10.0) - dropout_ceiling)
            ptr_gap = max(0.0, (ptr_val if pd.notnull(ptr_val) else 25.0) - ptr_ceiling)
            net_gap = max(0.0, internet_benchmark - (net_val if pd.notnull(net_val) else 40.0))

            # Penalty for negative velocity (deterioration)
            vel_penalty = 8.0 if ger_vel < 0 else 0.0

            # Composite Score (0 - 100)
            composite_score = min(
                100.0,
                (ger_gap * 1.4) + (drop_gap * 2.2) + (ptr_gap * 1.6) + (net_gap * 0.4) + (anom_score * 15.0) + vel_penalty
            )

            # Categorization into 3 risk tiers
            if composite_score >= 32.0 or drop_gap >= 8.0 or ptr_gap >= 6.0:
                risk_tier = "High Risk"
            elif composite_score >= 14.0 or drop_gap >= 3.0 or ptr_gap >= 2.0:
                risk_tier = "Moderate Risk"
            else:
                risk_tier = "Low Risk"

            # Plain-language explanation for why this region was flagged
            flagged = []
            flag_keys = []
            if ger_gap > 0:
                flagged.append(f"GER of {ger_val:.1f}% lags behind benchmark by {ger_gap:.1f} pts")
                flag_keys.append("GER Deficit")
            if drop_gap > 0:
                flagged.append(f"Secondary Dropout of {drop_val:.1f}% exceeds alert ceiling ({dropout_ceiling:.0f}%)")
                flag_keys.append("Elevated Dropout")
            if ptr_gap > 0:
                flagged.append(f"PTR of {ptr_val:.1f}:1 exceeds statutory ceiling ({ptr_ceiling:.0f}:1)")
                flag_keys.append("High PTR")
            if net_gap > 20:
                flagged.append(f"Low digital connectivity ({net_val:.1f}%)")
                flag_keys.append("Digital Infrastructure")
            if row.get("is_anomaly", False):
                flagged.append("Multivariate anomaly pattern detected via Isolation Forest")
                flag_keys.append("Structural Anomaly")

            explanation = "; ".join(flagged) if flagged else "All primary education indicators align with expected policy targets."

            risk_records.append({
                "region_type": region_type,
                "region_name": reg_name,
                "year": recent_year,
                "risk_level": risk_tier,
                "risk_score": round(composite_score, 1),
                "anomaly_score": round(anom_score, 3),
                "primary_flagged_indicators": ", ".join(flag_keys) if flag_keys else "None",
                "gap_summary": f"GER: {ger_val:.1f}%, Dropout: {drop_val:.1f}%, PTR: {ptr_val:.1f}:1",
                "explanation": explanation,
            })

        risk_df = pd.DataFrame(risk_records).sort_values(by="risk_score", ascending=False)

        if save_to_db:
            save_risk_records_batch(risk_records)

        return risk_df
