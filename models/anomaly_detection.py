"""
Anomaly Detection Model for Regional Educational Indicators.
Module 4.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class PolicyAnomalyDetector:
    """
    Employs Isolation Forest to detect multidimensional outliers and structural irregularities
    across educational indicators (e.g. sharp divergence between enrolment and infrastructure).
    """

    def __init__(self, contamination: float = 0.15, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
        )
        self.scaler = StandardScaler()

    def fit_and_detect(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        region_col: str = "state",
    ) -> pd.DataFrame:
        """
        Fit Isolation Forest and assign anomaly labels (-1 = anomalous, 1 = normal)
        and normalized anomaly scores (0.0 = completely normal, 1.0 = severe anomaly).
        """
        clean_df = df.dropna(subset=feature_cols).copy()
        if len(clean_df) < 5:
            # Not enough data for isolation forest; return neutral scores
            clean_df["is_anomaly"] = False
            clean_df["anomaly_score"] = 0.0
            return clean_df

        X = clean_df[feature_cols].values
        scaled_X = self.scaler.fit_transform(X)

        self.model.fit(scaled_X)
        preds = self.model.predict(scaled_X)  # -1 for anomaly, 1 for normal
        raw_scores = self.model.decision_function(scaled_X)

        # Normalize score between 0 (normal) and 1 (extreme anomaly)
        # decision_function gives negative values for anomalies, positive for normal
        min_s, max_s = raw_scores.min(), raw_scores.max()
        if max_s > min_s:
            norm_scores = 1.0 - ((raw_scores - min_s) / (max_s - min_s))
        else:
            norm_scores = np.zeros_like(raw_scores)

        clean_df["is_anomaly"] = (preds == -1)
        clean_df["anomaly_score"] = np.round(norm_scores, 3)

        return clean_df
