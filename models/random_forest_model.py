"""
Random Forest Regression for Indicator Association & Feature Importance.
Module 3: Algorithm 2.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


class PolicyRandomForestModel:
    """
    Random Forest Regression to examine predictive associations between infrastructure/teacher
    features and educational outcomes (e.g. GER, Dropout Rate, Retention).
    """

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        self.feature_names: List[str] = []

    def fit_and_evaluate(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
        n_splits: int = 4,
    ) -> Dict[str, Any]:
        """
        Fit Random Forest regressor with cross-validation.
        Returns feature importances, predictive metrics, and causal caveat note.
        """
        default_features = [
            "pupil_teacher_ratio", "female_teacher_percentage", "electricity_availability",
            "drinking_water_availability", "toilet_availability", "internet_availability"
        ]

        feats = feature_cols if feature_cols else default_features
        avail_feats = [f for f in feats if f in df.columns and f != target_col]

        if not avail_feats:
            raise ValueError("No matching feature columns found in dataset for Random Forest regression.")

        clean_sub = df[[target_col] + avail_feats].dropna()
        if len(clean_sub) < 12:
            raise ValueError(f"Insufficient non-null observations ({len(clean_sub)} records) for Random Forest analysis.")

        X = clean_sub[avail_feats].values
        y = clean_sub[target_col].values
        self.feature_names = avail_feats

        # K-Fold Cross Validation
        actual_splits = min(n_splits, max(2, len(clean_sub) // 4))
        kf = KFold(n_splits=actual_splits, shuffle=True, random_state=self.random_state)

        mae_scores = []
        rmse_scores = []
        r2_scores = []

        for train_idx, test_idx in kf.split(X):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            fold_rf = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state)
            fold_rf.fit(X_tr, y_tr)
            preds = fold_rf.predict(X_te)

            mae_scores.append(mean_absolute_error(y_te, preds))
            rmse_scores.append(np.sqrt(mean_squared_error(y_te, preds)))
            r2_scores.append(r2_score(y_te, preds))

        # Fit on full dataset to extract importances
        self.model.fit(X, y)
        importances = self.model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in self.model.estimators_], axis=0)

        importance_list = [
            {
                "feature": f,
                "importance": round(float(imp), 4),
                "std": round(float(s), 4),
                "relative_rank": int(rank + 1),
            }
            for rank, (f, imp, s) in enumerate(
                sorted(zip(avail_feats, importances, std), key=lambda x: x[1], reverse=True)
            )
        ]

        metrics = {
            "cross_val_folds": actual_splits,
            "mean_mae": round(float(np.mean(mae_scores)), 3),
            "mean_rmse": round(float(np.mean(rmse_scores)), 3),
            "mean_r2": round(float(np.mean(r2_scores)), 3),
            "train_r2": round(float(self.model.score(X, y)), 3),
            "total_samples": len(clean_sub),
        }

        # Academic disclaimer on feature importance
        causal_note = (
            "Methodological Notice: Feature importance scores measure statistical predictive association "
            "with the target variable across the decision tree ensemble. In accordance with econometrics "
            "standards, these values do not constitute mathematical proof of direct causal impact."
        )

        return {
            "metrics": metrics,
            "feature_importances": importance_list,
            "target_variable": target_col,
            "features_used": avail_feats,
            "causal_disclaimer": causal_note,
        }
