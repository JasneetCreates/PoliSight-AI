"""
K-Means Clustering for Regional Education Performance Cohorts.
Module 3: Algorithm 3.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


class PolicyRegionalClusterModel:
    """
    Groups States and Districts into empirical performance tiers using K-Means clustering.
    Validates cluster quality using Silhouette Score and Davies-Bouldin Index.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.best_model: Optional[KMeans] = None

    def find_optimal_clusters(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        region_col: str = "state",
        k_min: int = 2,
        k_max: int = 5,
    ) -> Dict[str, Any]:
        """
        Evaluate candidate cluster numbers using Silhouette and Davies-Bouldin scores.
        Selects optimal k with maximum Silhouette coefficient.
        """
        agg_df = df.groupby(region_col)[feature_cols].mean().dropna()
        if len(agg_df) < k_min + 1:
            raise ValueError(f"Insufficient distinct regions ({len(agg_df)}) for cluster evaluation.")

        X = agg_df[feature_cols].values
        scaled_X = self.scaler.fit_transform(X)

        max_allowed_k = min(k_max, len(agg_df) - 1)
        evaluations = []

        best_k = k_min
        best_sil = -1.0

        for k in range(k_min, max_allowed_k + 1):
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(scaled_X)
            sil = float(silhouette_score(scaled_X, labels))
            db = float(davies_bouldin_score(scaled_X, labels))

            evaluations.append({
                "k": k,
                "silhouette_score": round(sil, 3),
                "davies_bouldin_index": round(db, 3),
            })

            if sil > best_sil:
                best_sil = sil
                best_k = k

        # Fit best model
        optimal_km = KMeans(n_clusters=best_k, random_state=self.random_state, n_init=10)
        final_labels = optimal_km.fit_predict(scaled_X)
        self.best_model = optimal_km

        # Assign meaningful cohort labels based on GER and dropout performance
        agg_df["cluster_id"] = final_labels
        agg_df = agg_df.reset_index()

        # Label cohorts based on relative mean performance
        cluster_means = agg_df.groupby("cluster_id")[feature_cols].mean()
        
        # Determine cohort ranking (higher GER or lower dropout gets higher rank)
        rank_metric = "ger" if "ger" in feature_cols else feature_cols[0]
        sorted_clusters = cluster_means.sort_values(by=rank_metric, ascending=False).index.tolist()

        cohort_names = ["Higher-Performing Cohort", "Moderate-Performing Cohort", "Priority Attention Cohort", "Emerging Needs Cohort", "Cohort 5"]
        label_map = {cid: cohort_names[idx] for idx, cid in enumerate(sorted_clusters)}
        agg_df["performance_cohort"] = agg_df["cluster_id"].map(label_map)

        centroids = [
            {
                "cohort": label_map.get(cid, f"Cluster {cid}"),
                **{feat: round(float(val), 2) for feat, val in zip(feature_cols, optimal_km.cluster_centers_[cid])}
            }
            for cid in range(best_k)
        ]

        return {
            "optimal_k": best_k,
            "best_silhouette_score": round(best_sil, 3),
            "evaluations": evaluations,
            "regional_clusters": agg_df[[region_col, "performance_cohort", "cluster_id"] + feature_cols].to_dict(orient="records"),
            "cluster_centroids": centroids,
            "features_used": feature_cols,
        }
