"""
Page 3: Impact Analysis (Machine Learning)
Module 3: Linear Regression, Random Forest, K-Means Clustering, and Time-Series Forecasting.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from database.db_connection import (
    get_all_policies,
    get_policy_objectives_with_indicators,
    query_education_records,
)
from modules.impact_analyzer import ImpactAnalyzer
from modules.indicator_matcher import IndicatorMatcher
from utils.helpers import (
    apply_custom_css,
    render_kpi,
    render_academic_disclaimer,
)
from utils.visualizations import plot_state_comparison_bar

apply_custom_css()

st.markdown('<div class="main-header">📈 Module 3: Impact Analysis & Machine Learning</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Evaluates empirical trends, predicts progress toward 2030 milestones, and discovers regional cohorts.</div>',
    unsafe_allow_html=True,
)

analyzer = ImpactAnalyzer()
matcher = IndicatorMatcher()
records = query_education_records()

if records.empty:
    st.info("No education records in database. Please ingest UDISE+ data in Module 2.")
    st.stop()

policies = get_all_policies()
active_policy = policies[0] if policies else None
policy_indicators = get_policy_objectives_with_indicators(active_policy["id"]) if active_policy else []

# Streamlined Control Row
col_ind, col_algo = st.columns([1.2, 1.2])

with col_ind:
    ind_choices = [
        ("Gross Enrolment Ratio (GER)", "ger"),
        ("Secondary Dropout Rate", "dropout_rate"),
        ("Pupil-Teacher Ratio (PTR)", "pupil_teacher_ratio"),
        ("Internet & Digital Facilities", "internet_availability"),
        ("Electricity Coverage", "electricity_availability"),
    ]
    sel_ind_tuple = st.selectbox("Select Indicator to Evaluate", options=ind_choices, format_func=lambda x: x[0])
    sel_ind_label, target_col = sel_ind_tuple

with col_algo:
    algo_choices = [
        "1. Linear Regression (2030 Projection)",
        "2. Random Forest (Feature Importance)",
        "3. K-Means (Regional Performance Cohorts)",
        "4. Time-Series (Pre vs Post Policy Velocity)",
    ]
    sel_algo = st.selectbox("Select Machine Learning Model", options=algo_choices)

# Fetch statutory target metadata
target_meta = next((o for o in policy_indicators if matcher.match_objective_to_indicator(o) == target_col), {})
target_val = target_meta.get("target_value")
target_year = target_meta.get("target_year", 2030)
target_unit = target_meta.get("target_unit", "%")

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------------------
# MODEL 1: LINEAR REGRESSION
# -------------------------------------------------------------
if "Linear Regression" in sel_algo:
    try:
        reg_result = analyzer.run_linear_regression_analysis(
            df=records,
            indicator_code=target_col,
            target_year=int(target_year) if target_year else 2030,
            target_val=float(target_val) if target_val is not None else None,
            save_to_db=False,
        )
        m = reg_result["metrics"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Annual Velocity (Slope)", f"{m['slope_annual_velocity']:+.2f} pts/yr")
        c2.metric("Goodness of Fit (R²)", f"{m['r2_score']:.3f}")
        c3.metric("Root Mean Squared Error", f"{m['rmse']:.2f}")
        c4.metric(f"Projected {m['projected_target_year']} Value", f"{m['projected_value']:.1f}{target_unit}")

        # Clean trajectory plot
        traj_df = pd.DataFrame(reg_result["trajectory"])
        fig = go.Figure()

        hist_df = traj_df[traj_df["type"] == "Historical Observation"]
        fig.add_trace(go.Scatter(
            x=hist_df["year"], y=hist_df["actual"],
            mode="lines+markers", name="Observed Data",
            line=dict(color="#1E3A8A", width=3),
            marker=dict(size=7),
        ))

        proj_df = traj_df[traj_df["type"] == "Model Projection"]
        if not proj_df.empty:
            fig.add_trace(go.Scatter(
                x=proj_df["year"], y=proj_df["fitted"],
                mode="lines+markers", name="Projected Path",
                line=dict(color="#10B981", width=2, dash="dot"),
                marker=dict(size=6),
            ))

        if target_val is not None:
            fig.add_hline(
                y=target_val, line_dash="solid", line_color="#DC2626", line_width=1.5,
                annotation_text=f"NEP Target ({target_val}{target_unit})",
                annotation_position="top left",
            )

        fig.update_layout(
            title=f"<b>Trajectory & 2030 Target Projection: {sel_ind_label}</b>",
            xaxis_title="Academic Year",
            yaxis_title=f"{sel_ind_label} ({target_unit})",
            template="plotly_white",
            height=380,
            margin=dict(l=30, r=20, t=40, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error in Regression: {e}")

# -------------------------------------------------------------
# MODEL 2: RANDOM FOREST
# -------------------------------------------------------------
elif "Random Forest" in sel_algo:
    try:
        rf_result = analyzer.run_random_forest_analysis(df=records, target_col=target_col, save_to_db=False)
        m = rf_result["metrics"]

        st.caption("ℹ️ Evaluates predictive associations across 100 decision trees. Does not constitute proof of causation.")

        c1, c2, c3 = st.columns(3)
        c1.metric("Model R² Score", f"{m['mean_r2']:.3f}")
        c2.metric("Cross-Validation RMSE", f"{m['mean_rmse']:.2f}")
        c3.metric("Samples Evaluated", m["total_samples"])

        fi_df = pd.DataFrame(rf_result["feature_importances"]).sort_values(by="importance", ascending=True)
        fig = px.bar(
            fi_df,
            x="importance",
            y="feature",
            orientation="h",
            labels={"importance": "Predictive Weight", "feature": "Factor"},
            title=f"<b>Feature Association with: {sel_ind_label}</b>",
            template="plotly_white",
            height=320,
            color="importance",
            color_continuous_scale="Blues",
        )
        fig.update_layout(coloraxis_showscale=False, margin=dict(l=30, r=20, t=40, b=30))
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error in Random Forest: {e}")

# -------------------------------------------------------------
# MODEL 3: K-MEANS CLUSTERING
# -------------------------------------------------------------
elif "K-Means" in sel_algo:
    try:
        cluster_result = analyzer.run_clustering_analysis(
            df=records,
            feature_cols=["ger", "dropout_rate", "pupil_teacher_ratio"],
            region_col="state",
            save_to_db=False,
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Optimal Clusters (k)", cluster_result["optimal_k"], "Auto-selected")
        c2.metric("Silhouette Score", f"{cluster_result['best_silhouette_score']:.3f}", "Cluster separation")
        c3.metric("Clustered States", len(cluster_result["regional_clusters"]))

        clustered_df = pd.DataFrame(cluster_result["regional_clusters"])
        fig = px.scatter(
            clustered_df,
            x="ger",
            y="dropout_rate",
            color="performance_cohort",
            hover_name="state",
            size="pupil_teacher_ratio",
            labels={
                "ger": "GER (%)",
                "dropout_rate": "Dropout Rate (%)",
                "performance_cohort": "Cohort",
            },
            title="<b>Regional Performance Clusters (GER vs Dropout Rate vs PTR)</b>",
            template="plotly_white",
            height=380,
        )
        fig.update_layout(margin=dict(l=30, r=20, t=40, b=30))
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error in Clustering: {e}")

# -------------------------------------------------------------
# MODEL 4: TIME-SERIES TRAJECTORY
# -------------------------------------------------------------
elif "Time-Series" in sel_algo:
    try:
        ts_result = analyzer.run_time_series_forecast(
            df=records,
            indicator_code=target_col,
            policy_year=2020,
            horizon_years=5,
            save_to_db=False,
        )
        m = ts_result["metrics"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Pre-Policy Velocity (≤2020)", f"{m['pre_policy_annual_velocity']:+.2f} pts/yr" if m['pre_policy_annual_velocity'] else "N/A")
        c2.metric("Post-Policy Velocity (≥2020)", f"{m['post_policy_annual_velocity']:+.2f} pts/yr" if m['post_policy_annual_velocity'] else "N/A")
        c3.metric("Average Annual Drift", f"{m['average_annual_drift']:+.2f} pts/yr")

        t_df = pd.DataFrame(ts_result["trajectory"])
        fig = go.Figure()

        h_df = t_df[t_df["type"] == "Historical"]
        fig.add_trace(go.Scatter(
            x=h_df["year"], y=h_df["actual"],
            mode="lines+markers", name="Empirical",
            line=dict(color="#1E3A8A", width=3),
            marker=dict(size=7),
        ))
        fig.add_vline(x=2020, line_dash="dash", line_color="#D97706", annotation_text="NEP 2020 Enactment", annotation_position="top left")

        f_df = t_df[t_df["type"] == "Forecast"]
        fig.add_trace(go.Scatter(
            x=f_df["year"], y=f_df["projected"],
            mode="lines+markers", name="5-Year Forecast",
            line=dict(color="#10B981", width=2, dash="dot"),
        ))

        fig.update_layout(
            title=f"<b>Time-Series Velocity & Forecast: {sel_ind_label}</b>",
            xaxis_title="Academic Year",
            yaxis_title=f"{sel_ind_label} ({target_unit})",
            template="plotly_white",
            height=380,
            margin=dict(l=30, r=20, t=40, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error in Time-Series: {e}")

render_academic_disclaimer()
