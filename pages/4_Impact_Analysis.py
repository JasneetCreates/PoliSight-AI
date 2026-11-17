"""
Page 4: Impact Analysis (Machine Learning)
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
    get_available_filters,
    get_model_results,
)
from modules.impact_analyzer import ImpactAnalyzer
from modules.indicator_matcher import IndicatorMatcher
from utils.helpers import (
    apply_custom_css,
    render_academic_disclaimer,
    render_kpi,
    format_percentage,
)
from utils.visualizations import (
    plot_indicator_trend_with_target,
    plot_state_comparison_bar,
    plot_clusters,
)

apply_custom_css()

st.markdown('<div class="main-header">Module 3: Impact Analysis (Machine Learning)</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Evaluating empirical trajectories, regression trends, regional clusters, and target gap estimation</div>',
    unsafe_allow_html=True,
)

render_academic_disclaimer()

analyzer = ImpactAnalyzer()
matcher = IndicatorMatcher()

policies = get_all_policies()
if not policies:
    st.warning("No policies registered. Ingest a policy document on Page 2.")
    st.stop()

active_policy = policies[0]
policy_indicators = get_policy_objectives_with_indicators(active_policy["id"])
records = query_education_records()

if records.empty:
    st.info("No education records found in the database. Ingest UDISE+ data on Page 3.")
    st.stop()

# Indicator & Model Selectors
col1, col2 = st.columns([2, 1])
with col1:
    ind_options = [
        ("Gross Enrolment Ratio (GER)", "ger"),
        ("Secondary Dropout Rate", "dropout_rate"),
        ("Pupil-Teacher Ratio (PTR)", "pupil_teacher_ratio"),
        ("Transition Rate to Secondary", "transition_rate"),
        ("School Retention Rate", "retention_rate"),
        ("School Electricity Coverage", "electricity_availability"),
        ("Drinking Water Availability", "drinking_water_availability"),
        ("Separate Functional Toilets", "toilet_availability"),
        ("Internet & Digital Facilities", "internet_availability"),
        ("Percentage of Female Teachers", "female_teacher_percentage"),
    ]
    avail_ind_options = [opt for opt in ind_options if opt[1] in records.columns]
    sel_ind_tuple = st.selectbox(
        "Select Policy Indicator for Evaluation",
        options=avail_ind_options,
        format_func=lambda x: x[0],
    )
    sel_ind_label, target_col = sel_ind_tuple

with col2:
    analysis_algo = st.selectbox("Select ML / Statistical Technique", options=[
        "1. Linear Regression (Velocity & Projections)",
        "2. Random Forest Regression (Feature Importance)",
        "3. K-Means Regional Clustering",
        "4. Time-Series Trajectory Analysis",
    ])

# Fetch statutory target metadata if mapped
target_meta = next((o for o in policy_indicators if matcher.match_objective_to_indicator(o) == target_col), {})
target_val = target_meta.get("target_value")
target_year = target_meta.get("target_year", 2030)
target_unit = target_meta.get("target_unit", "%")

st.markdown("---")

# -------------------------------------------------------------
# ALGORITHM 1: LINEAR REGRESSION
# -------------------------------------------------------------
if "Linear Regression" in analysis_algo:
    st.markdown("### 📈 Algorithm 1: Linear Regression (Temporal Velocity & Projections)")
    st.write(
        "Fits Ordinary Least Squares regression across academic years to establish historical velocity, "
        "evaluate goodness-of-fit with chronological holdout, and project milestones toward policy target years."
    )

    try:
        reg_result = analyzer.run_linear_regression_analysis(
            df=records,
            indicator_code=target_col,
            target_year=int(target_year) if target_year else 2030,
            target_val=float(target_val) if target_val is not None else None,
            save_to_db=True,
        )
        m = reg_result["metrics"]

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Annual Velocity (Slope)", f"{m['slope_annual_velocity']:+.2f} pts/yr")
        col_m2.metric("Goodness of Fit (R²)", f"{m['r2_score']:.3f}")
        col_m3.metric("Root Mean Squared Error", f"{m['rmse']:.2f}")
        col_m4.metric("Mean Absolute Error (MAE)", f"{m['mae']:.2f}")

        # Milestone projection highlight
        p_yr = m["projected_target_year"]
        p_val = m["projected_value"]
        p_gap = m["projected_gap"]
        
        if p_gap is not None:
            alert_type = st.success if p_gap <= 0 else st.warning
            alert_type(
                f"**Milestone Projection for Year {p_yr}:** Estimated at **{p_val:.1f}{target_unit}**. "
                f"Target: **{target_val}{target_unit}** | Status: **{m['target_status']}**"
            )

        # Plotly Trajectory Chart
        traj_df = pd.DataFrame(reg_result["trajectory"])
        fig = go.Figure()

        # Historical points
        hist_df = traj_df[traj_df["type"] == "Historical Observation"]
        fig.add_trace(go.Scatter(
            x=hist_df["year"], y=hist_df["actual"],
            mode="markers+lines", name="Observed UDISE+ Data",
            line=dict(color="#1E3A8A", width=3),
            marker=dict(size=8, color="#1E3A8A"),
        ))

        # Fitted regression line
        fig.add_trace(go.Scatter(
            x=hist_df["year"], y=hist_df["fitted"],
            mode="lines", name="Fitted Regression Trend",
            line=dict(color="#3B82F6", width=2, dash="dash"),
        ))

        # Model projection points
        proj_df = traj_df[traj_df["type"] == "Model Projection"]
        if not proj_df.empty:
            fig.add_trace(go.Scatter(
                x=proj_df["year"], y=proj_df["fitted"],
                mode="lines+markers", name=f"Projected Trajectory (to {p_yr})",
                line=dict(color="#10B981", width=2, dash="dot"),
                marker=dict(size=6, color="#10B981"),
            ))

        # Target line
        if target_val is not None:
            fig.add_hline(
                y=target_val, line_dash="solid", line_color="#DC2626", line_width=2,
                annotation_text=f"Statutory Policy Target ({target_val}{target_unit} by {target_year})",
                annotation_position="top left",
            )

        fig.update_layout(
            title=f"<b>Longitudinal Velocity & Policy Target Trajectory: {sel_ind_label}</b>",
            xaxis_title="Academic Year",
            yaxis_title=f"{sel_ind_label} ({target_unit})",
            template="plotly_white",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error executing regression analysis: {e}")

# -------------------------------------------------------------
# ALGORITHM 2: RANDOM FOREST
# -------------------------------------------------------------
elif "Random Forest" in analysis_algo:
    st.markdown("### 🌲 Algorithm 2: Random Forest Multivariate Feature Association")
    st.write(
        "Trains an ensemble of 100 decision trees to evaluate the relative predictive importance "
        "of infrastructure facilities and teacher availability in accounting for variations in the target outcome."
    )

    try:
        rf_result = analyzer.run_random_forest_analysis(
            df=records,
            target_col=target_col,
            save_to_db=True,
        )
        m = rf_result["metrics"]

        # Methodological notice
        st.info(f"📌 **{rf_result['causal_disclaimer']}**")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cross-Validation Folds", m["cross_val_folds"])
        c2.metric("Out-of-Fold R²", f"{m['mean_r2']:.3f}")
        c3.metric("Cross-Val RMSE", f"{m['mean_rmse']:.2f}")
        c4.metric("Samples Evaluated", m["total_samples"])

        # Feature Importance Bar Chart
        fi_df = pd.DataFrame(rf_result["feature_importances"]).sort_values(by="importance", ascending=True)
        fig = px.bar(
            fi_df,
            x="importance",
            y="feature",
            error_x="std",
            orientation="h",
            labels={"importance": "Relative Predictive Importance (Gini Importance)", "feature": "Predictor Variable"},
            title=f"<b>Random Forest Predictive Weight Distribution on Outcome: {sel_ind_label}</b>",
            template="plotly_white",
            color="importance",
            color_continuous_scale="Blues",
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(pd.DataFrame(rf_result["feature_importances"]), use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error executing Random Forest analysis: {e}")

# -------------------------------------------------------------
# ALGORITHM 3: K-MEANS REGIONAL CLUSTERING
# -------------------------------------------------------------
elif "K-Means" in analysis_algo:
    st.markdown("### 🎯 Algorithm 3: K-Means Regional Clustering")
    st.write(
        "Segments Indian States into performance cohorts based on educational outcomes. "
        "Validates cluster separation via the **Silhouette Score** and **Davies-Bouldin Index**."
    )

    try:
        cluster_result = analyzer.run_clustering_analysis(
            df=records,
            feature_cols=["ger", "dropout_rate", "pupil_teacher_ratio"],
            region_col="state",
            save_to_db=True,
        )

        opt_k = cluster_result["optimal_k"]
        sil_score = cluster_result["best_silhouette_score"]

        col_k1, col_k2, col_k3 = st.columns(3)
        col_k1.metric("Optimal Cluster Count (k)", opt_k, "Selected via validation")
        col_k2.metric("Silhouette Coefficient", f"{sil_score:.3f}", "Cluster separation quality")
        col_k3.metric("Features Clustered", "GER, Dropout, PTR", "3-dimensional performance vector")

        # Cluster scatter plot
        clustered_df = pd.DataFrame(cluster_result["regional_clusters"])
        fig = px.scatter(
            clustered_df,
            x="ger",
            y="dropout_rate",
            color="performance_cohort",
            hover_name="state",
            size="pupil_teacher_ratio",
            labels={
                "ger": "Gross Enrolment Ratio (%)",
                "dropout_rate": "Secondary Dropout Rate (%)",
                "pupil_teacher_ratio": "Pupil-Teacher Ratio",
                "performance_cohort": "Performance Tier",
            },
            title="<b>Regional Performance Cohorts (GER vs Dropout Rate vs PTR)</b>",
            template="plotly_white",
        )
        fig.update_traces(marker=dict(line=dict(width=1, color="DarkSlateGrey")))
        st.plotly_chart(fig, use_container_width=True)

        # Centroids table
        st.markdown("#### Performance Cohort Profiles & Centroids:")
        st.dataframe(pd.DataFrame(cluster_result["cluster_centroids"]), use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error executing clustering analysis: {e}")

# -------------------------------------------------------------
# ALGORITHM 4: TIME-SERIES TRAJECTORY
# -------------------------------------------------------------
elif "Time-Series" in analysis_algo:
    st.markdown("### ⏱️ Algorithm 4: Time-Series Trajectory Analysis")
    st.write(
        "Examines annual fluctuations, measures pre-NEP vs post-NEP annual velocity, "
        "and projects forward drift with statistical confidence bounds."
    )

    try:
        ts_result = analyzer.run_time_series_forecast(
            df=records,
            indicator_code=target_col,
            policy_year=2020,
            horizon_years=5,
            save_to_db=True,
        )
        m = ts_result["metrics"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pre-Policy Velocity (≤2020)", f"{m['pre_policy_annual_velocity']:+.2f} pts/yr" if m['pre_policy_annual_velocity'] else "N/A")
        c2.metric("Post-Policy Velocity (≥2020)", f"{m['post_policy_annual_velocity']:+.2f} pts/yr" if m['post_policy_annual_velocity'] else "N/A")
        c3.metric("Annual Drift Mean", f"{m['average_annual_drift']:+.2f} pts/yr")
        c4.metric("Validation RMSE", f"{m['validation_rmse']:.2f}")

        # Trajectory chart with bounds
        t_df = pd.DataFrame(ts_result["trajectory"])
        fig = go.Figure()

        # Historical
        h_df = t_df[t_df["type"] == "Historical"]
        fig.add_trace(go.Scatter(
            x=h_df["year"], y=h_df["actual"],
            mode="lines+markers", name="Empirical Observations",
            line=dict(color="#1E3A8A", width=3),
            marker=dict(size=7),
        ))

        # Policy marker vertical line
        fig.add_vline(x=2020, line_dash="dash", line_color="#F59E0B", annotation_text="NEP 2020 Enactment", annotation_position="top left")

        # Forecast
        f_df = t_df[t_df["type"] == "Forecast"]
        fig.add_trace(go.Scatter(
            x=f_df["year"], y=f_df["projected"],
            mode="lines+markers", name="Projected Drift",
            line=dict(color="#10B981", width=2, dash="dot"),
        ))

        # Upper & Lower confidence bounds
        fig.add_trace(go.Scatter(
            x=list(f_df["year"]) + list(f_df["year"][::-1]),
            y=list(f_df["upper_bound"]) + list(f_df["lower_bound"][::-1]),
            fill="toself",
            fillcolor="rgba(16, 185, 129, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="90% Confidence Interval",
        ))

        fig.update_layout(
            title=f"<b>Time-Series Velocity & 5-Year Forecast: {sel_ind_label}</b>",
            xaxis_title="Academic Year",
            yaxis_title=f"{sel_ind_label} ({target_unit})",
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error executing time-series analysis: {e}")

st.markdown("---")

# State-by-State Comparison Section
st.markdown(f"### Regional Performance Distribution: {sel_ind_label}")
recent_year = int(records["year"].max())
latest_st = records[records["year"] == recent_year].dropna(subset=[target_col])

fig_bar = plot_state_comparison_bar(
    df=latest_st,
    state_col="state",
    val_col=target_col,
    indicator_title=sel_ind_label,
    target_val=target_val,
    unit=target_unit,
    top_n=20,
)
st.plotly_chart(fig_bar, use_container_width=True)
