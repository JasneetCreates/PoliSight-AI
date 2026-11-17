"""
Interactive Plotly Visualizations for Policy-to-Impact Intelligence System.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_target_progress_gauge(
    current_value: float,
    target_value: float,
    indicator_name: str,
    unit: str = "%",
    is_higher_desirable: bool = True,
) -> go.Figure:
    """Create a gauge chart showing progress toward a numerical policy target."""
    max_val = max(100.0, target_value * 1.15, current_value * 1.15)
    
    # Color logic
    if is_higher_desirable:
        bar_color = "#10B981" if current_value >= target_value else "#3B82F6"
    else:
        bar_color = "#10B981" if current_value <= target_value else "#EF4444"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=current_value,
            number={"suffix": unit, "font": {"size": 28, "color": "#1E293B"}},
            delta={
                "reference": target_value,
                "increasing": {"color": "#10B981" if is_higher_desirable else "#EF4444"},
                "decreasing": {"color": "#EF4444" if is_higher_desirable else "#10B981"},
            },
            title={"text": f"<b>{indicator_name}</b><br><span style='font-size:0.8em;color:gray'>Target: {target_value}{unit}</span>", "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, max_val], "tickwidth": 1, "tickcolor": "#94A3B8"},
                "bar": {"color": bar_color},
                "bgcolor": "#F1F5F9",
                "borderwidth": 1,
                "bordercolor": "#CBD5E1",
                "threshold": {
                    "line": {"color": "#DC2626", "width": 4},
                    "thickness": 0.8,
                    "value": target_value,
                },
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def plot_indicator_trend_with_target(
    df: pd.DataFrame,
    year_col: str,
    val_col: str,
    target_val: Optional[float] = None,
    target_year: Optional[int] = None,
    indicator_title: str = "Indicator",
    unit: str = "%",
    group_col: Optional[str] = None,
) -> go.Figure:
    """Plot multi-year trend line with target benchmark."""
    fig = go.Figure()

    if group_col and group_col in df.columns:
        groups = df[group_col].unique()
        for g in groups:
            sub = df[df[group_col] == g].sort_values(by=year_col)
            fig.add_trace(
                go.Scatter(
                    x=sub[year_col],
                    y=sub[val_col],
                    mode="lines+markers",
                    name=str(g),
                    line=dict(width=2),
                )
            )
    else:
        df_sorted = df.sort_values(by=year_col)
        fig.add_trace(
            go.Scatter(
                x=df_sorted[year_col],
                y=df_sorted[val_col],
                mode="lines+markers",
                name="Observed Data",
                line=dict(color="#2563EB", width=3),
                marker=dict(size=7),
            )
        )

    # Policy target horizontal or milestone line
    if target_val is not None:
        fig.add_hline(
            y=target_val,
            line_dash="dash",
            line_color="#DC2626",
            annotation_text=f"Policy Target ({target_val}{unit}{f' by {target_year}' if target_year else ''})",
            annotation_position="top right",
        )

    fig.update_layout(
        title=f"<b>Historical Trajectory: {indicator_title}</b>",
        xaxis_title="Academic Year",
        yaxis_title=f"{indicator_title} ({unit})",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_state_comparison_bar(
    df: pd.DataFrame,
    state_col: str,
    val_col: str,
    indicator_title: str,
    target_val: Optional[float] = None,
    unit: str = "%",
    top_n: int = 15,
) -> go.Figure:
    """Plot horizontal state-wise comparison bar chart."""
    sorted_df = df.sort_values(by=val_col, ascending=True).tail(top_n)

    fig = px.bar(
        sorted_df,
        x=val_col,
        y=state_col,
        orientation="h",
        text=val_col,
        color=val_col,
        color_continuous_scale="Blues",
        labels={val_col: f"{indicator_title} ({unit})", state_col: "State / UT"},
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")

    if target_val is not None:
        fig.add_vline(
            x=target_val,
            line_dash="dot",
            line_color="#DC2626",
            annotation_text=f"Target: {target_val}{unit}",
            annotation_position="top left",
        )

    fig.update_layout(
        title=f"<b>State Performance: {indicator_title}</b>",
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40),
        coloraxis_showscale=False,
    )
    return fig


def plot_clusters(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    cluster_col: str,
    label_col: str,
    x_label: str,
    y_label: str,
) -> go.Figure:
    """Plot K-Means regional clusters scatter plot."""
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=cluster_col,
        hover_name=label_col,
        labels={x_col: x_label, y_col: y_label, cluster_col: "Cluster Group"},
        template="plotly_white",
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="DarkSlateGrey")))
    fig.update_layout(
        title=f"<b>Regional Performance Clusters ({x_label} vs {y_label})</b>",
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def plot_risk_distribution(risk_df: pd.DataFrame) -> go.Figure:
    """Plot doughnut chart of regional risk tiers."""
    counts = risk_df["risk_level"].value_counts().reset_index()
    counts.columns = ["Risk Level", "Count"]

    color_map = {
        "High Risk": "#EF4444",
        "Moderate Risk": "#F59E0B",
        "Low Risk": "#10B981",
    }

    fig = px.pie(
        counts,
        names="Risk Level",
        values="Count",
        hole=0.45,
        color="Risk Level",
        color_discrete_map=color_map,
    )
    fig.update_layout(
        title="<b>Regional Risk Level Distribution</b>",
        margin=dict(l=30, r=30, t=40, b=30),
    )
    return fig
