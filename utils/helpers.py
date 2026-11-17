"""
Utility helpers for UI styling, formatting, session state, and academic annotations.
"""

import streamlit as st
from typing import Any, Dict, Optional


def apply_custom_css():
    """Apply clean, academic styling to Streamlit app."""
    st.markdown(
        """
        <style>
        /* General layout and typography */
        .main-header {
            font-size: 2.1rem;
            font-weight: 700;
            color: #1E3A8A;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #475569;
            margin-bottom: 1.5rem;
        }
        
        /* Metric cards */
        .kpi-card {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-left: 4px solid #2563EB;
            padding: 1rem 1.2rem;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            margin-bottom: 0.8rem;
        }
        .kpi-title {
            font-size: 0.85rem;
            text-transform: uppercase;
            font-weight: 600;
            color: #64748B;
            letter-spacing: 0.05em;
        }
        .kpi-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #0F172A;
            margin: 0.2rem 0;
        }
        .kpi-desc {
            font-size: 0.8rem;
            color: #64748B;
        }

        /* Research Disclaimers & Alerts */
        .academic-alert {
            background-color: #EFF6FF;
            border-left: 4px solid #3B82F6;
            padding: 0.8rem 1rem;
            border-radius: 4px;
            font-size: 0.9rem;
            color: #1E40AF;
            margin-bottom: 1rem;
        }
        .warning-alert {
            background-color: #FEF3C7;
            border-left: 4px solid #F59E0B;
            padding: 0.8rem 1rem;
            border-radius: 4px;
            font-size: 0.9rem;
            color: #92400E;
            margin-bottom: 1rem;
        }
        .risk-badge-high {
            background-color: #FEE2E2;
            color: #991B1B;
            padding: 3px 8px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 0.8rem;
            display: inline-block;
        }
        .risk-badge-mod {
            background-color: #FEF3C7;
            color: #92400E;
            padding: 3px 8px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 0.8rem;
            display: inline-block;
        }
        .risk-badge-low {
            background-color: #DCFCE7;
            color: #166534;
            padding: 3px 8px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 0.8rem;
            display: inline-block;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_academic_disclaimer():
    """Render scientific caveat separating correlation from causation."""
    st.markdown(
        """
        <div class="academic-alert">
            <strong>Research Methodology Note:</strong> Observed changes across educational indicators 
            represent empirical trends over time. In accordance with rigorous policy evaluation methodology, 
            observed post-2020 progress should not be conflated with direct causal attribution of NEP 2020 
            without controlled causal inference (e.g., Difference-in-Differences / Synthetic Control).
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi(title: str, value: Any, desc: str = "", border_color: str = "#2563EB"):
    """Render a styled KPI card."""
    st.markdown(
        f"""
        <div class="kpi-card" style="border-left-color: {border_color};">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-desc">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_percentage(val: Optional[float], decimals: int = 1) -> str:
    """Format floating point number as percentage."""
    if val is None or val != val:
        return "N/A"
    return f"{val:.{decimals}f}%"


def format_ratio(val: Optional[float], decimals: int = 1) -> str:
    """Format ratio value."""
    if val is None or val != val:
        return "N/A"
    return f"{val:.{decimals}f}:1"
