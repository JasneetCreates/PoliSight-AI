"""
Utility helpers for UI styling, formatting, session state, and academic annotations.
Clean, modern, and uncluttered design system.
"""

import streamlit as st
from typing import Any, Dict, Optional


def apply_custom_css():
    """Apply clean, modern, minimalist SaaS/Academic styling to Streamlit app."""
    st.markdown(
        """
        <style>
        /* Base page styling */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 3rem !important;
            max-width: 1200px !important;
        }

        /* Clean modern headers */
        .main-header {
            font-size: 1.85rem;
            font-weight: 700;
            color: #0F172A;
            letter-spacing: -0.02em;
            margin-bottom: 0.25rem;
        }
        .sub-header {
            font-size: 0.95rem;
            color: #64748B;
            margin-bottom: 1.5rem;
            font-weight: 400;
        }

        /* KPI Metric cards */
        .kpi-container {
            display: flex;
            gap: 1rem;
            margin-bottom: 1.5rem;
        }
        .kpi-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 1rem 1.25rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
            transition: all 0.2s ease;
        }
        .kpi-card:hover {
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
            border-color: #CBD5E1;
        }
        .kpi-title {
            font-size: 0.75rem;
            text-transform: uppercase;
            font-weight: 600;
            color: #64748B;
            letter-spacing: 0.05em;
            margin-bottom: 0.25rem;
        }
        .kpi-value {
            font-size: 1.75rem;
            font-weight: 700;
            color: #0F172A;
            line-height: 1.2;
        }
        .kpi-desc {
            font-size: 0.8rem;
            color: #94A3B8;
            margin-top: 0.25rem;
        }

        /* Badges */
        .badge-pill {
            display: inline-block;
            padding: 0.2rem 0.65rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .badge-blue { background-color: #EFF6FF; color: #1D4ED8; }
        .badge-green { background-color: #ECFDF5; color: #047857; }
        .badge-amber { background-color: #FFFBEB; color: #B45309; }
        .badge-red { background-color: #FEF2F2; color: #B91C1C; }

        /* Minimal step pill */
        .step-pill {
            display: inline-flex;
            align-items: center;
            background: #F1F5F9;
            color: #334155;
            padding: 0.25rem 0.75rem;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }

        /* Clean cards for action sections */
        .action-card {
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 1.25rem;
            margin-bottom: 1rem;
        }

        /* Footer disclaimer */
        .academic-footer {
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid #E2E8F0;
            font-size: 0.78rem;
            color: #94A3B8;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_academic_disclaimer():
    """Render subtle, clean academic methodology footer rather than a giant in-your-face banner."""
    st.markdown(
        """
        <div class="academic-footer">
            ⚖️ <strong>Academic Methodology:</strong> Empirical indicator trends evaluate longitudinal progress. 
            Observed improvements are not conflated with direct causal attribution of NEP 2020 without controlled econometric inference.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi(title: str, value: Any, desc: str = "", border_color: str = "#2563EB"):
    """Render a clean styled KPI card."""
    st.markdown(
        f"""
        <div class="kpi-card" style="border-top: 3px solid {border_color};">
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
