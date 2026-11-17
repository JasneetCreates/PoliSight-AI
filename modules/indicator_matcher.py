"""
Policy Indicator Matcher.
Bridges textual policy objectives (Module 1) with empirical indicators in education datasets (Module 2).
Calculates directional target gaps, percentage velocities, and baseline comparisons.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Core Indicator Ontology with Directional Semantics
INDICATOR_METRIC_CONFIG = {
    "ger": {
        "display_name": "Gross Enrolment Ratio (GER)",
        "unit": "%",
        "is_higher_better": True,
        "default_target": 100.0,
        "default_target_year": 2030,
        "keywords": ["ger", "enrolment", "universal access", "participation"],
    },
    "ger_ecce": {
        "display_name": "ECCE / Preschool Enrolment Ratio",
        "unit": "%",
        "is_higher_better": True,
        "default_target": 100.0,
        "default_target_year": 2030,
        "keywords": ["ecce", "preschool", "balvatika", "early childhood"],
    },
    "dropout_rate": {
        "display_name": "Secondary Dropout Rate",
        "unit": "%",
        "is_higher_better": False,  # Lower is desirable
        "default_target": 0.0,
        "default_target_year": 2030,
        "keywords": ["dropout", "curtailing dropout", "retention"],
    },
    "pupil_teacher_ratio": {
        "display_name": "Pupil-Teacher Ratio (PTR)",
        "unit": "ratio",
        "is_higher_better": False,  # Lower is desirable (statutory ceiling 30:1)
        "default_target": 30.0,
        "default_target_year": 2025,
        "keywords": ["ptr", "pupil-teacher", "teacher"],
    },
    "ptr_sedg": {
        "display_name": "PTR (Disadvantaged / SEDG Areas)",
        "unit": "ratio",
        "is_higher_better": False,
        "default_target": 25.0,
        "default_target_year": 2025,
        "keywords": ["ptr 25:1", "special education zone", "sedg ptr"],
    },
    "transition_rate": {
        "display_name": "Transition Rate to Secondary",
        "unit": "%",
        "is_higher_better": True,
        "default_target": 100.0,
        "default_target_year": 2030,
        "keywords": ["transition", "continuity"],
    },
    "retention_rate": {
        "display_name": "School Retention Rate",
        "unit": "%",
        "is_higher_better": True,
        "default_target": 100.0,
        "default_target_year": 2030,
        "keywords": ["retention", "completion"],
    },
    "electricity_availability": {
        "display_name": "School Electricity Coverage",
        "unit": "%",
        "is_higher_better": True,
        "default_target": 100.0,
        "default_target_year": 2025,
        "keywords": ["electricity", "power"],
    },
    "drinking_water_availability": {
        "display_name": "Drinking Water Availability",
        "unit": "%",
        "is_higher_better": True,
        "default_target": 100.0,
        "default_target_year": 2025,
        "keywords": ["drinking water", "water facility", "potable water"],
    },
    "toilet_availability": {
        "display_name": "Separate Functional Toilets",
        "unit": "%",
        "is_higher_better": True,
        "default_target": 100.0,
        "default_target_year": 2025,
        "keywords": ["toilet", "sanitation", "wash"],
    },
    "internet_availability": {
        "display_name": "Internet & Digital Facilities",
        "unit": "%",
        "is_higher_better": True,
        "default_target": 100.0,
        "default_target_year": 2025,
        "keywords": ["internet", "ict", "computing", "digital"],
    },
    "female_teacher_percentage": {
        "display_name": "Percentage of Female Teachers",
        "unit": "%",
        "is_higher_better": True,
        "default_target": 50.0,
        "default_target_year": 2030,
        "keywords": ["female teacher", "women teachers", "gender parity"],
    },
}


class IndicatorMatcher:
    """Matches policy objectives with empirical dataset columns and computes gaps."""

    def __init__(self):
        self.config = INDICATOR_METRIC_CONFIG

    def match_objective_to_indicator(self, objective: Dict[str, Any]) -> str:
        """Find best dataset indicator column code matching a policy objective record."""
        # 1. Direct code if present
        ind_code = objective.get("indicator_code")
        if ind_code and ind_code in self.config:
            return ind_code

        # 2. Textual search in indicator name / objective title
        search_corpus = f"{objective.get('indicator_name', '')} {objective.get('objective_title', '')} {objective.get('expected_outcome', '')}".lower()

        for code, conf in self.config.items():
            if any(kw in search_corpus for kw in conf["keywords"]):
                return code

        return "ger"  # Default fallback

    def calculate_target_gap(self, actual_val: float, target_val: float, indicator_code: str) -> float:
        """
        Calculate gap to policy target with directional interpretation.
        - For indicators where HIGHER is better (e.g. GER, Water): Gap = Target - Actual (positive means deficit)
        - For indicators where LOWER is better (e.g. Dropout, PTR): Gap = Actual - Target (positive means excess above target)
        """
        conf = self.config.get(indicator_code, {"is_higher_better": True})
        is_higher = conf.get("is_higher_better", True)

        if is_higher:
            return round(target_val - actual_val, 2)
        else:
            return round(actual_val - target_val, 2)

    def calculate_percentage_change(self, baseline_val: float, current_val: float) -> Optional[float]:
        """
        Calculate empirical percentage change:
        Percentage Change = ((Current - Previous) / Previous) * 100
        """
        if baseline_val == 0 or pd.isna(baseline_val) or pd.isna(current_val):
            return None
        return round(((current_val - baseline_val) / abs(baseline_val)) * 100, 2)

    def evaluate_indicator_progress(
        self,
        df: pd.DataFrame,
        indicator_code: str,
        target_val: Optional[float] = None,
        target_year: Optional[int] = None,
        group_by: Optional[str] = "state",
    ) -> pd.DataFrame:
        """
        Calculate multi-year progress, annual percentage change, and target gap across regions.
        """
        if indicator_code not in df.columns:
            return pd.DataFrame()

        conf = self.config.get(indicator_code, {"is_higher_better": True, "default_target": 100.0, "default_target_year": 2030})
        t_val = target_val if target_val is not None else conf["default_target"]
        t_yr = target_year if target_year is not None else conf["default_target_year"]

        min_year = df["year"].min()
        max_year = df["year"].max()

        results = []

        if group_by and group_by in df.columns:
            regions = df[group_by].dropna().unique()
            for r in regions:
                sub = df[df[group_by] == r].sort_values(by="year")
                base_series = sub[sub["year"] == min_year][indicator_code].dropna()
                curr_series = sub[sub["year"] == max_year][indicator_code].dropna()

                if not base_series.empty and not curr_series.empty:
                    base_val = float(base_series.iloc[0])
                    curr_val = float(curr_series.iloc[0])
                    pct_change = self.calculate_percentage_change(base_val, curr_val)
                    gap = self.calculate_target_gap(curr_val, t_val, indicator_code)

                    results.append({
                        "region": r,
                        "indicator_code": indicator_code,
                        "baseline_year": int(min_year),
                        "baseline_value": round(base_val, 2),
                        "current_year": int(max_year),
                        "current_value": round(curr_val, 2),
                        "percentage_change": pct_change,
                        "target_value": t_val,
                        "target_year": t_yr,
                        "target_gap": gap,
                        "is_on_track": (gap <= 0),
                    })
        else:
            # National aggregate
            sub = df.sort_values(by="year")
            base_val = float(sub[sub["year"] == min_year][indicator_code].mean())
            curr_val = float(sub[sub["year"] == max_year][indicator_code].mean())
            pct_change = self.calculate_percentage_change(base_val, curr_val)
            gap = self.calculate_target_gap(curr_val, t_val, indicator_code)

            results.append({
                "region": "National Average",
                "indicator_code": indicator_code,
                "baseline_year": int(min_year),
                "baseline_value": round(base_val, 2),
                "current_year": int(max_year),
                "current_value": round(curr_val, 2),
                "percentage_change": pct_change,
                "target_value": t_val,
                "target_year": t_yr,
                "target_gap": gap,
                "is_on_track": (gap <= 0),
            })

        return pd.DataFrame(results)
