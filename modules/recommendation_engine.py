"""
Evidence-Based Recommendation Engine.
Module 5.
Generates prioritized, explainable diagnostic guidance linked to empirical policy gaps.
Maintains strict academic integrity by separating correlation from causation.
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from database.db_connection import save_recommendations_batch


class PolicyRecommendationEngine:
    """
    Synthesizes empirical findings from Impact Analysis and Risk Detection
    into structured, prioritized policy guidance with diagnostic action pathways.
    """

    def __init__(self):
        pass

    def generate_recommendations(
        self,
        risk_df: pd.DataFrame,
        education_df: pd.DataFrame,
        save_to_db: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Generate evidence-based recommendations for flagged states and districts.
        """
        if risk_df.empty or education_df.empty:
            return []

        recent_year = int(education_df["year"].max())
        recommendations: List[Dict[str, Any]] = []

        # Focus primarily on High and Moderate risk regions
        flagged_regions = risk_df[risk_df["risk_level"].isin(["High Risk", "Moderate Risk"])]

        for _, r_row in flagged_regions.iterrows():
            reg_name = r_row["region_name"]
            reg_type = r_row.get("region_type", "state")
            r_tier = r_row["risk_level"]
            score = r_row.get("risk_score", 0.0)

            # Retrieve actual latest indicators for this region
            col_filter = "state" if reg_type == "state" else "district"
            sub = education_df[(education_df[col_filter] == reg_name) & (education_df["year"] == recent_year)]
            if sub.empty:
                continue

            row = sub.iloc[0]
            ger = row.get("ger", 85.0)
            drop = row.get("dropout_rate", 10.0)
            ptr = row.get("pupil_teacher_ratio", 25.0)
            elec = row.get("electricity_availability", 90.0)
            water = row.get("drinking_water_availability", 95.0)
            toilet = row.get("toilet_availability", 90.0)
            internet = row.get("internet_availability", 45.0)

            # Rule 1: High Dropout Rate / Low Retention
            if drop >= 14.0:
                prio = "Critical" if drop >= 18.0 else "High"
                recommendations.append({
                    "title": f"Secondary School Transition & Retention Taskforce ({reg_name})",
                    "identified_problem": (
                        f"Secondary school dropout rate in {reg_name} stands at {drop:.1f}%, "
                        f"significantly exceeding the NEP 2020 zero-dropout target."
                    ),
                    "affected_region": reg_name,
                    "related_indicator": "Secondary Dropout Rate / Transition Rate",
                    "supporting_data": f"Observed Dropout: {drop:.1f}% | Year: {recent_year} | National Alert Ceiling: 12.0%",
                    "reason": (
                        "High attrition between elementary and secondary grades threatens statutory universal "
                        "participation goals and impedes higher education transition."
                    ),
                    "priority_level": prio,
                    "suggested_investigation": (
                        "Audit spatial distribution of secondary schools within a 5 km radius; "
                        "evaluate availability of school transport vouchers and bicycle schemes; "
                        "examine seasonal agrarian migration patterns and attendance registers."
                    ),
                    "is_approved": 1,
                })

            # Rule 2: High Pupil-Teacher Ratio (PTR)
            if ptr > 30.0:
                prio = "Critical" if ptr > 35.0 else "High"
                recommendations.append({
                    "title": f"Teacher Rationalization and Deployment Initiative ({reg_name})",
                    "identified_problem": (
                        f"Pupil-Teacher Ratio in {reg_name} is {ptr:.1f}:1, violating the statutory "
                        f"NEP 2020 norm of 30:1 (and 25:1 in disadvantaged SEDG areas)."
                    ),
                    "affected_region": reg_name,
                    "related_indicator": "Pupil-Teacher Ratio (PTR)",
                    "supporting_data": f"Observed PTR: {ptr:.1f}:1 | Statutory Norm: ≤30:1 | Year: {recent_year}",
                    "reason": (
                        "Excessive class sizes compromise foundational numeracy/literacy instruction "
                        "and diminish individual learning attention."
                    ),
                    "priority_level": prio,
                    "suggested_investigation": (
                        "Conduct school-level vacancy audits to identify single-teacher schools; "
                        "evaluate subject-specific teacher deficits (STEM and languages); "
                        "implement transparent digital teacher transfer systems to balance rural-urban staffing."
                    ),
                    "is_approved": 1,
                })

            # Rule 3: Digital Connectivity Deficit
            if internet < 50.0:
                prio = "High" if internet < 35.0 else "Medium"
                recommendations.append({
                    "title": f"Digital School Infrastructure & ICT Enablement Mission ({reg_name})",
                    "identified_problem": (
                        f"Broadband and digital connectivity is available in only {internet:.1f}% of schools "
                        f"in {reg_name}, creating a severe educational digital divide."
                    ),
                    "affected_region": reg_name,
                    "related_indicator": "School Internet & Computing Facilities",
                    "supporting_data": f"Internet Coverage: {internet:.1f}% | NEP 2020 Target: 100% | Year: {recent_year}",
                    "reason": (
                        "Digital infrastructure deficits obstruct ICT-enabled pedagogy, digital textbook access, "
                        "and modern vocational training."
                    ),
                    "priority_level": prio,
                    "suggested_investigation": (
                        "Coordinate with BharatNet and regional electricity distribution companies (DISCOMs) "
                        "to ensure uninterrupted power before laying fiber; prioritize secondary schools "
                        "for centralized computer laboratories."
                    ),
                    "is_approved": 1,
                })

            # Rule 4: Basic Infrastructure (Water / Sanitation)
            if toilet < 92.0 or water < 92.0:
                recommendations.append({
                    "title": f"WASH Infrastructure & Sanitation Parity Drive ({reg_name})",
                    "identified_problem": (
                        f"Functional toilet availability ({toilet:.1f}%) and safe water access ({water:.1f}%) "
                        f"fall short of universal 100% statutory standards in {reg_name}."
                    ),
                    "affected_region": reg_name,
                    "related_indicator": "Functional Separate Toilets & Drinking Water",
                    "supporting_data": f"Toilet Coverage: {toilet:.1f}%, Water Coverage: {water:.1f}% | Year: {recent_year}",
                    "reason": (
                        "Substandard WASH facilities disproportionately impact female student retention "
                        "and increase health-related school absenteeism."
                    ),
                    "priority_level": "High" if toilet < 85.0 else "Medium",
                    "suggested_investigation": (
                        "Review Samagra Shiksha civil works fund utilization; verify availability of dedicated "
                        "running water connections to girls' toilet complexes in rural schools."
                    ),
                    "is_approved": 1,
                })

        if save_to_db and recommendations:
            save_recommendations_batch(recommendations)

        return recommendations
