"""
Data Preprocessor for Education Indicators.
Handles missing values, duplicate removal, state/district standardization,
year-format normalization, outlier detection, and data quality audits.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Standard mapping for Indian States and Union Territories
STATE_NAME_MAPPING = {
    "andaman and nicobar": "Andaman and Nicobar Islands",
    "andaman & nicobar islands": "Andaman and Nicobar Islands",
    "andhra": "Andhra Pradesh",
    "andhra pradesh": "Andhra Pradesh",
    "arunachal": "Arunachal Pradesh",
    "arunachal pradesh": "Arunachal Pradesh",
    "assam": "Assam",
    "bihar": "Bihar",
    "chandigarh": "Chandigarh",
    "chhattisgarh": "Chhattisgarh",
    "chattisgarh": "Chhattisgarh",
    "dadra and nagar haveli and daman and diu": "Dadra and Nagar Haveli and Daman and Diu",
    "daman and diu": "Dadra and Nagar Haveli and Daman and Diu",
    "dadra and nagar haveli": "Dadra and Nagar Haveli and Daman and Diu",
    "delhi": "Delhi",
    "nct of delhi": "Delhi",
    "goa": "Goa",
    "gujarat": "Gujarat",
    "haryana": "Haryana",
    "himachal": "Himachal Pradesh",
    "himachal pradesh": "Himachal Pradesh",
    "jammu and kashmir": "Jammu and Kashmir",
    "jammu & kashmir": "Jammu and Kashmir",
    "j&k": "Jammu and Kashmir",
    "jharkhand": "Jharkhand",
    "karnataka": "Karnataka",
    "kerala": "Kerala",
    "ladakh": "Ladakh",
    "lakshadweep": "Lakshadweep",
    "madhya pradesh": "Madhya Pradesh",
    "mp": "Madhya Pradesh",
    "maharashtra": "Maharashtra",
    "manipur": "Manipur",
    "meghalaya": "Meghalaya",
    "mizoram": "Mizoram",
    "nagaland": "Nagaland",
    "odisha": "Odisha",
    "orissa": "Odisha",
    "puducherry": "Puducherry",
    "pondicherry": "Puducherry",
    "punjab": "Punjab",
    "rajasthan": "Rajasthan",
    "sikkim": "Sikkim",
    "tamil nadu": "Tamil Nadu",
    "tamilnadu": "Tamil Nadu",
    "telangana": "Telangana",
    "tripura": "Tripura",
    "uttar pradesh": "Uttar Pradesh",
    "up": "Uttar Pradesh",
    "uttarakhand": "Uttarakhand",
    "uttaranchal": "Uttarakhand",
    "west bengal": "West Bengal",
}

# Standard column aliases for UDISE+ and related education datasets
COLUMN_ALIASES = {
    "year": ["academic_year", "session", "ac_year", "period", "yr"],
    "state": ["state_name", "state_ut", "state/ut", "st_name", "province"],
    "district": ["district_name", "dist_name", "dtname", "dist"],
    "total_enrolment": ["enrolment", "total_enrollment", "tot_enrol", "students", "total_students"],
    "male_enrolment": ["boys_enrolment", "boys", "male_students", "enrol_boys"],
    "female_enrolment": ["girls_enrolment", "girls", "female_students", "enrol_girls"],
    "ger": ["gross_enrolment_ratio", "ger_school", "ger_total", "ger_all"],
    "dropout_rate": ["dropout", "secondary_dropout", "dr_total", "dropout_percentage"],
    "transition_rate": ["transition", "transition_to_sec", "tr_rate"],
    "retention_rate": ["retention", "retention_percentage", "rr_rate"],
    "pupil_teacher_ratio": ["ptr", "student_teacher_ratio", "pupil_teacher", "ptr_total"],
    "female_teacher_percentage": ["female_teachers_pct", "pct_female_teachers", "female_teacher_ratio"],
    "school_count": ["total_schools", "num_schools", "schools", "school_total"],
    "electricity_availability": ["electricity", "power_pct", "schools_with_electricity", "elec_pct"],
    "drinking_water_availability": ["drinking_water", "water_pct", "safe_water", "water_facility"],
    "toilet_availability": ["toilets", "functional_toilets", "separate_toilets", "toilet_pct"],
    "internet_availability": ["internet", "broadband", "connectivity", "ict_facility", "internet_pct"],
}


class EducationDataPreprocessor:
    """Preprocesses, validates, standardizes, and audits education data."""

    def __init__(self):
        pass

    def preprocess_dataset(self, df: pd.DataFrame, source_name: str = "Uploaded Dataset") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Full preprocessing pipeline:
        1. Map and harmonize column names.
        2. Normalize year representations (e.g., '2020-21' -> 2021).
        3. Standardize State & District names.
        4. Detect and handle duplicates.
        5. Convert data types and handle invalid values.
        6. Detect statistical outliers without destructive deletion.
        7. Generate a comprehensive Data Quality Report.
        """
        processed_df = df.copy()

        # Step 1: Map column names
        processed_df = self.standardize_column_names(processed_df)

        # Step 2: Normalize year
        if "year" in processed_df.columns:
            processed_df["year"] = processed_df["year"].apply(self.normalize_year)
            processed_df = processed_df.dropna(subset=["year"])
            processed_df["year"] = processed_df["year"].astype(int)

        # Step 3: Standardize State names
        if "state" in processed_df.columns:
            processed_df["state"] = processed_df["state"].apply(self.standardize_state_name)

        # Step 4: Standardize District names
        if "district" in processed_df.columns:
            processed_df["district"] = processed_df["district"].apply(self.standardize_district_name)

        # Step 5: Clean and convert numeric columns
        numeric_cols = [
            "total_enrolment", "male_enrolment", "female_enrolment", "ger",
            "dropout_rate", "transition_rate", "retention_rate", "pupil_teacher_ratio",
            "female_teacher_percentage", "school_count", "electricity_availability",
            "drinking_water_availability", "toilet_availability", "internet_availability"
        ]

        for col in numeric_cols:
            if col in processed_df.columns:
                processed_df[col] = pd.to_numeric(
                    processed_df[col].astype(str).str.replace("%", "").str.replace(",", "").str.strip(),
                    errors="coerce",
                )

        # Step 6: Deduplication
        dedup_keys = [c for c in ["year", "state", "district"] if c in processed_df.columns]
        initial_count = len(processed_df)
        if dedup_keys:
            processed_df = processed_df.drop_duplicates(subset=dedup_keys, keep="last")
        duplicate_count = initial_count - len(processed_df)

        # Step 7: Detect outliers
        outlier_summary = self.detect_outliers(processed_df, numeric_cols)

        # Step 8: Quality report
        quality_report = self.generate_quality_report(
            df_original=df,
            df_cleaned=processed_df,
            source_name=source_name,
            duplicate_count=duplicate_count,
            outliers=outlier_summary,
        )

        return processed_df, quality_report

    def standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map diverse raw headers to standardized ontology columns."""
        rename_dict = {}
        for col in df.columns:
            clean_col = re.sub(r"[^\w\s]", "", str(col)).strip().lower().replace(" ", "_")
            # Check exact match
            matched = False
            for target_col, aliases in COLUMN_ALIASES.items():
                if clean_col == target_col or clean_col in aliases:
                    rename_dict[col] = target_col
                    matched = True
                    break
            if not matched:
                # Substring match
                for target_col, aliases in COLUMN_ALIASES.items():
                    if any(alias in clean_col for alias in aliases):
                        rename_dict[col] = target_col
                        break

        return df.rename(columns=rename_dict)

    def normalize_year(self, val: Any) -> Optional[int]:
        """Convert academic year formats (e.g. '2020-21', '2020/21', '2020', 2020) to calendar year."""
        if pd.isna(val):
            return None
        val_str = str(val).strip()

        # Handle '2020-21' or '2020-2021'
        m = re.match(r"^(\d{4})[-/](\d{2,4})$", val_str)
        if m:
            start_yr = int(m.group(1))
            return start_yr + 1  # 2020-21 academic year refers to 2021 outcome

        # Handle 4 digit year
        m4 = re.search(r"\b(19\d{2}|20\d{2})\b", val_str)
        if m4:
            return int(m4.group(1))

        return None

    def standardize_state_name(self, val: Any) -> str:
        """Standardize Indian State / UT name."""
        if pd.isna(val):
            return "Unknown State"
        val_clean = re.sub(r"[^\w\s&]", "", str(val)).strip().lower()
        return STATE_NAME_MAPPING.get(val_clean, str(val).strip().title())

    def standardize_district_name(self, val: Any) -> str:
        """Clean and title-case district name."""
        if pd.isna(val):
            return "Unknown District"
        return str(val).strip().title()

    def detect_outliers(self, df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
        """
        Flag statistical outliers using the Interquartile Range (IQR) method.
        Preserves data while tagging regions with extreme values for researcher review.
        """
        outlier_flags = {}
        for col in numeric_cols:
            if col in df.columns:
                series = df[col].dropna()
                if len(series) >= 10:
                    q1 = series.quantile(0.25)
                    q3 = series.quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr

                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                    if not outliers.empty:
                        outlier_flags[col] = {
                            "count": len(outliers),
                            "lower_bound": round(float(lower_bound), 2),
                            "upper_bound": round(float(upper_bound), 2),
                            "sample_regions": outliers["state"].head(3).tolist() if "state" in outliers.columns else [],
                        }
        return outlier_flags

    def generate_quality_report(
        self,
        df_original: pd.DataFrame,
        df_cleaned: pd.DataFrame,
        source_name: str,
        duplicate_count: int,
        outliers: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate structured data quality dossier."""
        total_records = len(df_cleaned)
        missing_counts = df_cleaned.isna().sum().to_dict()
        total_cells = df_cleaned.shape[0] * df_cleaned.shape[1] if df_cleaned.shape[0] > 0 else 1
        missing_total = sum(missing_counts.values())
        completeness_pct = round((1.0 - (missing_total / total_cells)) * 100, 2)

        available_states = df_cleaned["state"].dropna().unique().tolist() if "state" in df_cleaned.columns else []
        available_districts = df_cleaned["district"].dropna().unique().tolist() if "district" in df_cleaned.columns else []
        available_years = sorted([int(y) for y in df_cleaned["year"].dropna().unique()]) if "year" in df_cleaned.columns else []

        return {
            "source_name": source_name,
            "total_records": total_records,
            "original_records": len(df_original),
            "duplicate_records_removed": duplicate_count,
            "completeness_score_pct": completeness_pct,
            "total_missing_cells": int(missing_total),
            "missing_by_indicator": {k: int(v) for k, v in missing_counts.items() if v > 0},
            "available_years_count": len(available_years),
            "years_range": [min(available_years), max(available_years)] if available_years else [],
            "states_count": len(available_states),
            "districts_count": len(available_districts),
            "outlier_analysis": outliers,
            "last_evaluated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
