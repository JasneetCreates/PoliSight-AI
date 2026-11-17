"""
Unit and Integration Tests for Module 2: Data Preprocessing & Collection (Phase 3).
"""

import pytest
import pandas as pd
from pathlib import Path

from modules.data_preprocessor import EducationDataPreprocessor
from modules.data_collector import EducationDataCollector
from database.db_connection import (
    init_db,
    query_education_records,
    get_available_filters,
)

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_CSV = BASE_DIR / "data" / "sample" / "udise_education_dataset.csv"


@pytest.fixture(scope="module")
def setup_db():
    """Ensure database schema is ready."""
    init_db()
    yield


def test_year_normalization():
    """Test standardizing varied academic year formats."""
    pre = EducationDataPreprocessor()
    assert pre.normalize_year("2020-21") == 2021
    assert pre.normalize_year("2019/20") == 2020
    assert pre.normalize_year("2022") == 2022
    assert pre.normalize_year(2023) == 2023
    assert pre.normalize_year("invalid") is None


def test_state_name_standardization():
    """Test standardizing historical and abbreviated Indian state names."""
    pre = EducationDataPreprocessor()
    assert pre.standardize_state_name("Orissa") == "Odisha"
    assert pre.standardize_state_name("orissa") == "Odisha"
    assert pre.standardize_state_name("Pondicherry") == "Puducherry"
    assert pre.standardize_state_name("J&K") == "Jammu and Kashmir"
    assert pre.standardize_state_name("Uttaranchal") == "Uttarakhand"
    assert pre.standardize_state_name("UP") == "Uttar Pradesh"
    assert pre.standardize_state_name("Kerala") == "Kerala"


def test_column_name_standardization():
    """Test mapping diverse raw header aliases to standard ontology."""
    pre = EducationDataPreprocessor()
    df_raw = pd.DataFrame({
        "academic_year": ["2020-21"],
        "state_name": ["Kerala"],
        "dist_name": ["Ernakulam"],
        "gross_enrolment_ratio": ["98.5%"],
        "secondary_dropout": ["2.1%"],
        "ptr": [18.2],
        "safe_water": ["99.0%"],
    })
    df_clean = pre.standardize_column_names(df_raw)
    assert "year" in df_clean.columns
    assert "state" in df_clean.columns
    assert "district" in df_clean.columns
    assert "ger" in df_clean.columns
    assert "dropout_rate" in df_clean.columns
    assert "pupil_teacher_ratio" in df_clean.columns
    assert "drinking_water_availability" in df_clean.columns


def test_preprocessing_pipeline_and_quality_report():
    """Test end-to-end preprocessing pipeline and data quality auditing."""
    pre = EducationDataPreprocessor()
    df_raw = pd.DataFrame([
        {"academic_year": "2021-22", "state_name": "Orissa", "dist_name": "Cuttack", "ger": "88.5%", "ptr": 25.0},
        {"academic_year": "2021-22", "state_name": "Orissa", "dist_name": "Cuttack", "ger": "88.5%", "ptr": 25.0},  # Duplicate
        {"academic_year": "2020-21", "state_name": "UP", "dist_name": "Lucknow", "ger": "78.2%", "ptr": 35.0},
    ])

    df_cleaned, report = pre.preprocess_dataset(df_raw, source_name="Test Raw")
    
    # Check deduplication
    assert len(df_cleaned) == 2
    assert report["duplicate_records_removed"] == 1
    assert report["completeness_score_pct"] > 90.0

    # Check state harmonization
    assert "Odisha" in df_cleaned["state"].values
    assert "Uttar Pradesh" in df_cleaned["state"].values

    # Check numeric conversion
    assert df_cleaned["ger"].dtype in [float, int]
    assert df_cleaned["year"].dtype in [int, "int64", "int32"]


def test_collector_generate_and_ingest(setup_db):
    """Test generating sample benchmark dataset and ingesting into database."""
    collector = EducationDataCollector()
    df_sample = collector.generate_and_save_udise_sample()
    assert len(df_sample) >= 300
    assert SAMPLE_CSV.exists()

    dataset_id, report = collector.ingest_and_store(
        df=df_sample,
        source_name="UDISE+ Benchmark Sample",
        source_type="UDISE+",
    )
    assert dataset_id > 0
    assert report["total_records"] == len(df_sample)

    records = query_education_records()
    assert len(records) >= len(df_sample)

    meta = get_available_filters()
    assert len(meta["states"]) >= 15
    assert meta["min_year"] == 2018
    assert meta["max_year"] == 2024
