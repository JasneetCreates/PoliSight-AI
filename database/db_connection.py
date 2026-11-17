"""
Database connection and operations module for Policy-to-Impact Intelligence System.
Supports SQLite (default) and PostgreSQL compatibility.
"""

import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

# Default DB Path
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = BASE_DIR / "database" / "policy_impact.db"
SCHEMA_PATH = BASE_DIR / "database" / "schema.sql"


def get_db_path() -> Path:
    """Return path to SQLite database."""
    env_path = os.getenv("DB_PATH")
    if env_path:
        p = Path(env_path)
        return p if p.is_absolute() else BASE_DIR / p
    return DEFAULT_DB_PATH


def get_connection() -> sqlite3.Connection:
    """Get SQLite database connection with row factory enabled."""
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(force_recreate: bool = False) -> None:
    """Initialize database tables from schema.sql."""
    db_path = get_db_path()
    if force_recreate and db_path.exists():
        db_path.unlink()

    with get_connection() as conn:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema_script = f.read()
        conn.executescript(schema_script)
        conn.commit()


# -------------------------------------------------------------
# Policy CRUD Operations
# -------------------------------------------------------------

def insert_policy(
    title: str,
    code: str,
    sector: str = "School Education",
    publication_year: int = 2020,
    document_name: Optional[str] = None,
    document_path: Optional[str] = None,
    document_text: Optional[str] = None,
    summary: Optional[str] = None,
) -> int:
    """Insert or update a policy document record. Returns policy_id."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO policies (title, code, sector, publication_year, document_name, document_path, document_text, summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(code) DO UPDATE SET
                title=excluded.title,
                sector=excluded.sector,
                publication_year=excluded.publication_year,
                document_name=coalesce(excluded.document_name, policies.document_name),
                document_path=coalesce(excluded.document_path, policies.document_path),
                document_text=coalesce(excluded.document_text, policies.document_text),
                summary=coalesce(excluded.summary, policies.summary)
            RETURNING id;
            """,
            (title, code, sector, publication_year, document_name, document_path, document_text, summary),
        )
        row = cursor.fetchone()
        policy_id = row[0] if row else cursor.lastrowid
        conn.commit()
        return policy_id


def get_all_policies() -> List[Dict[str, Any]]:
    """Fetch all registered policies."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM policies ORDER BY publication_year DESC, title ASC")
        return [dict(r) for r in cursor.fetchall()]


def get_policy_by_id(policy_id: int) -> Optional[Dict[str, Any]]:
    """Fetch policy by ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM policies WHERE id = ?", (policy_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def insert_policy_objective(
    policy_id: int,
    title: str,
    section_reference: Optional[str] = None,
    page_number: Optional[int] = None,
    target_group: Optional[str] = None,
    expected_outcome: Optional[str] = None,
    timeline_year: Optional[int] = None,
    status: str = "approved",
) -> int:
    """Insert an objective extracted from a policy."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO policy_objectives (policy_id, title, section_reference, page_number, target_group, expected_outcome, timeline_year, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (policy_id, title, section_reference, page_number, target_group, expected_outcome, timeline_year, status),
        )
        conn.commit()
        return cursor.lastrowid


def insert_policy_indicator(
    objective_id: int,
    indicator_name: str,
    indicator_code: Optional[str] = None,
    target_value: Optional[float] = None,
    target_unit: Optional[str] = "%",
    target_year: Optional[int] = None,
    baseline_value: Optional[float] = None,
    baseline_year: Optional[int] = None,
    is_numerical_target: int = 0,
    source_reference: Optional[str] = None,
) -> int:
    """Insert measurable indicator linked to an objective."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO policy_indicators (
                objective_id, indicator_name, indicator_code, target_value,
                target_unit, target_year, baseline_value, baseline_year,
                is_numerical_target, source_reference
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                objective_id,
                indicator_name,
                indicator_code,
                target_value,
                target_unit,
                target_year,
                baseline_value,
                baseline_year,
                is_numerical_target,
                source_reference,
            ),
        )
        conn.commit()
        return cursor.lastrowid


def get_policy_objectives_with_indicators(policy_id: int) -> List[Dict[str, Any]]:
    """Fetch objectives and indicators for a given policy."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT 
                po.id as objective_id,
                po.policy_id,
                po.title as objective_title,
                po.section_reference,
                po.page_number,
                po.target_group,
                po.expected_outcome,
                po.timeline_year,
                po.status as objective_status,
                pi.id as indicator_id,
                pi.indicator_name,
                pi.indicator_code,
                pi.target_value,
                pi.target_unit,
                pi.target_year,
                pi.baseline_value,
                pi.baseline_year,
                pi.is_numerical_target,
                pi.source_reference
            FROM policy_objectives po
            LEFT JOIN policy_indicators pi ON po.id = pi.objective_id
            WHERE po.policy_id = ?
            ORDER BY po.id ASC, pi.id ASC
            """,
            (policy_id,),
        )
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def update_policy_objective(
    objective_id: int,
    title: str,
    target_group: Optional[str],
    expected_outcome: Optional[str],
    timeline_year: Optional[int],
    status: str = "approved",
) -> None:
    """Update an objective record after user review."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE policy_objectives
            SET title = ?, target_group = ?, expected_outcome = ?, timeline_year = ?, status = ?
            WHERE id = ?
            """,
            (title, target_group, expected_outcome, timeline_year, status, objective_id),
        )
        conn.commit()


def update_policy_indicator(
    indicator_id: int,
    indicator_name: str,
    target_value: Optional[float],
    target_unit: Optional[str],
    target_year: Optional[int],
    is_numerical_target: int,
) -> None:
    """Update an indicator record after user review."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE policy_indicators
            SET indicator_name = ?, target_value = ?, target_unit = ?, target_year = ?, is_numerical_target = ?
            WHERE id = ?
            """,
            (indicator_name, target_value, target_unit, target_year, is_numerical_target, indicator_id),
        )
        conn.commit()


def delete_policy_objective(objective_id: int) -> None:
    """Delete an objective and its cascade indicators."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM policy_objectives WHERE id = ?", (objective_id,))
        conn.commit()


# -------------------------------------------------------------
# Data Sources & Education Records CRUD
# -------------------------------------------------------------

def register_data_source(name: str, source_type: str, source_url: str = "", description: str = "") -> int:
    """Register an official data source (e.g. UDISE+)."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO data_sources (name, source_type, source_url, description)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                source_type=excluded.source_type,
                source_url=excluded.source_url,
                description=excluded.description,
                last_accessed=CURRENT_TIMESTAMP
            RETURNING id;
            """,
            (name, source_type, source_url, description),
        )
        row = cursor.fetchone()
        source_id = row[0] if row else cursor.lastrowid
        conn.commit()
        return source_id


def insert_education_dataset(
    source_id: int,
    dataset_name: str,
    file_path: str,
    record_count: int,
    year_start: int,
    year_end: int,
    is_preprocessed: int = 1,
    data_quality_report: str = "{}",
) -> int:
    """Register an uploaded or processed education dataset."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO education_datasets (
                source_id, dataset_name, file_path, record_count,
                year_start, year_end, is_preprocessed, data_quality_report
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_id,
                dataset_name,
                file_path,
                record_count,
                year_start,
                year_end,
                is_preprocessed,
                data_quality_report,
            ),
        )
        conn.commit()
        return cursor.lastrowid


def insert_education_records_batch(dataset_id: int, df: pd.DataFrame) -> int:
    """Batch insert pandas dataframe records into education_records."""
    records_df = df.copy()
    records_df["dataset_id"] = dataset_id

    # Expected column list
    expected_cols = [
        "dataset_id", "year", "state", "district", "total_enrolment",
        "male_enrolment", "female_enrolment", "ger", "dropout_rate",
        "transition_rate", "retention_rate", "pupil_teacher_ratio",
        "female_teacher_percentage", "school_count", "electricity_availability",
        "drinking_water_availability", "toilet_availability", "internet_availability"
    ]

    for col in expected_cols:
        if col not in records_df.columns:
            records_df[col] = None

    insert_df = records_df[expected_cols]

    with get_connection() as conn:
        insert_df.to_sql("education_records", conn, if_exists="append", index=False)
        conn.commit()
        return len(insert_df)


def query_education_records(
    state: Optional[str] = None,
    district: Optional[str] = None,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
) -> pd.DataFrame:
    """Query education records as DataFrame with filters."""
    query = "SELECT * FROM education_records WHERE 1=1"
    params: List[Any] = []

    if state and state != "All States":
        query += " AND state = ?"
        params.append(state)

    if district and district != "All Districts":
        query += " AND district = ?"
        params.append(district)

    if year_start is not None:
        query += " AND year >= ?"
        params.append(year_start)

    if year_end is not None:
        query += " AND year <= ?"
        params.append(year_end)

    query += " ORDER BY state ASC, district ASC, year ASC"

    with get_connection() as conn:
        return pd.read_sql_query(query, conn, params=params)


def get_available_filters() -> Dict[str, List[Any]]:
    """Retrieve available states, districts, and years present in education_records."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT state FROM education_records WHERE state IS NOT NULL ORDER BY state ASC")
        states = [r[0] for r in cursor.fetchall()]

        cursor.execute("SELECT DISTINCT district FROM education_records WHERE district IS NOT NULL ORDER BY district ASC")
        districts = [r[0] for r in cursor.fetchall()]

        cursor.execute("SELECT MIN(year), MAX(year), COUNT(*) FROM education_records")
        min_year, max_year, total_count = cursor.fetchone()

        return {
            "states": states,
            "districts": districts,
            "min_year": min_year,
            "max_year": max_year,
            "total_records": total_count or 0,
        }


# -------------------------------------------------------------
# Risk, Recommendations & Model Results CRUD
# -------------------------------------------------------------

def save_risk_records_batch(risk_records: List[Dict[str, Any]]) -> int:
    """Save risk detection records into risk_analysis."""
    if not risk_records:
        return 0
    df = pd.DataFrame(risk_records)
    with get_connection() as conn:
        df.to_sql("risk_analysis", conn, if_exists="append", index=False)
        conn.commit()
    return len(df)


def get_latest_risk_analysis(region_type: Optional[str] = None) -> pd.DataFrame:
    """Retrieve latest risk analysis findings."""
    query = "SELECT * FROM risk_analysis WHERE 1=1"
    params: List[Any] = []
    if region_type:
        query += " AND region_type = ?"
        params.append(region_type)
    query += " ORDER BY risk_score DESC"

    with get_connection() as conn:
        return pd.read_sql_query(query, conn, params=params)


def save_recommendations_batch(recommendations: List[Dict[str, Any]]) -> int:
    """Save generated recommendations."""
    if not recommendations:
        return 0
    df = pd.DataFrame(recommendations)
    with get_connection() as conn:
        df.to_sql("recommendations", conn, if_exists="append", index=False)
        conn.commit()
    return len(df)


def get_all_recommendations(priority_level: Optional[str] = None) -> pd.DataFrame:
    """Retrieve recommendations."""
    query = "SELECT * FROM recommendations WHERE 1=1"
    params: List[Any] = []
    if priority_level and priority_level != "All":
        query += " AND priority_level = ?"
        params.append(priority_level)
    query += " ORDER BY CASE priority_level WHEN 'Critical' THEN 1 WHEN 'High' THEN 2 WHEN 'Medium' THEN 3 ELSE 4 END ASC, id ASC"

    with get_connection() as conn:
        return pd.read_sql_query(query, conn, params=params)


def save_model_result(
    model_name: str,
    model_type: str,
    target_variable: str,
    features_used: str,
    train_year_range: str,
    metrics_json: str,
    model_params_json: str = "{}",
) -> int:
    """Save ML model evaluation results."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO model_results (
                model_name, model_type, target_variable, features_used,
                train_year_range, metrics_json, model_params_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_name,
                model_type,
                target_variable,
                features_used,
                train_year_range,
                metrics_json,
                model_params_json,
            ),
        )
        conn.commit()
        return cursor.lastrowid


def get_model_results() -> pd.DataFrame:
    """Retrieve all logged ML model runs."""
    with get_connection() as conn:
        return pd.read_sql_query("SELECT * FROM model_results ORDER BY id DESC", conn)
