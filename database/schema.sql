-- Policy-to-Impact Intelligence System Database Schema
-- Compatible with SQLite and PostgreSQL

CREATE TABLE IF NOT EXISTS policies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    code TEXT UNIQUE,
    sector TEXT,
    publication_year INTEGER,
    document_name TEXT,
    document_path TEXT,
    document_text TEXT,
    summary TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS policy_objectives (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    policy_id INTEGER REFERENCES policies(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    section_reference TEXT,
    page_number INTEGER,
    target_group TEXT,
    expected_outcome TEXT,
    timeline_year INTEGER,
    status TEXT DEFAULT 'approved',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS policy_indicators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    objective_id INTEGER REFERENCES policy_objectives(id) ON DELETE CASCADE,
    indicator_name TEXT NOT NULL,
    indicator_code TEXT,
    target_value REAL,
    target_unit TEXT,
    target_year INTEGER,
    baseline_value REAL,
    baseline_year INTEGER,
    is_numerical_target INTEGER DEFAULT 0,
    source_reference TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS data_sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    source_type TEXT,
    source_url TEXT,
    description TEXT,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS education_datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER REFERENCES data_sources(id),
    dataset_name TEXT NOT NULL,
    file_path TEXT,
    record_count INTEGER DEFAULT 0,
    year_start INTEGER,
    year_end INTEGER,
    is_preprocessed INTEGER DEFAULT 0,
    data_quality_report TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS education_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER REFERENCES education_datasets(id) ON DELETE CASCADE,
    year INTEGER NOT NULL,
    state TEXT NOT NULL,
    district TEXT,
    total_enrolment REAL,
    male_enrolment REAL,
    female_enrolment REAL,
    ger REAL,
    dropout_rate REAL,
    transition_rate REAL,
    retention_rate REAL,
    pupil_teacher_ratio REAL,
    female_teacher_percentage REAL,
    school_count INTEGER,
    electricity_availability REAL,
    drinking_water_availability REAL,
    toilet_availability REAL,
    internet_availability REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_records_state_year ON education_records(state, year);
CREATE INDEX IF NOT EXISTS idx_records_district ON education_records(district);

CREATE TABLE IF NOT EXISTS impact_analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    policy_indicator_id INTEGER REFERENCES policy_indicators(id),
    analysis_type TEXT,
    region_type TEXT,
    region_name TEXT,
    baseline_observation REAL,
    current_observation REAL,
    percentage_change REAL,
    target_gap REAL,
    projected_value REAL,
    projected_year INTEGER,
    target_achievable_status TEXT,
    details_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS risk_analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    region_type TEXT,
    region_name TEXT NOT NULL,
    year INTEGER,
    risk_level TEXT NOT NULL,
    risk_score REAL,
    anomaly_score REAL,
    primary_flagged_indicators TEXT,
    gap_summary TEXT,
    explanation TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    risk_id INTEGER REFERENCES risk_analysis(id),
    policy_objective_id INTEGER REFERENCES policy_objectives(id),
    title TEXT NOT NULL,
    identified_problem TEXT NOT NULL,
    affected_region TEXT NOT NULL,
    related_indicator TEXT NOT NULL,
    supporting_data TEXT,
    reason TEXT NOT NULL,
    priority_level TEXT NOT NULL,
    suggested_investigation TEXT NOT NULL,
    is_approved INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    model_type TEXT NOT NULL,
    target_variable TEXT,
    features_used TEXT,
    train_year_range TEXT,
    metrics_json TEXT,
    model_params_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
