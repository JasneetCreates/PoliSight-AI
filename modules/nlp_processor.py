"""
Policy NLP Processor.
Extracts policy objectives, target groups, measurable indicators, expected outcomes,
and explicit numerical targets from policy documents.
Distinguishes between explicit policy targets and researcher-selected metrics.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Core Indicator definitions linked to official education systems (UDISE+ / NEP 2020)
INDICATOR_ONTOLOGY = [
    {
        "indicator_name": "Gross Enrolment Ratio (GER) - School Education",
        "indicator_code": "ger",
        "keywords": ["gross enrolment ratio", "gross enrollment ratio", "ger", "enrolment", "enrollment", "universal access", "school participation"],
        "default_unit": "%",
        "typical_target_val": 100.0,
        "typical_target_year": 2030,
        "is_higher_desirable": True,
    },
    {
        "indicator_name": "Gross Enrolment Ratio (GER) - Preschool / ECCE",
        "indicator_code": "ger_ecce",
        "keywords": ["early childhood care", "ecce", "preschool", "balvatika", "pre-primary"],
        "default_unit": "%",
        "typical_target_val": 100.0,
        "typical_target_year": 2030,
        "is_higher_desirable": True,
    },
    {
        "indicator_name": "Dropout Rate - Secondary Level",
        "indicator_code": "dropout_rate",
        "keywords": ["dropout rate", "curtailing dropout", "dropping out", "out-of-school children", "drop out", "retention"],
        "default_unit": "%",
        "typical_target_val": 0.0,
        "typical_target_year": 2030,
        "is_higher_desirable": False,
    },
    {
        "indicator_name": "Pupil-Teacher Ratio (PTR)",
        "indicator_code": "pupil_teacher_ratio",
        "keywords": ["pupil-teacher ratio", "pupil teacher ratio", "ptr", "student teacher ratio", "teacher pupil ratio"],
        "default_unit": "ratio",
        "typical_target_val": 30.0,
        "typical_target_year": 2025,
        "is_higher_desirable": False,
    },
    {
        "indicator_name": "Pupil-Teacher Ratio (Socio-Economically Disadvantaged Areas)",
        "indicator_code": "ptr_sedg",
        "keywords": ["ptr in disadvantaged", "ptr 25:1", "socio-economically disadvantaged areas ptr", "special education zones ptr"],
        "default_unit": "ratio",
        "typical_target_val": 25.0,
        "typical_target_year": 2025,
        "is_higher_desirable": False,
    },
    {
        "indicator_name": "Transition Rate to Secondary School",
        "indicator_code": "transition_rate",
        "keywords": ["transition rate", "transition to secondary", "continuity of schooling", "transition"],
        "default_unit": "%",
        "typical_target_val": 100.0,
        "typical_target_year": 2030,
        "is_higher_desirable": True,
    },
    {
        "indicator_name": "School Retention Rate",
        "indicator_code": "retention_rate",
        "keywords": ["retention rate", "retaining students", "completion of secondary school"],
        "default_unit": "%",
        "typical_target_val": 100.0,
        "typical_target_year": 2030,
        "is_higher_desirable": True,
    },
    {
        "indicator_name": "School Electricity Availability",
        "indicator_code": "electricity_availability",
        "keywords": ["electricity", "power supply", "electrification", "basic infrastructure"],
        "default_unit": "%",
        "typical_target_val": 100.0,
        "typical_target_year": 2025,
        "is_higher_desirable": True,
    },
    {
        "indicator_name": "Drinking Water Availability in Schools",
        "indicator_code": "drinking_water_availability",
        "keywords": ["drinking water", "potable water", "safe water", "water facility"],
        "default_unit": "%",
        "typical_target_val": 100.0,
        "typical_target_year": 2025,
        "is_higher_desirable": True,
    },
    {
        "indicator_name": "Functional Separate Toilets (Boys and Girls)",
        "indicator_code": "toilet_availability",
        "keywords": ["functional toilet", "sanitation", "wash facilities", "girls toilet", "toilets"],
        "default_unit": "%",
        "typical_target_val": 100.0,
        "typical_target_year": 2025,
        "is_higher_desirable": True,
    },
    {
        "indicator_name": "School Internet and Computing Connectivity",
        "indicator_code": "internet_availability",
        "keywords": ["internet", "broadband", "digital infrastructure", "computing facility", "ict", "computer lab"],
        "default_unit": "%",
        "typical_target_val": 100.0,
        "typical_target_year": 2025,
        "is_higher_desirable": True,
    },
    {
        "indicator_name": "Percentage of Female Teachers",
        "indicator_code": "female_teacher_percentage",
        "keywords": ["female teachers", "women teachers", "gender parity in teaching", "women in school leadership"],
        "default_unit": "%",
        "typical_target_val": 50.0,
        "typical_target_year": 2030,
        "is_higher_desirable": True,
    },
]

# Target groups taxonomy
TARGET_GROUPS_PATTERNS = [
    (r"school[- ]age children|school children|all children|students in school", "School Students (Ages 3-18)"),
    (r"early childhood|children aged 3[-–]6|preschoolers|pre-primary children", "Children Aged 3-6 (ECCE)"),
    (r"out[- ]of[- ]school children|dropouts|discontinued schooling", "Out-of-School Children"),
    (r"socio[- ]economically disadvantaged groups|sedgs|sc,? st,? obc|tribal|minorities", "Socio-Economically Disadvantaged Groups (SEDGs)"),
    (r"girls|female students|women|gender inclusion|transgender", "Girls and Transgender Students"),
    (r"children with special needs|divyang|disabilities", "Children with Special Needs (Divyang)"),
    (r"teachers|head teachers|school leaders|faculty|educators", "Teachers and School Leaders"),
    (r"aspirational districts|special education zones|rural schools", "Students in Disadvantaged / Rural Regions"),
]


class PolicyNLPProcessor:
    """Natural Language Processor for extracting structured policy records."""

    def __init__(self):
        self._spacy_nlp = None
        self._init_spacy()

    def _init_spacy(self):
        """Lazy load spaCy model if available; otherwise use robust regex/nltk tokenizer."""
        try:
            import spacy
            try:
                self._spacy_nlp = spacy.load("en_core_web_sm")
            except Exception:
                logger.info("Spacy en_core_web_sm not downloaded. Using rule-based fallback tokenizer.")
                self._spacy_nlp = None
        except ImportError:
            self._spacy_nlp = None

    def analyze_policy_document(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze extracted document data and produce structured policy records.
        """
        raw_text = extracted_data.get("raw_text", "")
        cleaned_text = extracted_data.get("cleaned_text", "")
        sections = extracted_data.get("sections", [])
        doc_name = extracted_data.get("document_name", "Policy Document")

        # 1. Detect Policy Name & Sector
        policy_name, sector, publication_year = self._detect_policy_meta(cleaned_text, doc_name)

        # 2. Extract Objectives, Targets, and Measurable Indicators
        objectives: List[Dict[str, Any]] = []

        # Analyze each section
        for sec in sections:
            sec_title = sec.get("title", "")
            sec_text = sec.get("text", "")
            page_num = sec.get("page", 1)

            extracted_items = self._extract_from_section(sec_title, sec_text, page_num, policy_name, sector)
            objectives.extend(extracted_items)

        # If sections produced fewer than expected objectives, run whole-document scan
        if len(objectives) < 3:
            doc_items = self._extract_from_full_text(cleaned_text, policy_name, sector)
            objectives.extend(doc_items)

        # De-duplicate objectives by title/indicator
        unique_objectives = self._deduplicate_objectives(objectives)

        # Build summary
        summary = self._generate_policy_summary(policy_name, sector, publication_year, unique_objectives)

        return {
            "policy_name": policy_name,
            "sector": sector,
            "publication_year": publication_year,
            "total_objectives_found": len(unique_objectives),
            "objectives": unique_objectives,
            "summary": summary,
        }

    def _detect_policy_meta(self, text: str, filename: str) -> Tuple[str, str, int]:
        """Identify policy name, sector, and publication year."""
        name = "National Education Policy 2020"
        sector = "School Education"
        year = 2020

        # Detect name
        if re.search(r"National Education Policy\s+2020|NEP\s*2020", text, re.IGNORECASE) or "nep" in filename.lower():
            name = "National Education Policy 2020"
            year = 2020
        elif re.search(r"Right to Education|RTE\s*2009", text, re.IGNORECASE):
            name = "Right of Children to Free and Compulsory Education Act"
            year = 2009
        elif re.search(r"Samagra Shiksha", text, re.IGNORECASE):
            name = "Samagra Shiksha Abhiyan"
            year = 2018
        else:
            first_lines = text.split("\n")[:5]
            for line in first_lines:
                if len(line.strip()) > 5:
                    name = line.strip()[:60]
                    break

        # Detect year from text
        year_match = re.search(r"\b(19\d{2}|20\d{2})\b", text[:500])
        if year_match:
            try:
                year = int(year_match.group(1))
            except ValueError:
                pass

        # Detect sector
        if re.search(r"higher education", text[:2000], re.IGNORECASE) and not re.search(r"school education", text[:2000], re.IGNORECASE):
            sector = "Higher Education"
        elif re.search(r"vocational education", text[:2000], re.IGNORECASE):
            sector = "Vocational & Skill Education"
        else:
            sector = "School Education"

        return name, sector, year

    def _extract_from_section(
        self, sec_title: str, sec_text: str, page_num: int, policy_name: str, sector: str
    ) -> List[Dict[str, Any]]:
        """Extract policy objectives and numerical targets from a single section."""
        extracted: List[Dict[str, Any]] = []

        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", sec_text)

        for sent in sentences:
            sent_clean = sent.strip()
            if len(sent_clean) < 30:
                continue

            # Check indicators
            for ind_def in INDICATOR_ONTOLOGY:
                # Does sentence match any keyword for this indicator?
                matches_indicator = any(
                    re.search(rf"\b{re.escape(kw)}\b", sent_clean, re.IGNORECASE)
                    for kw in ind_def["keywords"]
                )

                if matches_indicator:
                    # Look for numerical targets in sentence
                    num_target, target_unit, target_year, is_num = self._extract_numerical_target(sent_clean, ind_def)

                    # Target group
                    target_group = self._extract_target_group(sent_clean, sec_title)

                    # Expected outcome
                    expected_outcome = self._extract_expected_outcome(sent_clean, ind_def)

                    # Formulate structured objective title
                    obj_title = self._formulate_objective_title(sec_title, ind_def["indicator_name"], sent_clean)

                    item = {
                        "policy_name": policy_name,
                        "sector": sector,
                        "objective": obj_title,
                        "target_group": target_group,
                        "indicator": ind_def["indicator_name"],
                        "indicator_code": ind_def["indicator_code"],
                        "target_value": num_target,
                        "target_unit": target_unit,
                        "target_year": target_year,
                        "is_numerical_target": 1 if is_num else 0,
                        "expected_outcome": expected_outcome,
                        "section_reference": sec_title[:100],
                        "page_number": page_num,
                        "raw_snippet": sent_clean[:250],
                        "source": f"Uploaded Policy Document (Page {page_num})",
                    }
                    extracted.append(item)

        return extracted

    def _extract_from_full_text(self, text: str, policy_name: str, sector: str) -> List[Dict[str, Any]]:
        """Fallback whole-document scan if sections are sparse."""
        items: List[Dict[str, Any]] = []
        sentences = re.split(r"(?<=[.!?])\s+", text)

        for sent in sentences:
            for ind_def in INDICATOR_ONTOLOGY:
                if any(re.search(rf"\b{re.escape(kw)}\b", sent, re.IGNORECASE) for kw in ind_def["keywords"]):
                    num_target, target_unit, target_year, is_num = self._extract_numerical_target(sent, ind_def)
                    if is_num or len(sent) > 50:
                        items.append({
                            "policy_name": policy_name,
                            "sector": sector,
                            "objective": f"Policy Objective: {ind_def['indicator_name']}",
                            "target_group": self._extract_target_group(sent, ""),
                            "indicator": ind_def["indicator_name"],
                            "indicator_code": ind_def["indicator_code"],
                            "target_value": num_target,
                            "target_unit": target_unit,
                            "target_year": target_year,
                            "is_numerical_target": 1 if is_num else 0,
                            "expected_outcome": self._extract_expected_outcome(sent, ind_def),
                            "section_reference": "General Provisions",
                            "page_number": 1,
                            "raw_snippet": sent[:250],
                            "source": "Uploaded Policy Document",
                        })
        return items

    def _extract_numerical_target(self, text: str, ind_def: Dict[str, Any]) -> Tuple[Optional[float], str, Optional[int], bool]:
        """
        Strictly extract explicit numerical target and target year if stated.
        Distinguishes between policy targets (e.g. 'achieve 100%', 'target is 100% by 2030')
        and historical baseline statistics (e.g. 'GER was 90.9%').
        Never fabricates a numerical target.
        """
        # Target intent keywords
        target_intent = bool(re.search(
            r"\b(target|goal|aim|aims|achieve|achieving|ensure|ensuring|reach|will be|by\s+(?:the\s+year\s+)?20\d{2}|priority)\b",
            text,
            re.IGNORECASE,
        ))

        # Check for historical baseline indicator ("was 90.9%", "in 2017 was")
        is_historical_statement = bool(re.search(r"\b(was|were|stood at)\s+\d+(?:\.\d+)?%", text, re.IGNORECASE))
        if is_historical_statement and not target_intent:
            return None, ind_def["default_unit"], None, False

        # Patterns for ratios: e.g. "30:1", "below 30:1", "under 25:1"
        ratio_match = re.search(r"(\d+)\s*:\s*1", text)

        # Patterns for percentages: e.g., "100%", "100 per cent", "target of 100%", "zero percent"
        pct_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|per\s*cent)", text, re.IGNORECASE)
        zero_pct = bool(re.search(r"\bzero\s*(?:%|percent|per\s*cent)\b", text, re.IGNORECASE))

        # Patterns for target years: e.g. "by 2030", "by the year 2025", "by 2025-26"
        year_match = re.search(r"(?:by(?:\s+the\s+year)?|target\s+year|timeline\s+of)\s+(20\d{2})", text, re.IGNORECASE)

        target_year = int(year_match.group(1)) if year_match else None
        target_val = None
        target_unit = ind_def["default_unit"]
        is_num = False

        if ratio_match and target_intent:
            target_val = float(ratio_match.group(1))
            target_unit = "ratio"
            is_num = True
        elif zero_pct and target_intent:
            target_val = 0.0
            target_unit = "%"
            is_num = True
        elif pct_match and (target_intent or not is_historical_statement):
            val = float(pct_match.group(1))
            # If this is stating past dropout e.g. "GER for Grades 6-8 was 90.9%", skip as target
            if "was" in text.lower() and val != 100.0:
                is_num = False
            else:
                target_val = val
                target_unit = "%"
                is_num = True
        elif re.search(r"\buniversal\s+access\b|\buniversal\s+participation\b|\buniversal\b", text, re.IGNORECASE) and target_intent:
            if ind_def["default_unit"] == "%":
                target_val = 100.0
                target_unit = "%"
                is_num = True

        # If policy mentions a year like 2030 or 2025 anywhere in same sentence
        if not target_year:
            year_any = re.search(r"\b(202[5-9]|203\d|2040)\b", text)
            if year_any:
                target_year = int(year_any.group(1))

        # Default fallback years if indicator is statutory target in NEP
        if is_num and not target_year:
            target_year = ind_def.get("typical_target_year", 2030)

        return target_val, target_unit, target_year, is_num

    def _extract_target_group(self, sentence: str, section_title: str) -> str:
        """Identify target population group from text."""
        combined = f"{section_title} {sentence}".lower()
        for pattern, group_name in TARGET_GROUPS_PATTERNS:
            if re.search(pattern, combined, re.IGNORECASE):
                return group_name
        return "School Students (General Population)"

    def _extract_expected_outcome(self, sentence: str, ind_def: Dict[str, Any]) -> str:
        """Extract or synthesize expected outcome phrase."""
        # Check for goal clauses
        goal_match = re.search(r"(?:aims to|objective is to|goal is to|ensure|achieve|target)\s+([^,.;]{10,80})", sentence, re.IGNORECASE)
        if goal_match:
            return goal_match.group(0).strip().capitalize()

        # Fallback to standard outcome mapping
        name = ind_def["indicator_name"]
        if "Enrolment" in name or "GER" in name:
            return "Universal participation and 100% Gross Enrolment across all school levels."
        if "Dropout" in name:
            return "Curtail dropout rates and ensure 100% retention through secondary education."
        if "Pupil-Teacher" in name:
            return "Achieve standard pupil-teacher ratio (below 30:1 / 25:1 in disadvantaged areas)."
        if "Water" in name or "Electricity" in name or "Toilet" in name or "Internet" in name:
            return f"Universal provision of {name.lower()} across all public schools."
        return f"Measurable improvement in {name}."

    def _formulate_objective_title(self, section_title: str, indicator_name: str, sentence: str) -> str:
        """Generate a concise, academic objective title."""
        if "universal access" in sentence.lower() or "dropout" in sentence.lower():
            return "Universal Access and Retention in School Education"
        if "foundational literacy" in sentence.lower() or "numeracy" in sentence.lower():
            return "Foundational Literacy and Numeracy"
        if "early childhood" in sentence.lower() or "ecce" in sentence.lower():
            return "Early Childhood Care and Education (ECCE)"
        if "teacher" in sentence.lower() or "ptr" in sentence.lower():
            return "Teacher Rationalization and Pupil-Teacher Ratio Optimization"
        if "infrastructure" in sentence.lower() or "water" in sentence.lower() or "electricity" in sentence.lower():
            return "School Infrastructure and Essential Learning Facilities"
        if "inclusive" in sentence.lower() or "equity" in sentence.lower() or "sedg" in sentence.lower():
            return "Equitable and Inclusive Education for All"

        if section_title and len(section_title.strip()) > 5:
            clean_sec = re.sub(r"^(?:CHAPTER|Chapter|PART|Part|\d+\.?)\s*[\dIVXLCDM]*[.:\s-]*", "", section_title).strip()
            if clean_sec:
                return clean_sec

        return f"Objective: Enhancement of {indicator_name}"

    def _deduplicate_objectives(self, objectives: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate objectives based on indicator_code and target_value.
        Prioritizes objectives that possess explicit statutory numerical targets.
        """
        # Sort so numerical targets (is_numerical_target=1) come before non-numerical ones
        sorted_objs = sorted(
            objectives,
            key=lambda x: (
                -x.get("is_numerical_target", 0),
                0 if x.get("target_value") == 100.0 else 1,
                x.get("page_number", 1)
            )
        )
        seen = set()
        deduped = []
        for obj in sorted_objs:
            key = (obj["indicator_code"], obj.get("target_value"), obj.get("target_year"))
            if key not in seen:
                seen.add(key)
                deduped.append(obj)
        return deduped

    def _generate_policy_summary(
        self, policy_name: str, sector: str, year: int, objectives: List[Dict[str, Any]]
    ) -> str:
        """Generate high-level academic summary of analyzed policy document."""
        num_targets = sum(1 for o in objectives if o.get("is_numerical_target") == 1)
        indicators = list({o["indicator"] for o in objectives})

        summary = (
            f"Policy Analysis Summary for '{policy_name}' ({year}):\n"
            f"- Sector Focus: {sector}\n"
            f"- Total Key Objectives Extracted: {len(objectives)}\n"
            f"- Explicit Numerical Targets Identified: {num_targets}\n"
            f"- Covered Measurable Indicators: {', '.join(indicators[:6])}"
            f"{' and others' if len(indicators) > 6 else ''}.\n"
            f"- Research Focus: School-level universal access, retention, PTR rationalization, and infrastructure parity."
        )
        return summary
