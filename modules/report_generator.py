"""
ReportLab-based PDF and CSV Report Generator for Policy-to-Impact Intelligence System.
"""

import io
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


class PolicyReportGenerator:
    """Generates formal academic policy evaluation reports in PDF and CSV format."""

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Configure academic document typography."""
        self.styles.add(
            ParagraphStyle(
                name="ReportTitle",
                parent=self.styles["Heading1"],
                fontSize=20,
                leading=24,
                textColor=colors.HexColor("#1E3A8A"),
                spaceAfter=6,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="ReportSubTitle",
                parent=self.styles["Normal"],
                fontSize=11,
                leading=14,
                textColor=colors.HexColor("#475569"),
                spaceAfter=15,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="SectionHeading",
                parent=self.styles["Heading2"],
                fontSize=13,
                leading=16,
                textColor=colors.HexColor("#1E3A8A"),
                spaceBefore=12,
                spaceAfter=6,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="DisclaimerText",
                parent=self.styles["Normal"],
                fontSize=8.5,
                leading=11,
                textColor=colors.HexColor("#1E40AF"),
            )
        )

    def generate_pdf_report(
        self,
        policy_meta: Dict[str, Any],
        objectives: List[Dict[str, Any]],
        data_summary: Dict[str, Any],
        risk_summary: List[Dict[str, Any]],
        recommendations: List[Dict[str, Any]],
    ) -> bytes:
        """Create a complete PDF policy evaluation report."""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40,
        )

        elements = []

        # 1. Header
        elements.append(Paragraph("Policy-to-Impact Intelligence System", self.styles["ReportTitle"]))
        elements.append(
            Paragraph(
                f"Education Policy Evaluation Dossier: <b>{policy_meta.get('title', 'NEP 2020')}</b> | "
                f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                self.styles["ReportSubTitle"],
            )
        )
        elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1E3A8A"), spaceAfter=12))

        # 2. Academic Disclaimer Box
        disclaimer_html = (
            "<b>Research Methodology Disclaimer:</b> Observed empirical indicator changes evaluate longitudinal progress "
            "over academic years. In accordance with policy evaluation standards, post-policy improvements should not be "
            "conflated with direct causal attribution of NEP 2020 without formal causal inference."
        )
        disc_table = Table(
            [[Paragraph(disclaimer_html, self.styles["DisclaimerText"])]],
            colWidths=[530],
        )
        disc_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EFF6FF")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#3B82F6")),
                ("PADDING", (0, 0), (-1, -1), 8),
            ])
        )
        elements.append(disc_table)
        elements.append(Spacer(1, 12))

        # 3. Policy Baseline Section
        elements.append(Paragraph("1. Policy Framework & Objectives", self.styles["SectionHeading"]))
        p_desc = (
            f"<b>Policy Name:</b> {policy_meta.get('title', 'NEP 2020')}<br/>"
            f"<b>Sector:</b> {policy_meta.get('sector', 'School Education')}<br/>"
            f"<b>Enactment Year:</b> {policy_meta.get('publication_year', 2020)}<br/>"
            f"<b>Summary:</b> {policy_meta.get('summary', 'Standardized national education policy framework.')}"
        )
        elements.append(Paragraph(p_desc, self.styles["Normal"]))
        elements.append(Spacer(1, 10))

        # Objectives Table
        if objectives:
            obj_data = [["Objective", "Indicator", "Target Value", "Target Year", "Type"]]
            for o in objectives[:8]:
                t_val = f"{o.get('target_value', 'N/A')} {o.get('target_unit', '')}" if o.get("target_value") is not None else "Descriptive"
                t_type = "Statutory Target" if o.get("is_numerical_target") == 1 else "Policy Goal"
                obj_data.append([
                    Paragraph(str(o.get("objective_title", o.get("objective", "")))[:40], self.styles["Normal"]),
                    Paragraph(str(o.get("indicator_name", o.get("indicator", "")))[:30], self.styles["Normal"]),
                    Paragraph(t_val, self.styles["Normal"]),
                    Paragraph(str(o.get("target_year", "N/A")), self.styles["Normal"]),
                    Paragraph(t_type, self.styles["Normal"]),
                ])

            t_obj = Table(obj_data, colWidths=[160, 140, 90, 60, 80])
            t_obj.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ])
            )
            elements.append(t_obj)
            elements.append(Spacer(1, 12))

        # 4. Regional Vulnerability & Risk Section
        elements.append(Paragraph("2. Regional Risk & Target Gap Analysis", self.styles["SectionHeading"]))
        if risk_summary:
            risk_data = [["Region", "Risk Tier", "Score", "Gap Summary", "Key Alert Factors"]]
            for r in risk_summary[:8]:
                risk_data.append([
                    Paragraph(str(r.get("region_name", "")), self.styles["Normal"]),
                    Paragraph(f"<b>{r.get('risk_level', '')}</b>", self.styles["Normal"]),
                    Paragraph(str(r.get("risk_score", "")), self.styles["Normal"]),
                    Paragraph(str(r.get("gap_summary", ""))[:45], self.styles["Normal"]),
                    Paragraph(str(r.get("explanation", ""))[:65], self.styles["Normal"]),
                ])

            t_risk = Table(risk_data, colWidths=[80, 80, 45, 145, 180])
            t_risk.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F8FAFC")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ])
            )
            elements.append(t_risk)
            elements.append(Spacer(1, 12))

        # 5. Prioritized Policy Recommendations
        elements.append(Paragraph("3. Evidence-Based Policy Recommendations", self.styles["SectionHeading"]))
        if recommendations:
            rec_data = [["Priority", "Title & Affected Region", "Empirical Evidence & Action Pathway"]]
            for rec in recommendations[:6]:
                prio_txt = f"<b>{rec.get('priority_level', 'High')}</b>"
                title_region = f"<b>{rec.get('title', '')}</b><br/>Region: {rec.get('affected_region', '')}"
                detail_txt = (
                    f"<i>Evidence:</i> {rec.get('supporting_data', '')}<br/>"
                    f"<i>Action:</i> {rec.get('suggested_investigation', '')}"
                )
                rec_data.append([
                    Paragraph(prio_txt, self.styles["Normal"]),
                    Paragraph(title_region, self.styles["Normal"]),
                    Paragraph(detail_txt, self.styles["Normal"]),
                ])

            t_rec = Table(rec_data, colWidths=[70, 160, 300])
            t_rec.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ])
            )
            elements.append(t_rec)

        # Build PDF
        doc.build(elements)
        buffer.seek(0)
        return buffer.getvalue()
