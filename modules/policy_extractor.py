"""
Policy Document Extractor.
Extracts and structures text from PDF, DOCX, and TXT files for NLP processing.
"""

import io
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


class PolicyExtractor:
    """Extracts, cleans, and structures text from policy documents."""

    def __init__(self):
        pass

    def extract_from_file(self, file_path_or_bytes: Union[str, Path, io.BytesIO], filename: str) -> Dict[str, Any]:
        """
        Extract text from file or uploaded BytesIO object.
        Returns a structured dictionary with metadata, full text, and detected sections.
        """
        ext = Path(filename).suffix.lower()
        if ext == ".pdf":
            raw_text, page_map = self._extract_pdf(file_path_or_bytes)
        elif ext in [".docx", ".doc"]:
            raw_text, page_map = self._extract_docx(file_path_or_bytes)
        elif ext in [".txt", ".md"]:
            raw_text, page_map = self._extract_txt(file_path_or_bytes)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Please upload a PDF, DOCX, or TXT file.")

        cleaned_text = self.clean_text(raw_text)
        sections = self.segment_into_sections(cleaned_text, page_map)

        return {
            "document_name": filename,
            "file_type": ext,
            "total_pages": len(page_map) if page_map else 1,
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "page_map": page_map,
            "sections": sections,
        }

    def _extract_pdf(self, file_source: Union[str, Path, io.BytesIO]) -> Tuple[str, Dict[int, str]]:
        """Extract text page-by-page from PDF using pypdf."""
        import pypdf

        page_map: Dict[int, str] = {}
        full_text_parts = []

        if isinstance(file_source, (str, Path)):
            reader = pypdf.PdfReader(str(file_source))
        else:
            if hasattr(file_source, "seek"):
                file_source.seek(0)
            reader = pypdf.PdfReader(file_source)

        for idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            page_num = idx + 1
            page_map[page_num] = page_text
            full_text_parts.append(f"\n--- PAGE {page_num} ---\n{page_text}")

        return "".join(full_text_parts), page_map

    def _extract_docx(self, file_source: Union[str, Path, io.BytesIO]) -> Tuple[str, Dict[int, str]]:
        """Extract text from DOCX using python-docx."""
        import docx

        if isinstance(file_source, (str, Path)):
            doc = docx.Document(str(file_source))
        else:
            if hasattr(file_source, "seek"):
                file_source.seek(0)
            doc = docx.Document(file_source)

        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        full_text = "\n\n".join(paragraphs)
        page_map = {1: full_text}
        return full_text, page_map

    def _extract_txt(self, file_source: Union[str, Path, io.BytesIO]) -> Tuple[str, Dict[int, str]]:
        """Extract text from plain text or markdown file."""
        if isinstance(file_source, (str, Path)):
            with open(file_source, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        else:
            if hasattr(file_source, "seek"):
                file_source.seek(0)
            content = file_source.read().decode("utf-8", errors="ignore")

        page_map = {1: content}
        return content, page_map

    def clean_text(self, text: str) -> str:
        """
        Clean formatting, normalize whitespace and unicode, and remove artifacts
        while preserving paragraph structure and punctuation.
        """
        if not text:
            return ""

        # Normalize unicode quotes and dashes
        text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
        text = text.replace("–", "-").replace("—", "-")

        # Replace excessive non-breaking spaces or tabs
        text = re.sub(r"[\xa0\t]+", " ", text)

        # Remove repetitive header/footer page markers e.g. "National Education Policy 2020 Page 12"
        text = re.sub(r"National Education Policy\s+2020\s+Page\s+\d+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"Ministry of Human Resource Development\s+Government of India", "", text, flags=re.IGNORECASE)

        # Standardize multiple consecutive newlines to at most two
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Normalize space before punctuation
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)

        return text.strip()

    def segment_into_sections(self, text: str, page_map: Optional[Dict[int, str]] = None) -> List[Dict[str, Any]]:
        """
        Segment text into logical sections / chapters using regex patterns.
        Detects headings like 'CHAPTER 1', 'PART I', '1.1.', or bold numbered headings.
        """
        sections: List[Dict[str, Any]] = []

        # Heading patterns common in Indian policy documents (NEP 2020)
        heading_pattern = re.compile(
            r"(?:^|\n\n)(?P<heading>(?:(?:CHAPTER|Chapter|Part|PART|Section|SECTION)\s+[0-9IVXLCDM]+[.:\s\w\s,–—-]+)|(?:\d+\.\d+\.?\s+[A-Z][^\n]+))",
            re.MULTILINE,
        )

        matches = list(heading_pattern.finditer(text))

        if not matches:
            # Fallback: Split by double newlines or large paragraphs if no strict chapter headings found
            paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 60]
            for i, p in enumerate(paragraphs[:30]):
                first_line = p.split("\n")[0][:80]
                sections.append({
                    "section_id": f"SEC_{i+1:02d}",
                    "title": first_line,
                    "text": p,
                    "page": self._find_page_for_snippet(p[:100], page_map),
                })
            return sections

        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            heading_text = match.group("heading").strip()
            section_body = text[start:end].strip()

            sections.append({
                "section_id": f"SEC_{i+1:02d}",
                "title": heading_text,
                "text": section_body,
                "page": self._find_page_for_snippet(heading_text, page_map),
            })

        return sections

    def _find_page_for_snippet(self, snippet: str, page_map: Optional[Dict[int, str]]) -> int:
        """Find which page number contains the given snippet."""
        if not page_map:
            return 1

        cleaned_snip = re.sub(r"[^\w\s]", "", snippet[:40]).strip().lower()
        if not cleaned_snip:
            return 1

        for page_num, page_text in page_map.items():
            cleaned_page = re.sub(r"[^\w\s]", "", page_text).strip().lower()
            if cleaned_snip in cleaned_page:
                return page_num
        return 1
