"""
extraction/processing/section_detect.py
---------------------------------------
Section header detection for NTSB aviation accident reports.

Detects section headers using regex patterns based on observed
NTSB report structure. Provides fallback strategies when pattern
matching fails.

Detection methods:
- pattern_match: Regex matched known section patterns
- paragraph_fallback: No sections found, using paragraph breaks
- no_structure: No detectable structure, treat as single section
"""

import re
from dataclasses import dataclass


# Known NTSB section names (comprehensive list from document analysis)
KNOWN_SECTIONS = [
    # Standard ICAO/NTSB sections
    "SYNOPSIS",
    "INVESTIGATION",
    "HISTORY OF FLIGHT",
    "HISTORY OF THE FLIGHT",
    "INJURIES TO PERSONS",
    "DAMAGE TO AIRCRAFT",
    "OTHER DAMAGE",
    "CREW INFORMATION",
    "PILOT INFORMATION",
    "AIRCRAFT INFORMATION",
    "METEOROLOGICAL INFORMATION",
    "AIDS TO NAVIGATION",
    "COMMUNICATIONS",
    "AERODROME AND GROUND FACILITIES",
    "AIRPORT INFORMATION",
    "FLIGHT RECORDERS",
    "WRECKAGE",
    "WRECKAGE AND IMPACT INFORMATION",
    "FIRE",
    "SURVIVAL ASPECTS",
    "TESTS AND RESEARCH",
    "MEDICAL AND PATHOLOGICAL INFORMATION",
    "ORGANIZATIONAL AND MANAGEMENT INFORMATION",
    "MAINTENANCE RECORDS",
    "COMPANY INFORMATION",
    # Conclusions sections
    "ANALYSIS AND CONCLUSIONS",
    "ANALYSIS",
    "CONCLUSIONS",
    "FINDINGS",
    "PROBABLE CAUSE",
    "PROBABLE CAUSE AND FINDINGS",
    # Recommendations
    "RECOMMENDATIONS",
    "SAFETY RECOMMENDATIONS",
    # Appendices
    "APPENDIX",
    "APPENDICES",
    # Other common sections
    "THE ACCIDENT",
    "FACTUAL INFORMATION",
    "ADDITIONAL INFORMATION",
    "USEFUL OR EFFECTIVE INVESTIGATION TECHNIQUES",
]

# Build regex pattern for section names (escaped and joined)
_SECTION_NAMES_ESCAPED = [re.escape(name) for name in KNOWN_SECTIONS]
SECTION_NAME_PATTERN = "|".join(_SECTION_NAMES_ESCAPED)

# Section number pattern - handles "1.", "1.1", "1. 8" (with space), "1.8.1"
# Captures the full number including decimals
SECTION_NUMBER_PATTERN = r"(\d+(?:\s*\.\s*\d+)*)"

# Main section header regex
# Matches:
# - "1.8 AIDS TO NAVIGATION" (numbered section)
# - "SYNOPSIS" (standalone header)
# - "1. INVESTIGATION" (number + name)
SECTION_HEADER_REGEX = re.compile(
    rf"^\s*(?:"
    rf"(?:{SECTION_NUMBER_PATTERN})\s*\.?\s*({SECTION_NAME_PATTERN})"  # Numbered: "1.8 AIDS"
    rf"|"
    rf"({SECTION_NAME_PATTERN})"  # Standalone: "SYNOPSIS"
    rf")"
    rf"\s*$",
    re.MULTILINE | re.IGNORECASE
)

# Letter subsections: "(a) Findings", "(b) Probable Cause"
SUBSECTION_REGEX = re.compile(
    r"^\s*\(([a-z])\)\s*(Findings|Probable Cause|Recommendations|Analysis|Conclusions)\s*$",
    re.MULTILINE | re.IGNORECASE
)

# Page number pattern (to IGNORE, not detect as section)
# Matches "- 17 -" style page numbers
PAGE_NUMBER_REGEX = re.compile(r"^\s*-\s*\d+\s*-\s*$", re.MULTILINE)

# Paragraph break pattern for fallback detection
PARAGRAPH_BREAK_REGEX = re.compile(r"\n\s*\n")


@dataclass
class Section:
    """Represents a detected section in the document."""
    name: str
    number: str | None
    start: int  # Character position in text
    end: int | None  # Character position (None until next section found)
    detection_method: str  # 'pattern_match', 'paragraph_fallback', 'no_structure'


def normalize_section_number(number_str: str | None) -> str | None:
    """
    Normalize section numbers by removing internal spaces.

    "1. 8" -> "1.8"
    "1.  8.  1" -> "1.8.1"
    """
    if not number_str:
        return None
    # Remove spaces around periods
    normalized = re.sub(r"\s*\.\s*", ".", number_str.strip())
    return normalized


def detect_sections(text: str) -> tuple[list[Section], str]:
    """
    Detect section headers in document text.

    Returns:
        Tuple of (list of Section objects, detection_method)
        detection_method is one of: 'pattern_match', 'paragraph_fallback', 'no_structure'
    """
    if not text or not text.strip():
        return [], "no_structure"

    # Strategy 1: Pattern-based detection
    sections = _detect_by_pattern(text)
    if sections:
        return sections, "pattern_match"

    # Strategy 2: Paragraph-based fallback
    sections = _detect_by_paragraphs(text)
    if len(sections) > 1:
        return sections, "paragraph_fallback"

    # Strategy 3: No structure detected - entire document as single section
    return [Section(
        name="DOCUMENT",
        number=None,
        start=0,
        end=len(text),
        detection_method="no_structure"
    )], "no_structure"


def _detect_by_pattern(text: str) -> list[Section]:
    """Detect sections using regex patterns."""
    sections = []

    # Find all section header matches
    for match in SECTION_HEADER_REGEX.finditer(text):
        # Extract components based on which group matched
        if match.group(2):  # Numbered section: "1.8 AIDS TO NAVIGATION"
            number = normalize_section_number(match.group(1))
            name = match.group(2).upper()
        else:  # Standalone section: "SYNOPSIS"
            number = None
            name = match.group(3).upper()

        sections.append(Section(
            name=name,
            number=number,
            start=match.start(),
            end=None,
            detection_method="pattern_match"
        ))

    # Also find letter subsections
    for match in SUBSECTION_REGEX.finditer(text):
        letter = match.group(1).lower()
        name = match.group(2).upper()

        sections.append(Section(
            name=f"({letter}) {name}",
            number=None,
            start=match.start(),
            end=None,
            detection_method="pattern_match"
        ))

    # Sort by position
    sections.sort(key=lambda s: s.start)

    # Set end positions (each section ends where the next begins)
    for i, section in enumerate(sections):
        if i + 1 < len(sections):
            section.end = sections[i + 1].start
        else:
            section.end = len(text)

    return sections


def _detect_by_paragraphs(text: str) -> list[Section]:
    """
    Fallback: Create sections from paragraph breaks.

    Used when no section headers are detected. Treats each
    paragraph as a potential section boundary.
    """
    # Find paragraph break positions
    breaks = [0]
    for match in PARAGRAPH_BREAK_REGEX.finditer(text):
        breaks.append(match.end())
    breaks.append(len(text))

    # Create sections from paragraph spans
    # Only create if we have meaningful paragraphs (> 100 chars each on average)
    if len(breaks) <= 2:
        return []

    avg_length = len(text) / (len(breaks) - 1)
    if avg_length < 100:
        # Too fragmented, not useful sections
        return []

    sections = []
    for i in range(len(breaks) - 1):
        start = breaks[i]
        end = breaks[i + 1]

        # Skip tiny sections
        if end - start < 50:
            continue

        sections.append(Section(
            name=f"PARAGRAPH_{i + 1}",
            number=None,
            start=start,
            end=end,
            detection_method="paragraph_fallback"
        ))

    return sections


def get_section_at_position(sections: list[Section], position: int) -> Section | None:
    """
    Find which section contains a given character position.

    Args:
        sections: List of detected sections
        position: Character offset in document

    Returns:
        Section containing the position, or None if not found
    """
    for section in sections:
        if section.start <= position < (section.end or float('inf')):
            return section
    return None


def extract_section_text(text: str, section: Section) -> str:
    """
    Extract the text content of a section.

    Args:
        text: Full document text
        section: Section to extract

    Returns:
        Text content of the section
    """
    end = section.end if section.end is not None else len(text)
    return text[section.start:end]


def get_detection_stats(sections: list[Section]) -> dict:
    """
    Get statistics about detected sections.

    Returns:
        Dict with section detection statistics
    """
    if not sections:
        return {
            "total_sections": 0,
            "numbered_sections": 0,
            "standalone_sections": 0,
            "subsections": 0,
            "detection_method": "no_structure",
        }

    numbered = sum(1 for s in sections if s.number is not None)
    subsections = sum(1 for s in sections if s.name.startswith("("))

    return {
        "total_sections": len(sections),
        "numbered_sections": numbered,
        "standalone_sections": len(sections) - numbered - subsections,
        "subsections": subsections,
        "detection_method": sections[0].detection_method if sections else "no_structure",
        "section_names": [s.name for s in sections],
    }


# Version for tracking
SECTION_DETECT_VERSION = "1.0.0"
