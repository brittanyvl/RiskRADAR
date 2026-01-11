"""
extraction/processing/toc_detect.py
-----------------------------------
Table of Contents page detection for NTSB aviation accident reports.

TOC pages should be skipped during consolidation as they dilute
retrieval relevance with redundant section listings.

Detection indicators:
- Explicit "TABLE OF CONTENTS" header
- Multiple dotted leaders (...........) with page numbers
- High density of section names followed by numbers
"""

import re
from dataclasses import dataclass


# Explicit TOC header patterns
TOC_HEADER_PATTERNS = [
    r"TABLE\s+OF\s+CONTENTS",
    r"CONTENTS",
    r"INDEX",
]
TOC_HEADER_REGEX = re.compile(
    rf"^\s*({'|'.join(TOC_HEADER_PATTERNS)})\s*$",
    re.MULTILINE | re.IGNORECASE
)

# Dotted leader pattern: ".........." followed by page number
# Matches: "SYNOPSIS .............. 1" or "1.8 Meteorological .... 15"
DOTTED_LEADER_REGEX = re.compile(r"\.{5,}\s*\d+", re.MULTILINE)

# Page reference pattern: section name followed by page number at line end
# Matches: "History of Flight    12" or "ANALYSIS 45"
PAGE_REFERENCE_REGEX = re.compile(
    r"^[A-Z][A-Za-z\s]+\s+\d{1,3}\s*$",
    re.MULTILINE
)

# Roman numeral page numbers (common in TOC front matter)
ROMAN_NUMERAL_REGEX = re.compile(
    r"\b(i{1,3}|iv|vi{0,3}|ix|xi{0,3})\b",
    re.IGNORECASE
)


@dataclass
class TOCAnalysis:
    """Results of TOC detection analysis."""
    is_toc: bool
    confidence: float  # 0.0 to 1.0
    has_toc_header: bool
    dotted_leader_count: int
    page_reference_count: int
    reasons: list[str]


# Thresholds for TOC detection
MIN_DOTTED_LEADERS = 5  # At least 5 dotted leaders indicates TOC
MIN_PAGE_REFERENCES = 8  # High count of "Section Name   ##" patterns
COMBINED_THRESHOLD = 10  # Sum of dotted + page refs for mixed detection


def is_toc_page(text: str) -> bool:
    """
    Quick check if a page is a Table of Contents page.

    Args:
        text: Page text content

    Returns:
        True if page appears to be TOC, False otherwise
    """
    if not text or len(text.strip()) < 50:
        return False

    # Check for explicit TOC header
    if TOC_HEADER_REGEX.search(text):
        return True

    # Count dotted leaders
    dotted_count = len(DOTTED_LEADER_REGEX.findall(text))
    if dotted_count >= MIN_DOTTED_LEADERS:
        return True

    # Count page references
    page_ref_count = len(PAGE_REFERENCE_REGEX.findall(text))
    if page_ref_count >= MIN_PAGE_REFERENCES:
        return True

    # Combined check
    if dotted_count + page_ref_count >= COMBINED_THRESHOLD:
        return True

    return False


def analyze_toc_page(text: str) -> TOCAnalysis:
    """
    Detailed analysis of whether a page is a TOC page.

    Args:
        text: Page text content

    Returns:
        TOCAnalysis with detailed detection results
    """
    if not text or len(text.strip()) < 50:
        return TOCAnalysis(
            is_toc=False,
            confidence=0.0,
            has_toc_header=False,
            dotted_leader_count=0,
            page_reference_count=0,
            reasons=["Page too short or empty"]
        )

    reasons = []
    confidence = 0.0

    # Check for explicit TOC header
    has_toc_header = bool(TOC_HEADER_REGEX.search(text))
    if has_toc_header:
        confidence += 0.5
        reasons.append("Contains 'TABLE OF CONTENTS' header")

    # Count dotted leaders
    dotted_count = len(DOTTED_LEADER_REGEX.findall(text))
    if dotted_count >= MIN_DOTTED_LEADERS:
        confidence += 0.3
        reasons.append(f"Has {dotted_count} dotted leaders")
    elif dotted_count > 0:
        confidence += 0.1 * (dotted_count / MIN_DOTTED_LEADERS)

    # Count page references
    page_ref_count = len(PAGE_REFERENCE_REGEX.findall(text))
    if page_ref_count >= MIN_PAGE_REFERENCES:
        confidence += 0.2
        reasons.append(f"Has {page_ref_count} page references")
    elif page_ref_count > 0:
        confidence += 0.05 * (page_ref_count / MIN_PAGE_REFERENCES)

    # Cap confidence at 1.0
    confidence = min(1.0, confidence)

    # Determine if TOC based on confidence threshold
    is_toc = confidence >= 0.4 or has_toc_header

    if not reasons:
        reasons.append("No TOC indicators found")

    return TOCAnalysis(
        is_toc=is_toc,
        confidence=confidence,
        has_toc_header=has_toc_header,
        dotted_leader_count=dotted_count,
        page_reference_count=page_ref_count,
        reasons=reasons
    )


def detect_toc_pages(pages: list[tuple[int, str]]) -> list[int]:
    """
    Detect TOC pages from a list of page texts.

    TOC pages are typically at the beginning of the document
    (pages 0-3), so we apply stricter detection for later pages.

    Args:
        pages: List of (page_number, text) tuples

    Returns:
        List of page numbers that are TOC pages
    """
    toc_pages = []

    for page_num, text in pages:
        # TOC is usually in first few pages
        if page_num <= 3:
            # More lenient detection for early pages
            if is_toc_page(text):
                toc_pages.append(page_num)
        else:
            # Stricter detection for later pages
            # (require explicit header for pages after first few)
            if TOC_HEADER_REGEX.search(text):
                toc_pages.append(page_num)

    return toc_pages


def is_cover_page(text: str) -> bool:
    """
    Check if a page appears to be a cover/title page.

    Cover pages often have minimal text and should be skipped.

    Args:
        text: Page text content

    Returns:
        True if page appears to be a cover page
    """
    if not text:
        return True

    text = text.strip()

    # Very short pages are likely cover pages
    if len(text) < 200:
        return True

    # Check for typical cover page patterns
    cover_patterns = [
        r"NATIONAL\s+TRANSPORTATION\s+SAFETY\s+BOARD",
        r"AIRCRAFT\s+ACCIDENT\s+REPORT",
        r"ADOPTED\s+\w+\s+\d+,\s+\d{4}",
        r"REPORT\s+NUMBER",
    ]

    matches = 0
    for pattern in cover_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            matches += 1

    # If multiple cover patterns and short text, likely cover page
    if matches >= 2 and len(text) < 1000:
        return True

    return False


def should_skip_page(page_num: int, text: str) -> tuple[bool, str]:
    """
    Determine if a page should be skipped during consolidation.

    Args:
        page_num: Page number (0-indexed)
        text: Page text content

    Returns:
        Tuple of (should_skip, reason)
    """
    if not text or not text.strip():
        return True, "empty_page"

    if is_cover_page(text):
        return True, "cover_page"

    if is_toc_page(text):
        return True, "toc_page"

    # Check for appendix pages with mostly images/tables (low text density)
    if len(text.strip()) < 100:
        return True, "low_text_density"

    return False, ""


# Version for tracking
TOC_DETECT_VERSION = "1.0.0"
