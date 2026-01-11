"""
extraction/processing/footnote_parse.py
---------------------------------------
Footnote detection and extraction for NTSB aviation accident reports.

NTSB reports use a distinctive footnote style:
- Reference: "1/" or "crash1/" or "!/" (marker + slash)
- Definition: "1/ All times are Pacific standard..." (at page bottom)

This module extracts footnote definitions into a lookup map
and provides utilities for appending footnotes to chunks.
"""

import re
from dataclasses import dataclass


# Footnote reference pattern in body text
# Matches: "1/", "crash1/", "!/", "*/" etc.
# The marker can be digits, words, or symbols
FOOTNOTE_REFERENCE_REGEX = re.compile(
    r"(?<!\w)([a-zA-Z]*\d*[!*]?)/(?!\w)",
    re.MULTILINE
)

# Footnote definition pattern (typically at page bottom)
# Matches: "1/ All times are Pacific standard time"
# The definition continues until end of line or next footnote
FOOTNOTE_DEFINITION_REGEX = re.compile(
    r"^\s*(\d+|[!*])/\s*(.+?)(?=\n\s*\d+/|\n\s*$|\Z)",
    re.MULTILINE | re.DOTALL
)

# Alternative pattern for word-based footnotes
# Matches: "crash1/ Refers to the initial impact..."
WORD_FOOTNOTE_DEFINITION_REGEX = re.compile(
    r"^\s*([a-zA-Z]+\d*)/\s*(.+?)(?=\n\s*[a-zA-Z]+\d*/|\n\s*$|\Z)",
    re.MULTILINE | re.DOTALL
)


@dataclass
class Footnote:
    """Represents a single footnote."""
    marker: str  # e.g., "1", "crash1", "!"
    text: str    # The footnote definition text
    page: int | None = None  # Page where definition was found


@dataclass
class FootnoteResult:
    """Results of footnote extraction."""
    footnotes: dict[str, Footnote]  # marker -> Footnote
    reference_count: int  # Number of references found in text
    definition_count: int  # Number of definitions found


def extract_footnote_definitions(text: str, page_number: int | None = None) -> list[Footnote]:
    """
    Extract footnote definitions from text.

    Footnote definitions typically appear at the bottom of pages
    and follow the pattern: "marker/ definition text"

    Args:
        text: Text to search for footnote definitions
        page_number: Optional page number for tracking

    Returns:
        List of Footnote objects
    """
    footnotes = []

    # Find numeric footnotes (most common)
    for match in FOOTNOTE_DEFINITION_REGEX.finditer(text):
        marker = match.group(1).strip()
        definition = match.group(2).strip()

        # Clean up multi-line definitions
        definition = re.sub(r"\s+", " ", definition)

        if definition:  # Only add if there's actual content
            footnotes.append(Footnote(
                marker=marker,
                text=definition,
                page=page_number
            ))

    # Find word-based footnotes
    for match in WORD_FOOTNOTE_DEFINITION_REGEX.finditer(text):
        marker = match.group(1).strip()
        definition = match.group(2).strip()

        # Clean up multi-line definitions
        definition = re.sub(r"\s+", " ", definition)

        # Avoid duplicates
        if definition and marker not in [f.marker for f in footnotes]:
            footnotes.append(Footnote(
                marker=marker,
                text=definition,
                page=page_number
            ))

    return footnotes


def find_footnote_references(text: str) -> list[str]:
    """
    Find all footnote references in text.

    Args:
        text: Text to search for references

    Returns:
        List of footnote markers referenced (e.g., ["1", "2", "crash1"])
    """
    references = []
    for match in FOOTNOTE_REFERENCE_REGEX.finditer(text):
        marker = match.group(1)
        if marker and marker not in references:
            references.append(marker)
    return references


def build_footnote_map(pages: list[tuple[int, str]]) -> dict[str, Footnote]:
    """
    Build a complete footnote map from all pages.

    Args:
        pages: List of (page_number, text) tuples

    Returns:
        Dict mapping marker -> Footnote
    """
    footnote_map = {}

    for page_num, text in pages:
        footnotes = extract_footnote_definitions(text, page_num)
        for footnote in footnotes:
            # Keep first definition if duplicates (usually same content)
            if footnote.marker not in footnote_map:
                footnote_map[footnote.marker] = footnote

    return footnote_map


def append_footnotes_to_chunk(
    chunk_text: str,
    footnote_map: dict[str, Footnote]
) -> tuple[str, list[Footnote]]:
    """
    Append relevant footnote definitions to a chunk.

    Finds all footnote references in the chunk and appends
    their definitions at the end.

    Args:
        chunk_text: The chunk text
        footnote_map: Map of marker -> Footnote

    Returns:
        Tuple of (text_with_footnotes, list_of_appended_footnotes)
    """
    # Find references in this chunk
    references = find_footnote_references(chunk_text)

    # Get relevant footnotes
    appended_footnotes = []
    for marker in references:
        if marker in footnote_map:
            appended_footnotes.append(footnote_map[marker])

    if not appended_footnotes:
        return chunk_text, []

    # Build footnote section
    footnote_section = "\n\n---\nFootnotes:\n"
    for footnote in appended_footnotes:
        footnote_section += f"[{footnote.marker}/ {footnote.text}]\n"

    return chunk_text + footnote_section, appended_footnotes


def analyze_footnotes(text: str) -> FootnoteResult:
    """
    Analyze footnotes in a text.

    Args:
        text: Text to analyze

    Returns:
        FootnoteResult with counts and extracted footnotes
    """
    references = find_footnote_references(text)
    footnotes = extract_footnote_definitions(text)

    footnote_map = {f.marker: f for f in footnotes}

    return FootnoteResult(
        footnotes=footnote_map,
        reference_count=len(references),
        definition_count=len(footnotes)
    )


def footnotes_to_json(footnotes: list[Footnote]) -> list[dict]:
    """
    Convert footnotes to JSON-serializable format.

    Args:
        footnotes: List of Footnote objects

    Returns:
        List of dicts suitable for JSON serialization
    """
    return [
        {
            "marker": f.marker,
            "text": f.text,
            "page": f.page
        }
        for f in footnotes
    ]


def footnotes_from_json(data: list[dict]) -> list[Footnote]:
    """
    Reconstruct Footnote objects from JSON data.

    Args:
        data: List of dicts from JSON

    Returns:
        List of Footnote objects
    """
    return [
        Footnote(
            marker=d["marker"],
            text=d["text"],
            page=d.get("page")
        )
        for d in data
    ]


# Version for tracking
FOOTNOTE_PARSE_VERSION = "1.0.0"
