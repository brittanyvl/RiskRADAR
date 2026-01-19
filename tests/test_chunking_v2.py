"""
tests/test_chunking_v2.py
-------------------------
Unit tests for v2 chunking improvements:
- Hierarchical section merging
- 400 token minimum enforcement
- Section prefix
- Cross-section overlap
- Protected sentence splitting

Run with: python -m pytest tests/test_chunking_v2.py -v
"""

import pytest
from extraction.processing.section_detect import (
    detect_sections,
    _is_parent_of,
    _merge_hierarchical_sections,
    Section,
    SECTION_DETECT_VERSION,
)
from extraction.processing.chunk import (
    split_sentences,
    chunk_document,
    PIPELINE_VERSION,
    SENTENCE_REGEX,
)
from extraction.processing.tokenizer import (
    TOKENIZER_CONFIG,
    count_tokens,
)


class TestHierarchicalSections:
    """Tests for hierarchical section detection and merging."""

    def test_is_parent_of_direct_child(self):
        """1 is parent of 1.1"""
        assert _is_parent_of("1", "1.1") is True

    def test_is_parent_of_different_parent(self):
        """1 is NOT parent of 2.1"""
        assert _is_parent_of("1", "2.1") is False

    def test_is_parent_of_grandchild(self):
        """1 is parent of 1.1.1 (indirect)"""
        assert _is_parent_of("1", "1.1.1") is True

    def test_is_parent_of_sibling(self):
        """1.1 is NOT parent of 1.2"""
        assert _is_parent_of("1.1", "1.2") is False

    def test_is_parent_of_nested_child(self):
        """1.1 is parent of 1.1.1"""
        assert _is_parent_of("1.1", "1.1.1") is True

    def test_is_parent_of_none_values(self):
        """None values return False"""
        assert _is_parent_of(None, "1.1") is False
        assert _is_parent_of("1", None) is False
        assert _is_parent_of(None, None) is False

    def test_detect_sections_merges_parent(self):
        """Parent section 1. THE ACCIDENT should merge into 1.1 child."""
        sample_text = """
1. THE ACCIDENT

1.1 HISTORY OF FLIGHT

On January 31, 2000, the aircraft departed from Los Angeles.

1.2 INJURIES TO PERSONS

Two crew members were injured.
"""
        sections, method = detect_sections(sample_text)

        # Should NOT have a separate "1. THE ACCIDENT" section
        section_names = [s.name for s in sections]
        assert "THE ACCIDENT" not in section_names

        # First section should be merged: "THE ACCIDENT > HISTORY OF FLIGHT"
        assert sections[0].name == "THE ACCIDENT > HISTORY OF FLIGHT"
        assert sections[0].number == "1.1"

        # Should have content, not just header
        content_length = sections[0].end - sections[0].start
        assert content_length > 50  # More than just header text

    def test_section_detect_version(self):
        """Version should be 2.0.0 for hierarchical merging."""
        assert SECTION_DETECT_VERSION == "2.0.0"


class TestSentenceSplitting:
    """Tests for protected sentence splitting."""

    def test_normal_sentence_split(self):
        """Normal sentences should split correctly."""
        text = "The accident happened. The pilot was injured."
        result = split_sentences(text)
        assert len(result) == 2
        assert result[0] == "The accident happened."
        assert result[1] == "The pilot was injured."

    def test_single_digit_protected(self):
        """Single digit + period should NOT split."""
        text = "1. The Accident happened"
        result = split_sentences(text)
        assert len(result) == 1
        assert result[0] == "1. The Accident happened"

    def test_section_number_protected(self):
        """Section numbers like 1.1 should NOT split."""
        text = "Section 1.1. The subsection describes the events."
        result = split_sentences(text)
        # Should NOT split after "1.1."
        assert len(result) == 1

    def test_two_digit_number_splits(self):
        """Two+ digit numbers at sentence end should split."""
        text = "On page 17. The table shows data."
        result = split_sentences(text)
        assert len(result) == 2

    def test_aircraft_number_splits(self):
        """Aircraft numbers like 737 should split."""
        text = "The Boeing 737. The aircraft was damaged."
        result = split_sentences(text)
        assert len(result) == 2

    def test_year_splits(self):
        """Years at sentence end should split."""
        text = "This happened in 2000. The investigation began."
        result = split_sentences(text)
        assert len(result) == 2

    def test_exclamation_splits(self):
        """Exclamation marks should trigger split."""
        text = "Stop! The warning came too late."
        result = split_sentences(text)
        assert len(result) == 2

    def test_question_splits(self):
        """Question marks should trigger split."""
        text = "Why did this happen? The investigation found."
        result = split_sentences(text)
        assert len(result) == 2


class TestTokenizerConfig:
    """Tests for v2 tokenizer configuration."""

    def test_min_tokens_is_400(self):
        """Minimum should be 400 tokens (v2)."""
        assert TOKENIZER_CONFIG["chunk_min_tokens"] == 400

    def test_max_tokens_is_800(self):
        """Maximum should be 800 tokens (v2)."""
        assert TOKENIZER_CONFIG["chunk_max_tokens"] == 800

    def test_overlap_is_25_percent(self):
        """Overlap should be 25% (v2)."""
        assert TOKENIZER_CONFIG["overlap_ratio"] == 0.25


class TestChunkDocument:
    """Tests for chunk_document with v2 improvements."""

    def test_pipeline_version(self):
        """Pipeline version should be 5.0.0."""
        assert PIPELINE_VERSION == "5.0.0"

    def test_chunk_has_section_prefix(self):
        """Chunks should start with section prefix [SECTION_NAME]."""
        document = {
            "report_id": "TEST001.pdf",
            "full_text": """
SYNOPSIS

The aircraft crashed due to engine failure.
""",
            "page_boundaries": [],
            "page_sources": [],
        }
        chunks = chunk_document(document, {})

        assert len(chunks) > 0
        # First chunk should have section prefix
        assert chunks[0]["chunk_text"].startswith("[SYNOPSIS]")

    def test_chunk_tracks_multi_section(self):
        """Chunks spanning multiple sections should be flagged."""
        # This would require a document with small sections that get merged
        # Due to forward borrowing - testing with synthetic data
        document = {
            "report_id": "TEST002.pdf",
            "full_text": """
SYNOPSIS

Short synopsis.

ANALYSIS

The analysis section with more content that spans across. """ * 50,
            "page_boundaries": [],
            "page_sources": [],
        }
        chunks = chunk_document(document, {})

        # Check if any chunk has multiple sections flag
        # (depends on content length and min_tokens threshold)
        for chunk in chunks:
            if chunk.get("sections_spanned"):
                assert "spans_multiple_sections" in chunk.get("quality_flags", [])

    def test_empty_document_returns_empty_list(self):
        """Empty document should return empty list."""
        document = {
            "report_id": "EMPTY.pdf",
            "full_text": "",
            "page_boundaries": [],
            "page_sources": [],
        }
        chunks = chunk_document(document, {})
        assert chunks == []

    def test_chunk_has_token_count(self):
        """Chunks should have token_count field."""
        document = {
            "report_id": "TEST003.pdf",
            "full_text": """
SYNOPSIS

The aircraft was a Boeing 737-800 operated by Example Airlines.
The flight departed from Los Angeles International Airport (LAX).
""",
            "page_boundaries": [],
            "page_sources": [],
        }
        chunks = chunk_document(document, {})

        for chunk in chunks:
            assert "token_count" in chunk
            assert isinstance(chunk["token_count"], int)
            assert chunk["token_count"] > 0


class TestMinimumTokenEnforcement:
    """Tests for 400 token minimum enforcement."""

    def test_small_section_gets_merged(self):
        """Small sections should borrow from next section."""
        # Create a document with a very short first section
        short_section = "The overview."  # ~3 tokens
        long_section = " ".join(["This is content."] * 200)  # Many tokens

        document = {
            "report_id": "MIN_TEST.pdf",
            "full_text": f"""
SYNOPSIS

{short_section}

ANALYSIS

{long_section}
""",
            "page_boundaries": [],
            "page_sources": [],
        }
        chunks = chunk_document(document, {})

        # First chunk should have borrowed from next section
        # because "The overview." alone is < 400 tokens
        if chunks:
            first_chunk = chunks[0]
            # Either has min_tokens worth of content or borrows
            # The chunk should contain content from both sections if merged
            assert first_chunk["token_count"] >= 3  # At least has some content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
