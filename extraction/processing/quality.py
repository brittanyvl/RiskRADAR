"""
processing/quality.py
---------------------
Quality assessment heuristics for PDF text extraction.

Thresholds are calibrated for NTSB aviation reports (1966-2025).
Based on OCR quality assessment literature and corpus analysis.
"""

import string
import re


# Quality thresholds for triggering OCR fallback
QUALITY_THRESHOLDS = {
    # Minimum characters to consider extraction "successful"
    "min_char_count": 50,  # Empty or near-empty pages

    # Minimum ratio of alphabetic characters (a-zA-Z) to total characters
    # Low ratio indicates garbled extraction or binary/image data
    "min_alphabetic_ratio": 0.50,  # At least 50% should be letters

    # Maximum ratio of "garbage" characters (non-ASCII printable)
    # High ratio indicates encoding issues or binary data extraction
    "max_garbage_ratio": 0.15,  # Max 15% garbage

    # Minimum word-like tokens (3+ consecutive alphabetic chars)
    # Helps detect gibberish that has good alphabetic ratio
    "min_words": 10,  # At least 10 word-like tokens
}


def compute_quality_metrics(text: str) -> dict:
    """
    Compute quality metrics for extracted text.

    Args:
        text: Extracted text from PDF page

    Returns:
        Dict with keys:
        - char_count: int - Total character count
        - alphabetic_ratio: float - Ratio of alphabetic chars (0.0-1.0)
        - garbage_ratio: float - Ratio of non-printable chars (0.0-1.0)
        - word_count: int - Number of word-like tokens
        - passes_quality: bool - True if meets all thresholds
        - reason: str - Why it failed (if applicable), or "ok"

    Examples:
        >>> metrics = compute_quality_metrics("This is a test document.")
        >>> metrics["passes_quality"]
        True
        >>> metrics["alphabetic_ratio"]
        0.87...
    """
    if not text:
        return {
            "char_count": 0,
            "alphabetic_ratio": 0.0,
            "garbage_ratio": 0.0,
            "word_count": 0,
            "passes_quality": False,
            "reason": "empty_text"
        }

    char_count = len(text)

    # Count alphabetic characters
    alpha_count = sum(1 for c in text if c.isalpha())
    alphabetic_ratio = alpha_count / char_count if char_count > 0 else 0.0

    # Count garbage (non-printable, non-ASCII)
    printable_chars = set(string.printable)
    garbage_count = sum(1 for c in text if c not in printable_chars)
    garbage_ratio = garbage_count / char_count if char_count > 0 else 0.0

    # Count word-like tokens (3+ consecutive letters)
    words = re.findall(r'[a-zA-Z]{3,}', text)
    word_count = len(words)

    # Determine if quality passes
    passes = True
    reason = "ok"

    if char_count < QUALITY_THRESHOLDS["min_char_count"]:
        passes = False
        reason = f"too_short (char_count={char_count})"
    elif alphabetic_ratio < QUALITY_THRESHOLDS["min_alphabetic_ratio"]:
        passes = False
        reason = f"low_alpha_ratio ({alphabetic_ratio:.2f})"
    elif garbage_ratio > QUALITY_THRESHOLDS["max_garbage_ratio"]:
        passes = False
        reason = f"high_garbage ({garbage_ratio:.2f})"
    elif word_count < QUALITY_THRESHOLDS["min_words"]:
        passes = False
        reason = f"few_words ({word_count})"

    return {
        "char_count": char_count,
        "alphabetic_ratio": round(alphabetic_ratio, 4),
        "garbage_ratio": round(garbage_ratio, 4),
        "word_count": word_count,
        "passes_quality": passes,
        "reason": reason,
    }


def get_quality_thresholds() -> dict:
    """
    Get current quality thresholds.

    Returns:
        Dict of threshold values
    """
    return QUALITY_THRESHOLDS.copy()


if __name__ == "__main__":
    # Simple test cases
    test_cases = [
        ("This is a good quality text with many words and sentences. " * 3, "Good text"),
        ("", "Empty"),
        ("ABC", "Too short"),
        ("��������������������", "Garbage"),
        ("a" * 100, "Lacks words"),
    ]

    print("Quality Assessment Test Cases")
    print("=" * 60)
    for text, description in test_cases:
        metrics = compute_quality_metrics(text)
        print(f"\n{description}:")
        print(f"  Text length: {len(text)}")
        print(f"  Passes: {metrics['passes_quality']}")
        print(f"  Reason: {metrics['reason']}")
        print(f"  Metrics: char_count={metrics['char_count']}, "
              f"alpha={metrics['alphabetic_ratio']:.2f}, "
              f"garbage={metrics['garbage_ratio']:.2f}, "
              f"words={metrics['word_count']}")
