"""
processing/pdf_reader.py
------------------------
Embedded text extraction from PDFs using pymupdf (fitz).

pymupdf is faster and more powerful than pypdf for text extraction.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Tuple

try:
    import fitz  # pymupdf
except ImportError:
    raise ImportError(
        "pymupdf is required for PDF text extraction. "
        "Install with: pip install pymupdf"
    )

from .quality import compute_quality_metrics


def extract_embedded_text(pdf_path: Path, page_number: int) -> str:
    """
    Extract embedded text from a specific page using pymupdf.

    Args:
        pdf_path: Path to PDF file
        page_number: 0-indexed page number

    Returns:
        Extracted text (may be empty if page has no text layer)

    Raises:
        fitz.FileDataError: If PDF is corrupted
        IndexError: If page_number is out of range

    Examples:
        >>> pdf_path = Path("report.pdf")
        >>> text = extract_embedded_text(pdf_path, 0)
        >>> len(text) > 0
        True
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)

    if page_number >= len(doc):
        doc.close()
        raise IndexError(
            f"Page {page_number} out of range (total: {len(doc)} pages)"
        )

    page = doc[page_number]
    text = page.get_text()
    doc.close()

    return text.strip() if text else ""


def get_page_count(pdf_path: Path) -> int:
    """
    Get total number of pages in PDF.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Number of pages

    Raises:
        fitz.FileDataError: If PDF is corrupted
        FileNotFoundError: If PDF doesn't exist
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    count = len(doc)
    doc.close()

    return count


def extract_page_to_json(
    pdf_path: Path,
    report_id: str,
    page_number: int,
    output_dir: Path,
) -> Tuple[dict, Path]:
    """
    Extract embedded text from a page and save to JSON file.

    This function:
    1. Extracts embedded text using pymupdf
    2. Computes quality metrics
    3. Creates JSON with full extraction metadata
    4. Saves to output_dir/{report_id}/page_{number:04d}.json
    5. Returns the data dict and JSON path

    Args:
        pdf_path: Path to PDF file
        report_id: Report filename (e.g., "AIR2507.pdf")
        page_number: 0-indexed page number
        output_dir: Directory to save JSON (e.g., extraction/temp/)

    Returns:
        Tuple of (data_dict, json_path)
        - data_dict: The JSON data as a Python dict
        - json_path: Path to the saved JSON file

    Raises:
        fitz.FileDataError: If PDF is corrupted
        IndexError: If page_number is out of range
        OSError: If cannot write JSON file

    JSON Schema:
        {
            "report_id": str,
            "page_number": int,
            "extraction_method": "embedded",
            "extraction_pass": "initial",
            "text": str,
            "quality_metrics": {
                "char_count": int,
                "alphabetic_ratio": float,
                "garbage_ratio": float,
                "word_count": int,
                "passes_threshold": bool
            },
            "ocr_confidence": null,
            "metadata": {
                "extracted_at": str (ISO timestamp),
                "extraction_time_ms": int,
                "pdf_source": str (absolute path)
            }
        }
    """
    import time

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    start_time = time.time()

    # Extract text
    text = extract_embedded_text(pdf_path, page_number)

    # Compute quality metrics
    quality = compute_quality_metrics(text)

    # Calculate extraction time
    extraction_time_ms = int((time.time() - start_time) * 1000)

    # Build JSON data
    data = {
        "report_id": report_id,
        "page_number": page_number,
        "extraction_method": "embedded",
        "extraction_pass": "initial",
        "text": text,
        "quality_metrics": {
            "char_count": quality["char_count"],
            "alphabetic_ratio": quality["alphabetic_ratio"],
            "garbage_ratio": quality["garbage_ratio"],
            "word_count": quality["word_count"],
            "passes_threshold": quality["passes_quality"],
        },
        "ocr_confidence": None,
        "metadata": {
            "extracted_at": datetime.now().isoformat(),
            "extraction_time_ms": extraction_time_ms,
            "pdf_source": str(pdf_path.absolute()),
        }
    }

    # Create output directory
    report_dir = output_dir / report_id
    report_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON - include report ID in filename for clarity
    report_base = report_id.replace('.pdf', '')
    json_filename = f"{report_base}_page_{page_number:04d}.json"
    json_path = report_dir / json_filename

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return data, json_path


if __name__ == "__main__":
    # Simple test to verify pymupdf is working
    import sys

    print("pymupdf (fitz) PDF Reader Test")
    print("=" * 60)
    print(f"pymupdf version: {fitz.version}")

    if len(sys.argv) > 1:
        # Test with user-provided PDF
        test_pdf = Path(sys.argv[1])
        if test_pdf.exists():
            print(f"\nTesting with: {test_pdf}")
            try:
                page_count = get_page_count(test_pdf)
                print(f"Total pages: {page_count}")

                # Extract first page
                text = extract_embedded_text(test_pdf, 0)
                print(f"\nFirst page text length: {len(text)} characters")
                print(f"Preview (first 200 chars):\n{text[:200]}...")

                # Compute quality
                from quality import compute_quality_metrics
                metrics = compute_quality_metrics(text)
                print(f"\nQuality metrics:")
                print(f"  Passes: {metrics['passes_quality']}")
                print(f"  Reason: {metrics['reason']}")
                print(f"  Alphabetic ratio: {metrics['alphabetic_ratio']:.2f}")
                print(f"  Word count: {metrics['word_count']}")

            except Exception as e:
                print(f"Error: {e}")
        else:
            print(f"Error: File not found: {test_pdf}")
    else:
        print("\nUsage: python pdf_reader.py <path_to_pdf>")
        print("Example: python pdf_reader.py \\\\TRUENAS\\Photos\\RiskRADAR\\AIR2507.pdf")
