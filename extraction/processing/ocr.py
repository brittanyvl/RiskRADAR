"""
processing/ocr.py
-----------------
OCR processing for scanned PDFs using pytesseract + pdf2image.

Includes word-level confidence scoring for quality assessment.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional
import warnings

try:
    import pytesseract
except ImportError:
    raise ImportError(
        "pytesseract is required for OCR. "
        "Install with: pip install pytesseract\n"
        "Also install Tesseract OCR: https://github.com/tesseract-ocr/tesseract"
    )

try:
    from pdf2image import convert_from_path
except ImportError:
    raise ImportError(
        "pdf2image is required for PDF to image conversion. "
        "Install with: pip install pdf2image\n"
        "Also install poppler: https://github.com/oschwartz10612/poppler-windows/releases"
    )

try:
    import pandas as pd
except ImportError:
    raise ImportError(
        "pandas is required for OCR confidence analysis. "
        "Install with: pip install pandas"
    )

from .quality import compute_quality_metrics


# OCR Configuration
OCR_CONFIG = {
    "dpi": 300,  # 300 DPI balances quality vs processing time
    "lang": "eng",  # English language model
    "psm": 1,  # Page segmentation mode: 1 = Automatic with OSD (handles rotated/skewed)
    "oem": 3,  # OCR Engine Mode: 3 = Default (LSTM only, best accuracy)
}

# Confidence thresholds
CONFIDENCE_THRESHOLDS = {
    "good": 80.0,  # >= 80% confidence = good OCR
    "acceptable": 60.0,  # 60-80% = acceptable
    # < 60% = poor, flag for review
}


def ocr_page_with_confidence(
    pdf_path: Path,
    page_number: int,
    tesseract_cmd: Optional[str] = None,
) -> Tuple[str, dict]:
    """
    OCR a specific page from a PDF with word-level confidence scores.

    This function:
    1. Converts the PDF page to an image (300 DPI)
    2. Runs Tesseract OCR with word-level data extraction
    3. Computes confidence statistics
    4. Returns both the extracted text and confidence metrics

    Args:
        pdf_path: Path to PDF file
        page_number: 0-indexed page number
        tesseract_cmd: Path to tesseract executable (auto-detect if None)

    Returns:
        Tuple of (text, confidence_scores)
        - text: Extracted text as string
        - confidence_scores: Dict with:
            - mean_confidence: float (0-100)
            - min_confidence: float (0-100)
            - max_confidence: float (0-100)
            - low_confidence_words: int (count of words < 60% confidence)
            - total_words: int

    Raises:
        pytesseract.TesseractNotFoundError: If tesseract not installed
        Exception: If page out of range or PDF conversion fails

    Examples:
        >>> text, conf = ocr_page_with_confidence(Path("scan.pdf"), 0)
        >>> conf["mean_confidence"] > 70.0
        True
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Set tesseract command if provided
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    # Convert specific page to image
    # pdf2image uses 1-indexed pages, we use 0-indexed
    try:
        images = convert_from_path(
            pdf_path,
            dpi=OCR_CONFIG["dpi"],
            first_page=page_number + 1,
            last_page=page_number + 1,
        )
    except Exception as e:
        raise Exception(f"Failed to convert PDF page {page_number}: {e}")

    if not images:
        return "", {
            "mean_confidence": 0.0,
            "min_confidence": 0.0,
            "max_confidence": 0.0,
            "low_confidence_words": 0,
            "total_words": 0,
        }

    image = images[0]

    # Extract text with word-level confidence data
    # output_type='data.frame' returns a pandas DataFrame
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress pandas warnings
            data = pytesseract.image_to_data(
                image,
                lang=OCR_CONFIG["lang"],
                config=f"--psm {OCR_CONFIG['psm']} --oem {OCR_CONFIG['oem']}",
                output_type=pytesseract.Output.DATAFRAME,
            )
    except Exception as e:
        raise Exception(f"OCR failed for page {page_number}: {e}")

    # Filter to word-level data (level == 5)
    # Levels: 1=page, 2=block, 3=paragraph, 4=line, 5=word
    words_df = data[data['level'] == 5].copy()

    # Remove empty words and -1 confidence values
    words_df = words_df[words_df['text'].notna()]
    words_df = words_df[words_df['text'].str.strip() != '']
    words_df = words_df[words_df['conf'] != -1]

    # Compute confidence statistics
    if len(words_df) > 0:
        confidence_scores = {
            "mean_confidence": float(words_df['conf'].mean()),
            "min_confidence": float(words_df['conf'].min()),
            "max_confidence": float(words_df['conf'].max()),
            "low_confidence_words": int((words_df['conf'] < CONFIDENCE_THRESHOLDS["acceptable"]).sum()),
            "total_words": int(len(words_df)),
        }
    else:
        # No words detected
        confidence_scores = {
            "mean_confidence": 0.0,
            "min_confidence": 0.0,
            "max_confidence": 0.0,
            "low_confidence_words": 0,
            "total_words": 0,
        }

    # Extract full text
    text = pytesseract.image_to_string(
        image,
        lang=OCR_CONFIG["lang"],
        config=f"--psm {OCR_CONFIG['psm']} --oem {OCR_CONFIG['oem']}",
    )

    return text.strip(), confidence_scores


def ocr_page_to_json(
    pdf_path: Path,
    report_id: str,
    page_number: int,
    output_dir: Path,
    tesseract_cmd: Optional[str] = None,
) -> Tuple[dict, Path]:
    """
    OCR a page and save to JSON file with confidence metrics.

    This function:
    1. Runs OCR with confidence scoring
    2. Computes quality metrics on extracted text
    3. Creates JSON with full OCR metadata
    4. Saves to output_dir/{report_id}/page_{number:04d}.json
    5. Returns the data dict and JSON path

    Args:
        pdf_path: Path to PDF file
        report_id: Report filename (e.g., "AIR2507.pdf")
        page_number: 0-indexed page number
        output_dir: Directory to save JSON (e.g., extraction/ocr_retry/)
        tesseract_cmd: Path to tesseract executable (auto-detect if None)

    Returns:
        Tuple of (data_dict, json_path)

    JSON Schema:
        {
            "report_id": str,
            "page_number": int,
            "extraction_method": "ocr",
            "extraction_pass": "ocr_retry",
            "text": str,
            "quality_metrics": {...},
            "ocr_confidence": {
                "mean_confidence": float,
                "min_confidence": float,
                "max_confidence": float,
                "low_confidence_words": int,
                "total_words": int
            },
            "metadata": {
                "extracted_at": str,
                "extraction_time_ms": int,
                "pdf_source": str,
                "ocr_config": {...}
            }
        }
    """
    import time

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    start_time = time.time()

    # Run OCR with confidence
    text, ocr_confidence = ocr_page_with_confidence(pdf_path, page_number, tesseract_cmd)

    # Compute quality metrics
    quality = compute_quality_metrics(text)

    # Calculate extraction time
    extraction_time_ms = int((time.time() - start_time) * 1000)

    # Build JSON data
    data = {
        "report_id": report_id,
        "page_number": page_number,
        "extraction_method": "ocr",
        "extraction_pass": "ocr_retry",
        "text": text,
        "quality_metrics": {
            "char_count": quality["char_count"],
            "alphabetic_ratio": quality["alphabetic_ratio"],
            "garbage_ratio": quality["garbage_ratio"],
            "word_count": quality["word_count"],
            "passes_threshold": quality["passes_quality"],
        },
        "ocr_confidence": ocr_confidence,
        "metadata": {
            "extracted_at": datetime.now().isoformat(),
            "extraction_time_ms": extraction_time_ms,
            "pdf_source": str(pdf_path.absolute()),
            "ocr_config": OCR_CONFIG.copy(),
        }
    }

    # Create output directory
    report_dir = output_dir / report_id
    report_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_filename = f"page_{page_number:04d}.json"
    json_path = report_dir / json_filename

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return data, json_path


def get_ocr_config() -> dict:
    """Get current OCR configuration."""
    return OCR_CONFIG.copy()


def get_confidence_thresholds() -> dict:
    """Get confidence quality thresholds."""
    return CONFIDENCE_THRESHOLDS.copy()


if __name__ == "__main__":
    # Test OCR functionality
    import sys

    print("Tesseract OCR Test")
    print("=" * 60)

    try:
        version = pytesseract.get_tesseract_version()
        print(f"Tesseract version: {version}")
    except pytesseract.TesseractNotFoundError:
        print("ERROR: Tesseract not found!")
        print("Install Tesseract OCR:")
        print("  Windows: choco install tesseract")
        print("  Or download: https://github.com/tesseract-ocr/tesseract")
        sys.exit(1)

    print(f"OCR Config: DPI={OCR_CONFIG['dpi']}, PSM={OCR_CONFIG['psm']}, OEM={OCR_CONFIG['oem']}")
    print(f"Confidence thresholds: Good>={CONFIDENCE_THRESHOLDS['good']}, "
          f"Acceptable>={CONFIDENCE_THRESHOLDS['acceptable']}")

    if len(sys.argv) > 1:
        # Test with user-provided PDF
        test_pdf = Path(sys.argv[1])
        page_num = int(sys.argv[2]) if len(sys.argv) > 2 else 0

        if test_pdf.exists():
            print(f"\nTesting OCR on: {test_pdf}, page {page_num}")
            try:
                text, conf = ocr_page_with_confidence(test_pdf, page_num)

                print(f"\nExtracted text length: {len(text)} characters")
                print(f"Preview (first 200 chars):\n{text[:200]}...")

                print(f"\nOCR Confidence:")
                print(f"  Mean: {conf['mean_confidence']:.1f}%")
                print(f"  Range: {conf['min_confidence']:.1f}% - {conf['max_confidence']:.1f}%")
                print(f"  Low confidence words: {conf['low_confidence_words']}/{conf['total_words']}")

                quality_label = "Good" if conf["mean_confidence"] >= CONFIDENCE_THRESHOLDS["good"] else \
                               "Acceptable" if conf["mean_confidence"] >= CONFIDENCE_THRESHOLDS["acceptable"] else \
                               "Poor"
                print(f"  Quality: {quality_label}")

            except Exception as e:
                print(f"Error: {e}")
        else:
            print(f"Error: File not found: {test_pdf}")
    else:
        print("\nUsage: python ocr.py <path_to_pdf> [page_number]")
        print("Example: python ocr.py \\\\TRUENAS\\Photos\\RiskRADAR\\AIR1960.pdf 0")
