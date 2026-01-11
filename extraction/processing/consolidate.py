"""
extraction/processing/consolidate.py
------------------------------------
Pass 1: Consolidate pages.jsonl into a single documents.jsonl file.

For each report:
1. Load all pages from pages.jsonl
2. Detect and skip TOC/cover pages
3. Normalize text (fix hyphenation, whitespace)
4. Merge pages in sequential order
5. Extract footnote definitions
6. Track page boundaries (character offsets)
7. Write one line per document to documents.jsonl

Output: extraction/json_data/documents.jsonl (one line per document)
"""

import json
import logging
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .footnote_parse import build_footnote_map, footnotes_to_json
from .toc_detect import should_skip_page
from .tokenizer import count_tokens

logger = logging.getLogger(__name__)


# Pipeline version
CONSOLIDATE_VERSION = "1.0.0"


def normalize_text(text: str) -> str:
    """
    Normalize extracted text for consistent chunking.

    Applies:
    - Fix hyphenation at line breaks
    - Normalize whitespace
    - Normalize line breaks
    - Remove page number artifacts
    - Fix common OCR errors
    - Normalize quotes
    """
    if not text:
        return ""

    # 1. Fix hyphenation at line breaks: "aero-\nplane" -> "aeroplane"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # 2. Normalize multiple spaces to single space
    text = re.sub(r" +", " ", text)

    # 3. Normalize multiple newlines to double newline (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 4. Remove page number artifacts: "- 17 -" at line start/end
    text = re.sub(r"^\s*-\s*\d+\s*-\s*$", "", text, flags=re.MULTILINE)

    # 5. Fix common OCR artifacts
    text = text.replace("|", "I")  # pipe to I (common confusion)

    # 6. Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace("'", "'").replace("'", "'")

    # 7. Strip leading/trailing whitespace
    text = text.strip()

    return text


def load_pages_jsonl(pages_path: Path) -> dict[str, list[dict]]:
    """
    Load pages.jsonl and group by report_id.

    Args:
        pages_path: Path to pages.jsonl

    Returns:
        Dict mapping report_id -> list of page records (sorted by page_number)
    """
    reports = defaultdict(list)

    with open(pages_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                reports[record["report_id"]].append(record)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {line_num}: {e}")

    # Sort pages within each report
    for report_id in reports:
        reports[report_id].sort(key=lambda p: p["page_number"])

    return dict(reports)


def consolidate_report(
    report_id: str,
    pages: list[dict]
) -> dict:
    """
    Consolidate pages into a single document.

    Args:
        report_id: Report identifier
        pages: List of page records (sorted by page_number)

    Returns:
        Document record ready for JSONL output
    """
    # Track statistics
    total_pages = len(pages)
    skipped_pages = []
    included_pages = []
    embedded_count = 0
    ocr_count = 0

    # Build footnote map from all pages first
    page_tuples = [(p["page_number"], p["text"]) for p in pages]
    footnote_map = build_footnote_map(page_tuples)

    # Process pages
    text_parts = []
    page_boundaries = []
    page_sources = []
    current_pos = 0

    for page in pages:
        page_num = page["page_number"]
        page_text = page["text"]

        # Check if page should be skipped
        skip, reason = should_skip_page(page_num, page_text)
        if skip:
            skipped_pages.append({"page": page_num, "reason": reason})
            continue

        # Normalize text
        normalized = normalize_text(page_text)
        if not normalized:
            skipped_pages.append({"page": page_num, "reason": "empty_after_normalize"})
            continue

        # Track page info
        included_pages.append(page_num)

        # Track source
        if page["source"] == "embedded":
            embedded_count += 1
        else:
            ocr_count += 1

        page_sources.append({
            "page": page_num,
            "source": page["source"],
            "char_count": len(normalized),
            "alphabetic_ratio": page.get("alphabetic_ratio"),
            "ocr_confidence": page.get("ocr_confidence"),
        })

        # Track boundaries before adding
        start_pos = current_pos

        # Add text with page separator
        if text_parts:
            text_parts.append("\n\n")
            current_pos += 2

        text_parts.append(normalized)
        current_pos += len(normalized)

        # Record boundary
        page_boundaries.append({
            "page": page_num,
            "start": start_pos,
            "end": current_pos,
        })

    # Combine text
    full_text = "".join(text_parts)

    # Determine primary source
    if embedded_count > 0 and ocr_count > 0:
        primary_source = "mixed"
    elif embedded_count > 0:
        primary_source = "embedded"
    else:
        primary_source = "ocr"

    # Count tokens
    token_count = count_tokens(full_text)

    # Build document record
    document = {
        "report_id": report_id,
        "full_text": full_text,
        "total_pages": total_pages,
        "included_pages": len(included_pages),
        "skipped_pages": skipped_pages,
        "primary_source": primary_source,
        "embedded_page_count": embedded_count,
        "ocr_page_count": ocr_count,
        "footnotes": footnotes_to_json(list(footnote_map.values())),
        "page_boundaries": page_boundaries,
        "page_sources": page_sources,
        "token_count": token_count,
        "pipeline_version": CONSOLIDATE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    return document


def consolidate_all(
    pages_path: Path,
    output_path: Path,
    limit: int | None = None
) -> dict:
    """
    Consolidate all reports from pages.jsonl into a single documents.jsonl.

    Args:
        pages_path: Path to pages.jsonl
        output_path: Path to output documents.jsonl
        limit: Optional limit on number of reports

    Returns:
        Dict with consolidation statistics
    """
    # Load pages
    logger.info(f"Loading pages from {pages_path}")
    reports = load_pages_jsonl(pages_path)
    logger.info(f"Found {len(reports)} reports")

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Statistics
    stats = {
        "total_reports": len(reports),
        "processed": 0,
        "failed": 0,
        "total_pages": sum(len(pages) for pages in reports.values()),
        "total_included_pages": 0,
        "total_skipped_pages": 0,
        "total_tokens": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    # Process reports
    report_ids = list(reports.keys())
    if limit:
        report_ids = report_ids[:limit]
        logger.info(f"Limited to {limit} reports")

    # Write all documents to single file
    with open(output_path, "w", encoding="utf-8") as f:
        for i, report_id in enumerate(report_ids):
            try:
                # Consolidate report
                pages = reports[report_id]
                document = consolidate_report(report_id, pages)

                # Write one line per document
                f.write(json.dumps(document, ensure_ascii=False) + "\n")

                # Update stats
                stats["processed"] += 1
                stats["total_included_pages"] += document["included_pages"]
                stats["total_skipped_pages"] += len(document["skipped_pages"])
                stats["total_tokens"] += document["token_count"]

                if (i + 1) % 50 == 0:
                    logger.info(f"Processed {i + 1}/{len(report_ids)} reports")

            except Exception as e:
                logger.error(f"Failed to consolidate {report_id}: {e}")
                stats["failed"] += 1

    stats["completed_at"] = datetime.now(timezone.utc).isoformat()

    logger.info(
        f"Consolidation complete: {stats['processed']} processed, "
        f"{stats['failed']} failed"
    )

    return stats


def load_documents_jsonl(documents_path: Path) -> dict[str, dict]:
    """
    Load all documents from documents.jsonl into a dict.

    Args:
        documents_path: Path to documents.jsonl

    Returns:
        Dict mapping report_id -> document dict
    """
    documents = {}

    if not documents_path.exists():
        return documents

    with open(documents_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    doc = json.loads(line)
                    documents[doc["report_id"]] = doc
                except json.JSONDecodeError:
                    pass

    return documents


def get_document(documents_path: Path, report_id: str) -> dict | None:
    """
    Load a single document from documents.jsonl.

    Args:
        documents_path: Path to documents.jsonl
        report_id: Report identifier

    Returns:
        Document dict or None if not found
    """
    if not documents_path.exists():
        return None

    with open(documents_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    doc = json.loads(line)
                    if doc.get("report_id") == report_id:
                        return doc
                except json.JSONDecodeError:
                    pass

    return None


def get_consolidation_stats(documents_path: Path) -> dict:
    """
    Get statistics from documents.jsonl.

    Args:
        documents_path: Path to documents.jsonl

    Returns:
        Dict with statistics
    """
    if not documents_path.exists():
        return {"exists": False, "document_count": 0}

    stats = {
        "exists": True,
        "document_count": 0,
        "total_tokens": 0,
        "total_pages": 0,
        "by_source": {"embedded": 0, "ocr": 0, "mixed": 0},
    }

    with open(documents_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    doc = json.loads(line)
                    stats["document_count"] += 1
                    stats["total_tokens"] += doc.get("token_count", 0)
                    stats["total_pages"] += doc.get("included_pages", 0)
                    source = doc.get("primary_source", "unknown")
                    if source in stats["by_source"]:
                        stats["by_source"][source] += 1
                except json.JSONDecodeError:
                    pass

    return stats


if __name__ == "__main__":
    import sys

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Determine paths
    extraction_base = Path(__file__).parent.parent
    pages_path = extraction_base / "json_data" / "pages.jsonl"
    output_path = extraction_base / "json_data" / "documents.jsonl"

    # Check pages.jsonl exists
    if not pages_path.exists():
        logger.error(f"pages.jsonl not found at {pages_path}")
        logger.error("Run consolidate_pages.py first (Pass 0)")
        sys.exit(1)

    # Parse optional limit argument
    limit = None
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
            logger.info(f"Limiting to {limit} reports")
        except ValueError:
            logger.error(f"Invalid limit: {sys.argv[1]}")
            sys.exit(1)

    # Run consolidation
    stats = consolidate_all(
        pages_path=pages_path,
        output_path=output_path,
        limit=limit
    )

    # Print summary
    print("\nConsolidation Summary:")
    print(f"  Total reports: {stats['total_reports']}")
    print(f"  Processed: {stats['processed']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Total included pages: {stats['total_included_pages']}")
    print(f"  Total skipped pages: {stats['total_skipped_pages']}")
    print(f"  Total tokens: {stats['total_tokens']:,}")
    print(f"\nOutput: {output_path}")
