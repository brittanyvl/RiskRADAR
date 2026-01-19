"""
extraction/scripts/validate_chunking.py
---------------------------------------
Validation script for the chunking pipeline output.

Performs comprehensive checks on:
- pages.jsonl integrity
- document JSONL files
- chunk JSONL files
- Database consistency
- Token distribution
- Section detection coverage

Usage:
  python -m extraction.scripts.validate_chunking
  python -m extraction.scripts.validate_chunking --verbose
  python -m extraction.scripts.validate_chunking --sample 10
"""

import argparse
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

from riskradar.config import DB_PATH
from sqlite.connection import init_db

logger = logging.getLogger(__name__)


class ValidationResult:
    """Tracks validation results."""

    def __init__(self, name: str):
        self.name = name
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.errors = []
        self.warns = []

    def ok(self, message: str = ""):
        self.passed += 1
        if message:
            logger.debug(f"  OK: {message}")

    def fail(self, message: str):
        self.failed += 1
        self.errors.append(message)
        logger.error(f"  FAIL: {message}")

    def warn(self, message: str):
        self.warnings += 1
        self.warns.append(message)
        logger.warning(f"  WARN: {message}")

    def summary(self) -> str:
        status = "PASS" if self.failed == 0 else "FAIL"
        return (
            f"{self.name}: {status} "
            f"(passed={self.passed}, failed={self.failed}, warnings={self.warnings})"
        )


def validate_pages_jsonl(pages_path: Path, sample_size: int = 100) -> ValidationResult:
    """Validate pages.jsonl file."""
    result = ValidationResult("pages.jsonl")

    if not pages_path.exists():
        result.fail(f"File not found: {pages_path}")
        return result

    result.ok("File exists")

    # Check file is valid JSONL
    report_pages = defaultdict(list)
    line_count = 0
    parse_errors = 0

    with open(pages_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            line_count += 1
            try:
                record = json.loads(line)

                # Required fields
                required = ["report_id", "page_number", "text", "source"]
                for field in required:
                    if field not in record:
                        result.fail(f"Line {line_num}: Missing field '{field}'")

                # Track pages
                report_id = record.get("report_id")
                page_num = record.get("page_number")
                if report_id and page_num is not None:
                    report_pages[report_id].append(page_num)

            except json.JSONDecodeError as e:
                parse_errors += 1
                if parse_errors <= 5:
                    result.fail(f"Line {line_num}: JSON parse error: {e}")

    if parse_errors == 0:
        result.ok(f"All {line_count} lines are valid JSON")
    else:
        result.fail(f"{parse_errors} JSON parse errors")

    # Check for duplicates
    duplicates = 0
    for report_id, pages in report_pages.items():
        page_counts = Counter(pages)
        for page, count in page_counts.items():
            if count > 1:
                duplicates += 1
                if duplicates <= 5:
                    result.fail(f"Duplicate page: {report_id} page {page} appears {count} times")

    if duplicates == 0:
        result.ok("No duplicate pages found")
    else:
        result.fail(f"{duplicates} total duplicate page entries")

    # Check page ordering
    ordering_issues = 0
    for report_id, pages in report_pages.items():
        if pages != sorted(pages):
            ordering_issues += 1
            if ordering_issues <= 3:
                result.warn(f"Pages not in order for {report_id}")

    if ordering_issues == 0:
        result.ok("All pages in order within reports")
    else:
        result.warn(f"{ordering_issues} reports have out-of-order pages")

    logger.info(f"  Total pages: {line_count}")
    logger.info(f"  Unique reports: {len(report_pages)}")

    return result


def validate_documents(documents_dir: Path, sample_size: int = 10) -> ValidationResult:
    """Validate document JSONL files."""
    result = ValidationResult("documents/")

    if not documents_dir.exists():
        result.fail(f"Directory not found: {documents_dir}")
        return result

    result.ok("Directory exists")

    doc_files = list(documents_dir.glob("*.jsonl"))
    if not doc_files:
        result.fail("No document files found")
        return result

    result.ok(f"Found {len(doc_files)} document files")

    # Sample validation
    sample = random.sample(doc_files, min(sample_size, len(doc_files)))
    empty_docs = 0
    token_issues = 0

    for doc_path in sample:
        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                doc = json.loads(f.readline())

            # Required fields
            required = ["report_id", "full_text", "total_pages", "included_pages",
                        "primary_source", "token_count", "created_at"]
            for field in required:
                if field not in doc:
                    result.fail(f"{doc_path.name}: Missing field '{field}'")

            # Check text not empty
            if not doc.get("full_text", "").strip():
                empty_docs += 1
                result.warn(f"{doc_path.name}: Empty full_text")

            # Check token count reasonable
            token_count = doc.get("token_count", 0)
            text_len = len(doc.get("full_text", ""))
            expected_tokens = text_len // 4  # rough estimate

            if token_count > 0 and abs(token_count - expected_tokens) / max(1, expected_tokens) > 0.5:
                token_issues += 1
                if token_issues <= 3:
                    result.warn(
                        f"{doc_path.name}: Token count {token_count} seems off "
                        f"for {text_len} chars"
                    )

            # Check source validity
            source = doc.get("primary_source")
            if source not in ["embedded", "ocr", "mixed"]:
                result.fail(f"{doc_path.name}: Invalid primary_source '{source}'")

        except Exception as e:
            result.fail(f"{doc_path.name}: Error reading: {e}")

    if empty_docs == 0:
        result.ok("No empty documents in sample")

    return result


def validate_chunks(chunks_dir: Path, sample_size: int = 10) -> ValidationResult:
    """Validate chunk JSONL files."""
    result = ValidationResult("chunks/")

    if not chunks_dir.exists():
        result.fail(f"Directory not found: {chunks_dir}")
        return result

    result.ok("Directory exists")

    chunk_files = list(chunks_dir.glob("*.jsonl"))
    if not chunk_files:
        result.fail("No chunk files found")
        return result

    result.ok(f"Found {len(chunk_files)} chunk files")

    # Aggregate stats
    total_chunks = 0
    token_distribution = Counter()
    section_methods = Counter()
    chunks_with_footnotes = 0
    sequence_issues = 0

    # Sample validation
    sample = random.sample(chunk_files, min(sample_size, len(chunk_files)))

    for chunk_path in sample:
        try:
            chunks = []
            with open(chunk_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        chunks.append(json.loads(line))

            total_chunks += len(chunks)

            # Check sequence numbers
            sequences = [c.get("chunk_sequence") for c in chunks]
            expected = list(range(len(chunks)))
            if sequences != expected:
                sequence_issues += 1
                if sequence_issues <= 3:
                    result.warn(f"{chunk_path.name}: Sequence mismatch")

            for chunk in chunks:
                # Required fields
                required = ["chunk_id", "report_id", "chunk_sequence",
                            "page_start", "page_end", "chunk_text",
                            "token_count", "text_source"]
                for field in required:
                    if field not in chunk:
                        result.fail(f"{chunk['chunk_id']}: Missing field '{field}'")
                        break

                # Track token distribution
                tokens = chunk.get("token_count", 0)
                if tokens < 500:
                    token_distribution["< 500"] += 1
                elif tokens <= 700:
                    token_distribution["500-700"] += 1
                else:
                    token_distribution["> 700"] += 1

                # Track section detection
                method = chunk.get("section_detection_method", "unknown")
                section_methods[method] += 1

                # Track footnotes
                if chunk.get("has_footnotes"):
                    chunks_with_footnotes += 1

        except Exception as e:
            result.fail(f"{chunk_path.name}: Error reading: {e}")

    if sequence_issues == 0:
        result.ok("Chunk sequences are correct")

    # Report statistics
    logger.info(f"  Chunks in sample: {total_chunks}")
    logger.info(f"  Token distribution: {dict(token_distribution)}")
    logger.info(f"  Section methods: {dict(section_methods)}")
    logger.info(f"  Chunks with footnotes: {chunks_with_footnotes}")

    # Check token distribution
    in_range = token_distribution.get("500-700", 0)
    total = sum(token_distribution.values())
    if total > 0:
        in_range_pct = 100 * in_range / total
        if in_range_pct < 50:
            result.warn(f"Only {in_range_pct:.1f}% of chunks in target range (500-700)")
        else:
            result.ok(f"{in_range_pct:.1f}% of chunks in target range")

    return result


def validate_database(conn) -> ValidationResult:
    """Validate database consistency."""
    result = ValidationResult("database")

    try:
        # Check documents table
        doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        result.ok(f"Documents table has {doc_count} rows")

        # Check chunks table
        chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        result.ok(f"Chunks table has {chunk_count} rows")

        # Check for orphan chunks (chunks without documents)
        orphans = conn.execute("""
            SELECT COUNT(*) FROM chunks c
            LEFT JOIN documents d ON c.report_id = d.report_id
            WHERE d.id IS NULL
        """).fetchone()[0]
        if orphans > 0:
            result.warn(f"{orphans} chunks without matching documents")
        else:
            result.ok("No orphan chunks")

        # Check chunk sequence uniqueness
        duplicates = conn.execute("""
            SELECT report_id, chunk_sequence, COUNT(*)
            FROM chunks
            GROUP BY report_id, chunk_sequence
            HAVING COUNT(*) > 1
        """).fetchall()
        if duplicates:
            result.fail(f"{len(duplicates)} duplicate chunk sequences")
        else:
            result.ok("Chunk sequences are unique per report")

        # Check chunking runs
        runs = conn.execute("""
            SELECT status, COUNT(*) FROM chunking_runs
            GROUP BY status
        """).fetchall()
        for status, count in runs:
            logger.info(f"  Chunking runs ({status}): {count}")

        # Check errors
        errors = conn.execute("SELECT COUNT(*) FROM chunking_errors").fetchone()[0]
        if errors > 0:
            result.warn(f"{errors} chunking errors logged")
        else:
            result.ok("No chunking errors logged")

    except Exception as e:
        result.fail(f"Database error: {e}")

    return result


def validate_cross_consistency(
    pages_path: Path,
    documents_dir: Path,
    chunks_dir: Path
) -> ValidationResult:
    """Validate cross-file consistency."""
    result = ValidationResult("cross-consistency")

    # Get report IDs from each stage
    pages_reports = set()
    if pages_path.exists():
        with open(pages_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        pages_reports.add(record.get("report_id"))
                    except json.JSONDecodeError:
                        pass

    doc_reports = set()
    if documents_dir.exists():
        doc_reports = {p.stem for p in documents_dir.glob("*.jsonl")}

    chunk_reports = set()
    if chunks_dir.exists():
        chunk_reports = {p.stem for p in chunks_dir.glob("*.jsonl")}

    # Check consistency
    if not pages_reports:
        result.fail("No pages found")
    else:
        result.ok(f"{len(pages_reports)} reports in pages.jsonl")

    if not doc_reports:
        result.fail("No documents found")
    elif doc_reports == pages_reports:
        result.ok("Documents match pages (all reports processed)")
    else:
        missing = pages_reports - doc_reports
        extra = doc_reports - pages_reports
        if missing:
            result.warn(f"{len(missing)} reports in pages but not in documents")
        if extra:
            result.warn(f"{len(extra)} reports in documents but not in pages")

    if not chunk_reports:
        result.fail("No chunks found")
    elif chunk_reports == doc_reports:
        result.ok("Chunks match documents (all documents chunked)")
    else:
        missing = doc_reports - chunk_reports
        extra = chunk_reports - doc_reports
        if missing:
            result.warn(f"{len(missing)} documents not chunked")
        if extra:
            result.warn(f"{len(extra)} chunk files without documents")

    return result


def main():
    """Run all validations."""
    parser = argparse.ArgumentParser(
        description="Validate chunking pipeline output"
    )
    parser.add_argument(
        "--sample", "-s",
        type=int,
        default=20,
        help="Sample size for file validation"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging"
    )
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="Skip database validation"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Paths
    extraction_base = Path(__file__).parent.parent
    json_data = extraction_base / "json_data"
    pages_path = json_data / "pages.jsonl"
    documents_dir = json_data / "documents"
    chunks_dir = json_data / "chunks"

    print("\n" + "=" * 60)
    print("Chunking Pipeline Validation")
    print("=" * 60)

    results = []

    # Validate pages.jsonl
    print("\nValidating pages.jsonl...")
    results.append(validate_pages_jsonl(pages_path, args.sample))

    # Validate documents
    print("\nValidating documents/...")
    results.append(validate_documents(documents_dir, args.sample))

    # Validate chunks
    print("\nValidating chunks/...")
    results.append(validate_chunks(chunks_dir, args.sample))

    # Validate database
    if not args.skip_db:
        print("\nValidating database...")
        try:
            conn = init_db(DB_PATH)
            results.append(validate_database(conn))
            conn.close()
        except Exception as e:
            r = ValidationResult("database")
            r.fail(f"Could not connect: {e}")
            results.append(r)

    # Validate cross-consistency
    print("\nValidating cross-consistency...")
    results.append(validate_cross_consistency(pages_path, documents_dir, chunks_dir))

    # Summary
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)

    total_passed = 0
    total_failed = 0
    total_warnings = 0

    for result in results:
        print(f"\n{result.summary()}")
        total_passed += result.passed
        total_failed += result.failed
        total_warnings += result.warnings

        if result.errors and args.verbose:
            for error in result.errors[:5]:
                print(f"    - {error}")
            if len(result.errors) > 5:
                print(f"    ... and {len(result.errors) - 5} more errors")

    print("\n" + "-" * 60)
    overall = "PASS" if total_failed == 0 else "FAIL"
    print(
        f"Overall: {overall} "
        f"(passed={total_passed}, failed={total_failed}, warnings={total_warnings})"
    )
    print()

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
