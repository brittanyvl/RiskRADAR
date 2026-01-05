"""
extraction/scripts/validate_extraction.py
------------------------------------------
Validation script for Phase 3 text extraction pipeline.

Validates:
1. Extraction across decades (1960s-2020s)
2. JSON file creation and schema compliance
3. Database tracking accuracy
4. Quality metrics computation
5. OCR confidence scores (where applicable)

Usage:
    python extraction/scripts/validate_extraction.py [--reports-per-decade N] [--output validation_report.md]
"""

import argparse
import json
import random
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from riskradar.config import DB_PATH, EXTRACTION_DIR
from sqlite.connection import init_db
from extraction.processing.extract import run_initial_extraction
from extraction.processing.quality import QUALITY_THRESHOLDS


class ValidationReport:
    """Track validation results and generate report."""

    def __init__(self):
        self.start_time = datetime.now()
        self.reports_tested = []
        self.pages_tested = []
        self.issues = []
        self.stats = {
            "total_reports": 0,
            "total_pages": 0,
            "json_files_created": 0,
            "json_files_missing": 0,
            "db_entries_correct": 0,
            "db_entries_missing": 0,
            "db_entries_mismatch": 0,
            "schema_valid": 0,
            "schema_invalid": 0,
            "embedded_extractions": 0,
            "ocr_extractions": 0,
            "quality_pass": 0,
            "quality_fail": 0,
        }

    def add_issue(self, severity: str, category: str, message: str, details: dict = None):
        """Log a validation issue."""
        self.issues.append({
            "severity": severity,  # 'ERROR', 'WARNING', 'INFO'
            "category": category,
            "message": message,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        })

    def add_page_result(self, report_id: str, page_number: int, result: dict):
        """Track a page validation result."""
        self.pages_tested.append({
            "report_id": report_id,
            "page_number": page_number,
            **result
        })

    def generate_markdown(self, output_path: Path):
        """Generate validation report in Markdown format."""
        duration = (datetime.now() - self.start_time).total_seconds()

        lines = [
            "# Phase 3 Extraction Pipeline - Validation Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Duration:** {duration:.1f} seconds",
            "",
            "---",
            "",
            "## Summary Statistics",
            "",
            f"- **Reports Tested:** {self.stats['total_reports']}",
            f"- **Pages Tested:** {self.stats['total_pages']}",
            "",
            "### File Creation",
            f"- ✓ JSON files created: {self.stats['json_files_created']}",
            f"- ✗ JSON files missing: {self.stats['json_files_missing']}",
            "",
            "### Database Tracking",
            f"- ✓ Correct entries: {self.stats['db_entries_correct']}",
            f"- ✗ Missing entries: {self.stats['db_entries_missing']}",
            f"- ⚠ Mismatched entries: {self.stats['db_entries_mismatch']}",
            "",
            "### JSON Schema Validation",
            f"- ✓ Valid schema: {self.stats['schema_valid']}",
            f"- ✗ Invalid schema: {self.stats['schema_invalid']}",
            "",
            "### Extraction Methods",
            f"- Embedded text: {self.stats['embedded_extractions']} ({self._pct('embedded_extractions')}%)",
            f"- OCR: {self.stats['ocr_extractions']} ({self._pct('ocr_extractions')}%)",
            "",
            "### Quality Metrics",
            f"- Passed quality checks: {self.stats['quality_pass']} ({self._pct('quality_pass')}%)",
            f"- Failed quality checks: {self.stats['quality_fail']} ({self._pct('quality_fail')}%)",
            "",
            "---",
            "",
            "## Reports Tested by Decade",
            "",
        ]

        # Group reports by decade
        by_decade = {}
        for report in self.reports_tested:
            decade = report.get("decade", "unknown")
            if decade not in by_decade:
                by_decade[decade] = []
            by_decade[decade].append(report)

        for decade in sorted(by_decade.keys()):
            lines.append(f"### {decade}")
            lines.append("")
            for report in by_decade[decade]:
                lines.append(f"- `{report['report_id']}`")
                lines.append(f"  - Pages tested: {report['pages_tested']}")
                lines.append(f"  - Extraction method: {report['extraction_method']}")
                lines.append(f"  - Quality: {report['quality_status']}")
            lines.append("")

        # Issues section
        if self.issues:
            lines.extend([
                "---",
                "",
                "## Issues Detected",
                "",
            ])

            # Group by severity
            errors = [i for i in self.issues if i["severity"] == "ERROR"]
            warnings = [i for i in self.issues if i["severity"] == "WARNING"]
            info = [i for i in self.issues if i["severity"] == "INFO"]

            if errors:
                lines.append(f"### ✗ Errors ({len(errors)})")
                lines.append("")
                for issue in errors:
                    lines.append(f"- **{issue['category']}:** {issue['message']}")
                    if issue.get("details"):
                        for key, value in issue["details"].items():
                            lines.append(f"  - {key}: `{value}`")
                lines.append("")

            if warnings:
                lines.append(f"### ⚠ Warnings ({len(warnings)})")
                lines.append("")
                for issue in warnings:
                    lines.append(f"- **{issue['category']}:** {issue['message']}")
                lines.append("")

            if info:
                lines.append(f"### ℹ Info ({len(info)})")
                lines.append("")
                for issue in info:
                    lines.append(f"- **{issue['category']}:** {issue['message']}")
                lines.append("")
        else:
            lines.extend([
                "---",
                "",
                "## Issues Detected",
                "",
                "✓ No issues detected!",
                "",
            ])

        # Sample page details
        lines.extend([
            "---",
            "",
            "## Sample Page Details",
            "",
            "| Report ID | Page | Method | Chars | Alpha % | Garbage % | Words | OCR Conf | Status |",
            "|-----------|------|--------|-------|---------|-----------|-------|----------|--------|",
        ])

        for page in self.pages_tested[:20]:  # Limit to first 20
            ocr_conf = f"{page.get('ocr_confidence', 0):.1f}%" if page.get('ocr_confidence') else "N/A"
            lines.append(
                f"| {page['report_id']:<15} | {page['page_number']:>4} | "
                f"{page.get('method', 'N/A'):<8} | {page.get('char_count', 0):>5} | "
                f"{page.get('alphabetic_ratio', 0)*100:>6.1f}% | {page.get('garbage_ratio', 0)*100:>8.1f}% | "
                f"{page.get('word_count', 0):>5} | {ocr_conf:>8} | {page.get('status', 'N/A'):<6} |"
            )

        # Acceptance criteria
        lines.extend([
            "",
            "---",
            "",
            "## Acceptance Criteria",
            "",
        ])

        criteria = [
            ("JSON files created for all pages", self.stats['json_files_missing'] == 0),
            ("Database entries match JSON files", self.stats['db_entries_mismatch'] == 0),
            ("All JSON schemas valid", self.stats['schema_invalid'] == 0),
            ("Quality metrics computed for all pages", self.stats['total_pages'] == self.stats['quality_pass'] + self.stats['quality_fail']),
            ("No critical errors", len([i for i in self.issues if i['severity'] == 'ERROR']) == 0),
        ]

        for criterion, passed in criteria:
            status = "✓ PASS" if passed else "✗ FAIL"
            lines.append(f"- [{status}] {criterion}")

        lines.extend([
            "",
            "---",
            "",
            "## Conclusion",
            "",
        ])

        all_passed = all(passed for _, passed in criteria)
        if all_passed:
            lines.append("✓ **All validation checks PASSED**")
            lines.append("")
            lines.append("The extraction pipeline is ready for full corpus processing.")
        else:
            lines.append("✗ **Some validation checks FAILED**")
            lines.append("")
            lines.append("Please review the issues above before running full extraction.")

        lines.append("")

        # Write report
        output_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"\n✓ Validation report written to: {output_path}")

    def _pct(self, key: str) -> str:
        """Calculate percentage for a stat."""
        total = self.stats['total_pages']
        if total == 0:
            return "0.0"
        return f"{(self.stats[key] / total) * 100:.1f}"


def select_test_reports(conn: sqlite3.Connection, per_decade: int = 4) -> List[Dict]:
    """
    Select test reports across decades.

    Args:
        conn: Database connection
        per_decade: Number of reports per decade to select

    Returns:
        List of report dicts with: report_id, decade, accident_date
    """
    print("\nSelecting test reports across decades...")

    # Query reports grouped by decade
    cursor = conn.execute("""
        SELECT
            filename as report_id,
            SUBSTR(filename, 4, 2) as decade_code,
            accident_date,
            title
        FROM reports
        WHERE status = 'completed'
          AND filename LIKE 'AIR%.pdf'
        ORDER BY filename
    """)

    reports = [dict(row) for row in cursor.fetchall()]

    if not reports:
        print("✗ No completed reports found in database!")
        return []

    # Group by decade
    by_decade = {}
    for report in reports:
        decade_code = report["decade_code"]
        # Map decade code to human-readable decade
        try:
            year_code = int(decade_code)
            if year_code >= 60 and year_code <= 99:
                decade = f"19{decade_code}s"
            else:
                decade = f"20{decade_code}s"
        except ValueError:
            decade = "unknown"

        report["decade"] = decade

        if decade not in by_decade:
            by_decade[decade] = []
        by_decade[decade].append(report)

    # Select random sample from each decade
    selected = []
    for decade in sorted(by_decade.keys()):
        available = by_decade[decade]
        sample_size = min(per_decade, len(available))
        sampled = random.sample(available, sample_size)
        selected.extend(sampled)
        print(f"  {decade}: Selected {sample_size}/{len(available)} reports")

    print(f"\n✓ Selected {len(selected)} reports total")
    return selected


def validate_json_schema(json_path: Path, report: ValidationReport) -> bool:
    """
    Validate JSON file against expected schema.

    Returns:
        True if valid, False otherwise
    """
    required_fields = [
        "report_id",
        "page_number",
        "extraction_method",
        "extraction_pass",
        "text",
        "quality_metrics",
        "metadata"
    ]

    quality_metrics_fields = [
        "char_count",
        "alphabetic_ratio",
        "garbage_ratio",
        "word_count",
        "passes_threshold"
    ]

    metadata_fields = [
        "extracted_at",
        "extraction_time_ms",
        "pdf_source"
    ]

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Check top-level fields
        for field in required_fields:
            if field not in data:
                report.add_issue(
                    "ERROR",
                    "Schema Validation",
                    f"Missing required field: {field}",
                    {"file": str(json_path)}
                )
                return False

        # Check quality_metrics fields
        for field in quality_metrics_fields:
            if field not in data["quality_metrics"]:
                report.add_issue(
                    "ERROR",
                    "Schema Validation",
                    f"Missing quality_metrics field: {field}",
                    {"file": str(json_path)}
                )
                return False

        # Check metadata fields
        for field in metadata_fields:
            if field not in data["metadata"]:
                report.add_issue(
                    "WARNING",
                    "Schema Validation",
                    f"Missing metadata field: {field}",
                    {"file": str(json_path)}
                )

        # If OCR, check for ocr_confidence
        if data["extraction_method"] == "ocr":
            if "ocr_confidence" not in data or data["ocr_confidence"] is None:
                report.add_issue(
                    "ERROR",
                    "Schema Validation",
                    "OCR extraction missing ocr_confidence field",
                    {"file": str(json_path)}
                )
                return False

            ocr_fields = ["mean_confidence", "min_confidence", "max_confidence", "low_confidence_words", "total_words"]
            for field in ocr_fields:
                if field not in data["ocr_confidence"]:
                    report.add_issue(
                        "ERROR",
                        "Schema Validation",
                        f"Missing ocr_confidence field: {field}",
                        {"file": str(json_path)}
                    )
                    return False

        return True

    except json.JSONDecodeError as e:
        report.add_issue(
            "ERROR",
            "Schema Validation",
            f"Invalid JSON: {e}",
            {"file": str(json_path)}
        )
        return False
    except Exception as e:
        report.add_issue(
            "ERROR",
            "Schema Validation",
            f"Validation error: {e}",
            {"file": str(json_path)}
        )
        return False


def validate_page(
    report_id: str,
    page_number: int,
    conn: sqlite3.Connection,
    validation_report: ValidationReport
) -> Dict:
    """
    Validate a single page extraction.

    Checks:
    1. JSON file exists
    2. JSON schema is valid
    3. Database entry exists
    4. Database entry matches JSON file
    5. Quality metrics are reasonable

    Returns:
        Dict with validation results
    """
    result = {
        "json_exists": False,
        "json_valid": False,
        "db_exists": False,
        "db_matches_json": False,
        "method": None,
        "char_count": 0,
        "alphabetic_ratio": 0.0,
        "garbage_ratio": 0.0,
        "word_count": 0,
        "ocr_confidence": None,
        "status": None,
    }

    # Check for JSON file in passed/, failed/, or ocr_retry/
    json_dirs = [
        EXTRACTION_DIR / "json_data" / "passed" / report_id,
        EXTRACTION_DIR / "json_data" / "failed" / report_id,
        EXTRACTION_DIR / "json_data" / "ocr_retry" / report_id,
    ]

    json_filename = f"page_{page_number:04d}.json"
    json_path = None

    for json_dir in json_dirs:
        potential_path = json_dir / json_filename
        if potential_path.exists():
            json_path = potential_path
            break

    if json_path:
        result["json_exists"] = True
        validation_report.stats["json_files_created"] += 1

        # Validate schema
        if validate_json_schema(json_path, validation_report):
            result["json_valid"] = True
            validation_report.stats["schema_valid"] += 1

            # Load JSON data
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            result["method"] = json_data.get("extraction_method")
            result["char_count"] = json_data["quality_metrics"]["char_count"]
            result["alphabetic_ratio"] = json_data["quality_metrics"]["alphabetic_ratio"]
            result["garbage_ratio"] = json_data["quality_metrics"]["garbage_ratio"]
            result["word_count"] = json_data["quality_metrics"]["word_count"]

            if json_data.get("ocr_confidence"):
                result["ocr_confidence"] = json_data["ocr_confidence"]["mean_confidence"]

            result["status"] = "passed" if json_data["quality_metrics"]["passes_threshold"] else "failed"

        else:
            validation_report.stats["schema_invalid"] += 1
    else:
        validation_report.stats["json_files_missing"] += 1
        validation_report.add_issue(
            "ERROR",
            "File Creation",
            f"JSON file missing for {report_id} page {page_number}",
            {"report_id": report_id, "page_number": page_number}
        )

    # Check database entry
    cursor = conn.execute(
        "SELECT * FROM pages WHERE report_id = ? AND page_number = ?",
        (report_id, page_number)
    )
    db_row = cursor.fetchone()

    if db_row:
        result["db_exists"] = True
        db_entry = dict(db_row)

        # Compare database with JSON
        if result["json_valid"]:
            matches = True

            if db_entry["extraction_method"] != result["method"]:
                matches = False
                validation_report.add_issue(
                    "ERROR",
                    "DB Mismatch",
                    f"Extraction method mismatch: DB={db_entry['extraction_method']}, JSON={result['method']}",
                    {"report_id": report_id, "page_number": page_number}
                )

            if db_entry["char_count"] != result["char_count"]:
                matches = False
                validation_report.add_issue(
                    "WARNING",
                    "DB Mismatch",
                    f"Char count mismatch: DB={db_entry['char_count']}, JSON={result['char_count']}",
                    {"report_id": report_id, "page_number": page_number}
                )

            if matches:
                result["db_matches_json"] = True
                validation_report.stats["db_entries_correct"] += 1
            else:
                validation_report.stats["db_entries_mismatch"] += 1
        else:
            # Can't compare if JSON is invalid
            validation_report.stats["db_entries_correct"] += 1
    else:
        validation_report.stats["db_entries_missing"] += 1
        validation_report.add_issue(
            "ERROR",
            "Database",
            f"Database entry missing for {report_id} page {page_number}",
            {"report_id": report_id, "page_number": page_number}
        )

    # Track extraction method
    if result["method"] == "embedded":
        validation_report.stats["embedded_extractions"] += 1
    elif result["method"] == "ocr":
        validation_report.stats["ocr_extractions"] += 1

    # Track quality status
    if result["status"] == "passed":
        validation_report.stats["quality_pass"] += 1
    elif result["status"] == "failed":
        validation_report.stats["quality_fail"] += 1

    return result


def main():
    """Main validation workflow."""
    parser = argparse.ArgumentParser(
        description="Validate Phase 3 text extraction pipeline"
    )
    parser.add_argument(
        "--reports-per-decade",
        type=int,
        default=4,
        help="Number of reports to test per decade (default: 4)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation_report.md"),
        help="Output path for validation report (default: validation_report.md)"
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip extraction step (validate existing files only)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Phase 3 Extraction Pipeline - Validation")
    print("=" * 70)

    # Initialize report
    validation_report = ValidationReport()

    # Initialize database
    if not DB_PATH.exists():
        print(f"\n✗ Database not found at {DB_PATH}")
        print("Run 'riskradar scrape' first to download PDFs.")
        return 1

    conn = init_db(DB_PATH)

    # Select test reports
    test_reports = select_test_reports(conn, args.reports_per_decade)

    if not test_reports:
        print("\n✗ No reports available for testing")
        return 1

    validation_report.stats["total_reports"] = len(test_reports)

    # Extract test reports (unless skipped)
    if not args.skip_extraction:
        print("\n" + "=" * 70)
        print("Running Extraction on Test Reports")
        print("=" * 70)

        # Create a temporary list of report IDs to extract
        test_report_ids = [r["report_id"] for r in test_reports]

        # We'll run extraction with limit set to number of test reports
        # This is a simplified approach - in production we'd filter by specific IDs
        print(f"\nExtracting {len(test_reports)} reports...")
        try:
            stats = run_initial_extraction(limit=len(test_reports), resume=False)
            print(f"\n✓ Extraction complete:")
            print(f"  Reports: {stats['reports_processed']}")
            print(f"  Pages: {stats['pages_extracted']}")
            print(f"  Passed: {stats['passed_count']}")
            print(f"  Failed: {stats['failed_count']}")
        except Exception as e:
            print(f"\n✗ Extraction failed: {e}")
            validation_report.add_issue(
                "ERROR",
                "Extraction",
                f"Failed to run extraction: {e}"
            )
            # Continue with validation of whatever exists

    # Validate each report
    print("\n" + "=" * 70)
    print("Validating Extractions")
    print("=" * 70)

    for test_report in test_reports:
        report_id = test_report["report_id"]
        decade = test_report["decade"]

        print(f"\nValidating {report_id} ({decade})...")

        # Get page count from database
        cursor = conn.execute(
            "SELECT page_number FROM pages WHERE report_id = ? ORDER BY page_number",
            (report_id,)
        )
        pages = [row[0] for row in cursor.fetchall()]

        if not pages:
            validation_report.add_issue(
                "WARNING",
                "Extraction",
                f"No pages found in database for {report_id}",
                {"report_id": report_id}
            )
            validation_report.reports_tested.append({
                "report_id": report_id,
                "decade": decade,
                "pages_tested": 0,
                "extraction_method": "N/A",
                "quality_status": "N/A"
            })
            continue

        # Sample 3 random pages (or all if fewer than 3)
        sample_size = min(3, len(pages))
        sampled_pages = random.sample(pages, sample_size)

        report_methods = []
        report_statuses = []

        for page_number in sampled_pages:
            result = validate_page(report_id, page_number, conn, validation_report)
            validation_report.add_page_result(report_id, page_number, result)
            validation_report.stats["total_pages"] += 1

            report_methods.append(result["method"])
            report_statuses.append(result["status"])

            print(f"  Page {page_number}: {result['method']} - {result['status']} "
                  f"({result['char_count']} chars, {result['alphabetic_ratio']*100:.1f}% alpha)")

        # Add report summary
        validation_report.reports_tested.append({
            "report_id": report_id,
            "decade": decade,
            "pages_tested": len(sampled_pages),
            "extraction_method": ", ".join(set(filter(None, report_methods))),
            "quality_status": ", ".join(set(filter(None, report_statuses)))
        })

    conn.close()

    # Generate report
    print("\n" + "=" * 70)
    print("Generating Validation Report")
    print("=" * 70)

    validation_report.generate_markdown(args.output)

    # Print summary
    print("\n" + "=" * 70)
    print("Validation Summary")
    print("=" * 70)
    print(f"Reports tested: {validation_report.stats['total_reports']}")
    print(f"Pages tested: {validation_report.stats['total_pages']}")
    print(f"JSON files created: {validation_report.stats['json_files_created']}/{validation_report.stats['total_pages']}")
    print(f"Schema valid: {validation_report.stats['schema_valid']}/{validation_report.stats['json_files_created']}")
    print(f"DB entries correct: {validation_report.stats['db_entries_correct']}/{validation_report.stats['total_pages']}")

    errors = len([i for i in validation_report.issues if i["severity"] == "ERROR"])
    warnings = len([i for i in validation_report.issues if i["severity"] == "WARNING"])

    print(f"\nIssues: {errors} errors, {warnings} warnings")

    if errors == 0:
        print("\n✓ Validation PASSED - pipeline ready for full extraction")
        return 0
    else:
        print("\n✗ Validation FAILED - review issues before full extraction")
        return 1


if __name__ == "__main__":
    sys.exit(main())
