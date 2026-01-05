"""
riskradar/cli.py
----------------
Command-line interface for RiskRADAR pipeline.

Usage:
    riskradar scrape              # Download PDFs (resume from last position)
    riskradar scrape --fresh      # Start from beginning
    riskradar scrape --limit 10   # Download only 10 PDFs
    riskradar status              # Show pipeline progress
    riskradar extract initial     # Pass 1: Initial extraction
    riskradar extract ocr-retry   # Pass 2: OCR retry on failed pages
    riskradar extract all         # Full pipeline
    riskradar analytics quality   # Quality summary report
"""
import argparse
import sys


def cmd_scrape(args):
    """Run the NTSB PDF scraping pipeline."""
    from collection.scrape import scrape_all_reports

    stats = scrape_all_reports(
        resume=not args.fresh,
        fresh=args.fresh,
        limit=args.limit,
        retry_failed=args.retry_failed,
    )

    print(f"\nScrape complete:")
    print(f"  Downloaded: {stats['downloaded']}")
    print(f"  Skipped:    {stats['skipped']}")
    print(f"  Failed:     {stats['failed']}")


def cmd_status(args):
    """Show current pipeline status."""
    from sqlite.connection import init_db
    from sqlite.queries import get_scrape_stats, get_resume_point
    from riskradar.config import DB_PATH

    TOTAL_REPORTS = 781
    TOTAL_PAGES = 79

    if not DB_PATH.exists():
        print("RiskRADAR Pipeline Status")
        print("=" * 50)
        print(f"Database: {DB_PATH}")
        print()
        print("Database not initialized yet.")
        print("Run 'riskradar scrape' to start downloading PDFs.")
        return

    try:
        conn = init_db(DB_PATH)
        stats = get_scrape_stats(conn)
        page, idx = get_resume_point(conn)

        completed = stats.get('completed', 0)
        pending = stats.get('pending', 0)
        failed = stats.get('failed', 0)
        total_in_db = stats.get('total', 0)

        # Calculate progress
        processed = completed + failed
        remaining = TOTAL_REPORTS - processed
        pct = (processed / TOTAL_REPORTS) * 100 if TOTAL_REPORTS > 0 else 0

        print("RiskRADAR Pipeline Status")
        print("=" * 50)
        print(f"Database: {DB_PATH}")
        print()
        print("Progress:")
        print(f"  Processed:  {processed}/{TOTAL_REPORTS} ({pct:.1f}%)")
        print(f"  Remaining:  {remaining}")
        print(f"  Current:    Page {page + 1}/{TOTAL_PAGES}, Report {idx}")
        print()
        print("Breakdown:")
        print(f"  Completed:  {completed}")
        print(f"  Failed:     {failed}")
        print(f"  Pending:    {pending}")
        print()

        # Estimate time remaining (assuming ~5s per download)
        if remaining > 0:
            est_seconds = remaining * 5
            est_hours = est_seconds // 3600
            est_mins = (est_seconds % 3600) // 60
            print(f"Est. time remaining: ~{est_hours}h {est_mins}m (at ~5s/download)")

        # Show extraction stats if available
        try:
            from extraction.processing.analytics import get_quality_summary, get_latest_run
            from sqlite.queries import get_latest_run as get_run

            cursor = conn.execute("SELECT COUNT(*) FROM pages")
            page_count = cursor.fetchone()[0]

            if page_count > 0:
                print("\nExtraction Status:")
                summary = get_quality_summary(conn)

                # Get latest run info
                latest_run = get_run(conn)
                if latest_run:
                    status = latest_run["status"]
                    run_type = latest_run["run_type"]
                    print(f"  Latest run:  #{latest_run['id']} ({run_type}) - {status}")

                print(f"  Total pages:    {summary.get('total_pages', 0):,}")
                print(f"    Embedded:     {summary.get('embedded_count', 0):,}")
                print(f"    OCR:          {summary.get('ocr_count', 0):,}")
                print(f"    Failed:       {summary.get('failed_count', 0):,}")
        except ImportError:
            pass  # Phase 3 modules not yet available

        conn.close()
    except Exception as e:
        print(f"Error reading database: {e}")
        print("Run 'riskradar scrape' first to initialize.")


def cmd_extract(args):
    """Run PDF text extraction pipeline."""
    from extraction.processing.extract import run_initial_extraction, run_ocr_retry, run_full_pipeline

    if args.extract_command == "initial":
        stats = run_initial_extraction(limit=args.limit, resume=args.resume)
        print(f"\nPass 1 Complete:")
        print(f"  Reports: {stats['reports_processed']}")
        print(f"  Pages:   {stats['pages_extracted']}")
        print(f"  Passed:  {stats['passed_count']}")
        print(f"  Failed:  {stats['failed_count']}")

    elif args.extract_command == "ocr-retry":
        stats = run_ocr_retry(limit=args.limit)
        print(f"\nPass 2 Complete:")
        print(f"  Pages OCR'd:       {stats['pages_processed']}")
        print(f"  Low confidence:    {stats['low_confidence_count']}")

    elif args.extract_command == "all":
        stats = run_full_pipeline(limit=args.limit, resume=args.resume)
        print(f"\nFull Pipeline Complete!")
        print(f"  Pass 1 - Reports: {stats['pass1_reports']}")
        print(f"  Pass 1 - Passed:  {stats['pass1_passed']}")
        print(f"  Pass 1 - Failed:  {stats['pass1_failed']}")
        print(f"  Pass 2 - OCR'd:   {stats['pass2_ocr']}")


def cmd_analytics(args):
    """Run analytics and reporting queries."""
    from sqlite.connection import init_db
    from riskradar.config import DB_PATH
    from extraction.processing.analytics import (
        print_quality_summary,
        print_quality_by_decade,
        get_low_confidence_pages,
        get_extraction_runs,
        get_run_errors,
    )

    conn = init_db(DB_PATH)

    if args.analytics_command == "quality":
        print_quality_summary(conn)

    elif args.analytics_command == "by-decade":
        print_quality_by_decade(conn)

    elif args.analytics_command == "low-confidence":
        threshold = args.threshold if hasattr(args, 'threshold') else 60.0
        limit = args.limit if hasattr(args, 'limit') else 100
        pages = get_low_confidence_pages(conn, threshold, limit)

        if pages:
            print(f"\nLow Confidence Pages (< {threshold}%):")
            print("=" * 70)
            for page in pages:
                print(f"  {page['report_id']:<20} p{page['page_number']:<4}  "
                      f"{page['mean_ocr_confidence']:>5.1f}%  "
                      f"({page['low_confidence_word_count']} low-conf words)")
        else:
            print(f"No pages found with confidence < {threshold}%")

    elif args.analytics_command == "runs":
        runs = get_extraction_runs(conn)

        if runs:
            print("\nExtraction Run History:")
            print("=" * 80)
            print(f"{'ID':<5} {'Type':<15} {'Started':<20} {'Status':<12} {'Pages':>8} {'Passed':>8} {'Failed':>8}")
            print("-" * 80)
            for run in runs:
                started = run['started_at'][:19] if run['started_at'] else "N/A"
                print(f"{run['id']:<5} {run['run_type']:<15} {started:<20} "
                      f"{run['status']:<12} {run['total_pages']:>8} "
                      f"{run['passed_pages']:>8} {run['failed_pages']:>8}")
        else:
            print("No extraction runs found.")

    elif args.analytics_command == "errors":
        run_id = args.run_id if hasattr(args, 'run_id') else None
        errors = get_run_errors(conn, run_id)

        if errors:
            title = f"Extraction Errors (Run #{run_id})" if run_id else "All Extraction Errors"
            print(f"\n{title}:")
            print("=" * 80)
            for error in errors[:50]:  # Limit to 50
                page_info = f"p{error['page_number']}" if error['page_number'] is not None else "ALL"
                print(f"  {error['report_id']:<20} {page_info:<6} {error['error_type']:<20} "
                      f"{error['error_message'][:30]}")
            if len(errors) > 50:
                print(f"\n... and {len(errors) - 50} more errors")
        else:
            print("No errors found.")

    conn.close()


def cmd_app(args):
    """Launch the Streamlit app."""
    import subprocess
    from pathlib import Path

    app_path = Path(__file__).parent.parent / "app" / "main.py"
    if not app_path.exists():
        print(f"App not found at {app_path}")
        print("The Streamlit app will be implemented in Phase 8.")
        return

    subprocess.run(["streamlit", "run", str(app_path)])


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog="riskradar",
        description="RiskRADAR - NTSB Aviation Accident Report Analysis Pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # scrape command
    scrape_parser = subparsers.add_parser(
        "scrape",
        help="Download NTSB aviation PDFs",
    )
    scrape_parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start from beginning, ignore previous progress",
    )
    scrape_parser.add_argument(
        "--limit",
        type=int,
        metavar="N",
        help="Maximum number of PDFs to download",
    )
    scrape_parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Only retry previously failed downloads",
    )
    scrape_parser.set_defaults(func=cmd_scrape)

    # status command
    status_parser = subparsers.add_parser(
        "status",
        help="Show pipeline progress",
    )
    status_parser.set_defaults(func=cmd_status)

    # app command
    app_parser = subparsers.add_parser(
        "app",
        help="Launch Streamlit app",
    )
    app_parser.set_defaults(func=cmd_app)

    # extract command
    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract text from downloaded PDFs",
    )
    extract_subparsers = extract_parser.add_subparsers(
        dest="extract_command",
        help="Extraction pass to run"
    )

    # extract initial
    extract_initial = extract_subparsers.add_parser(
        "initial",
        help="Pass 1: Initial extraction (embedded text)",
    )
    extract_initial.add_argument("--limit", type=int, help="Process only N reports")
    extract_initial.add_argument("--resume", action="store_true", default=True, help="Resume from last position")

    # extract ocr-retry
    extract_ocr = extract_subparsers.add_parser(
        "ocr-retry",
        help="Pass 2: OCR retry on failed pages",
    )
    extract_ocr.add_argument("--limit", type=int, help="Process only N pages")

    # extract all
    extract_all = extract_subparsers.add_parser(
        "all",
        help="Full pipeline: initial + OCR retry",
    )
    extract_all.add_argument("--limit", type=int, help="Process only N reports")
    extract_all.add_argument("--resume", action="store_true", default=True, help="Resume from last position")

    extract_parser.set_defaults(func=cmd_extract)

    # analytics command
    analytics_parser = subparsers.add_parser(
        "analytics",
        help="View extraction quality analytics",
    )
    analytics_subparsers = analytics_parser.add_subparsers(
        dest="analytics_command",
        help="Analytics report to generate"
    )

    # analytics quality
    analytics_quality = analytics_subparsers.add_parser(
        "quality",
        help="Overall quality summary",
    )

    # analytics by-decade
    analytics_decade = analytics_subparsers.add_parser(
        "by-decade",
        help="Quality breakdown by decade",
    )

    # analytics low-confidence
    analytics_lowconf = analytics_subparsers.add_parser(
        "low-confidence",
        help="Pages with low OCR confidence",
    )
    analytics_lowconf.add_argument("--threshold", type=float, default=60.0, help="Confidence threshold (default: 60)")
    analytics_lowconf.add_argument("--limit", type=int, default=100, help="Max results (default: 100)")

    # analytics runs
    analytics_runs = analytics_subparsers.add_parser(
        "runs",
        help="Extraction run history",
    )

    # analytics errors
    analytics_errors = analytics_subparsers.add_parser(
        "errors",
        help="Extraction error log",
    )
    analytics_errors.add_argument("--run-id", type=int, help="Filter by run ID")

    analytics_parser.set_defaults(func=cmd_analytics)

    # Parse and execute
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
