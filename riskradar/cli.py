"""
riskradar/cli.py
----------------
Command-line interface for RiskRADAR pipeline.

Usage:
    riskradar scrape              # Download PDFs (resume from last position)
    riskradar scrape --fresh      # Start from beginning
    riskradar scrape --limit 10   # Download only 10 PDFs
    riskradar status              # Show pipeline progress
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

        conn.close()
    except Exception as e:
        print(f"Error reading database: {e}")
        print("Run 'riskradar scrape' first to initialize.")


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

    # Parse and execute
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
