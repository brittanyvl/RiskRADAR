"""
exploration/cli.py
------------------
CLI entry point for freeform topic exploration.

Usage:
    python -m exploration.cli discover       # Run BERTopic topic discovery
    python -m exploration.cli stats          # Show exploration statistics
    python -m exploration.cli visualize      # Generate visualizations
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(command: str, verbose: bool = False) -> logging.Logger:
    """Configure logging to console and file."""
    project_root = Path(__file__).parent.parent
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_file = logs_dir / f"exploration_{command}_{timestamp}.log"

    log_level = logging.DEBUG if verbose else logging.INFO
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers = []

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    ))
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    root_logger.addHandler(file_handler)

    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")
    return logger


def cmd_discover(args, logger):
    """Run BERTopic topic discovery."""
    from .discover import discover_topics

    logger.info("Starting Topic Exploration")

    results = discover_topics(
        run_id=args.run_id,
        generate_viz=not args.no_viz,
    )

    print("\n" + "=" * 60)
    print("EXPLORATION COMPLETE")
    print("=" * 60)
    print(f"Chunks processed: {results['n_chunks']:,}")
    print(f"Topics discovered: {results['n_topics']}")
    print(f"Outlier chunks: {results['n_outliers']:,} ({results['outlier_pct']:.1f}%)")
    print("\nOutput files:")
    for name, path in results['paths'].items():
        print(f"  {name}: {path}")
    print("=" * 60)

    return 0


def cmd_stats(args, logger):
    """Show exploration statistics."""
    from .config import EXPLORATION_DATA_DIR
    from .prepare_data import load_chunks, get_section_statistics

    print("\n" + "=" * 60)
    print("EXPLORATION STATISTICS")
    print("=" * 60)

    try:
        chunks = load_chunks()
        print(f"\nTotal chunks: {len(chunks):,}")

        section_stats = get_section_statistics(chunks)
        print("\nTop sections by chunk count:")
        print(section_stats.head(15).to_string())
    except Exception as e:
        print(f"Could not load chunk statistics: {e}")

    print("\n" + "-" * 40)
    print("Discovery Results:")

    for run_id in range(1, 10):
        model_path = EXPLORATION_DATA_DIR / f"topic_model_run{run_id}.pkl"
        if model_path.exists():
            info_path = EXPLORATION_DATA_DIR / f"topic_info_run{run_id}.parquet"
            if info_path.exists():
                import pandas as pd
                topic_info = pd.read_parquet(info_path)
                n_topics = len(topic_info[topic_info["Topic"] != -1])
                print(f"  Run {run_id}: {n_topics} topics discovered")

    print("=" * 60)
    return 0


def cmd_visualize(args, logger):
    """Generate topic visualizations."""
    from .discover import load_topic_model, generate_visualizations
    from .prepare_data import prepare_discovery_dataset

    logger.info("Generating visualizations...")

    topic_model = load_topic_model(args.run_id)
    documents_df, embeddings = prepare_discovery_dataset(save_filtered=False)

    paths = generate_visualizations(
        topic_model, documents_df, embeddings, args.run_id
    )

    print("\n" + "=" * 60)
    print("VISUALIZATIONS GENERATED")
    print("=" * 60)
    for name, path in paths.items():
        print(f"  {name}: {path}")
    print("=" * 60)

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="RiskRADAR Topic Exploration - Freeform BERTopic Discovery"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # discover command
    discover_parser = subparsers.add_parser(
        "discover", help="Run BERTopic topic discovery"
    )
    discover_parser.add_argument(
        "--run-id", type=int, default=1, help="Run identifier"
    )
    discover_parser.add_argument(
        "--no-viz", action="store_true", help="Skip visualization generation"
    )

    # stats command
    subparsers.add_parser("stats", help="Show exploration statistics")

    # visualize command
    viz_parser = subparsers.add_parser(
        "visualize", help="Generate topic visualizations"
    )
    viz_parser.add_argument(
        "--run-id", type=int, default=1, help="Run identifier"
    )

    args = parser.parse_args()

    logger = setup_logging(args.command, args.verbose)

    commands = {
        "discover": cmd_discover,
        "stats": cmd_stats,
        "visualize": cmd_visualize,
    }

    try:
        exit_code = commands[args.command](args, logger)
        sys.exit(exit_code)
    except Exception as e:
        logger.exception(f"Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
