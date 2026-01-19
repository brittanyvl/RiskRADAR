"""
embeddings/cli.py
-----------------
CLI entry point for embedding pipeline.

Usage:
    python -m embeddings.cli embed minilm        # Embed with MiniLM
    python -m embeddings.cli embed mika          # Embed with MIKA
    python -m embeddings.cli embed both          # Embed with both models

    python -m embeddings.cli upload minilm       # Upload to Qdrant
    python -m embeddings.cli upload mika
    python -m embeddings.cli upload both

    python -m embeddings.cli all                 # Embed + upload both

    python -m embeddings.cli verify minilm       # Verify Qdrant collection
    python -m embeddings.cli stats               # Show statistics
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from riskradar.config import DB_PATH, get_qdrant_config
from sqlite.connection import init_db
from sqlite import queries

from .config import MODELS, get_parquet_path, CHUNKS_JSONL_PATH
from .embed import embed_all_models, embed_chunks
from .storage import get_parquet_stats
from .upload import upload_all_models, upload_embeddings, verify_collection


def setup_logging(log_name: str, verbose: bool = False) -> logging.Logger:
    """Configure logging to console and file."""
    # Determine paths
    project_root = Path(__file__).parent.parent
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_file = logs_dir / f"embed_{log_name}_{timestamp}.log"

    log_level = logging.DEBUG if verbose else logging.INFO

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers
    root_logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    ))
    root_logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    root_logger.addHandler(file_handler)

    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")

    return logger


def validate_prerequisites(command: str, logger: logging.Logger) -> bool:
    """
    Check prerequisites before running commands.

    Returns True if all prerequisites are met, False otherwise.
    """
    errors = []

    # For embed command: check chunks.jsonl exists
    if command in ("embed", "all"):
        if not CHUNKS_JSONL_PATH.exists():
            errors.append(
                f"Chunks file not found: {CHUNKS_JSONL_PATH}\n"
                "  Run Phase 4 (chunking) first:\n"
                "    python -m extraction.processing.chunk all"
            )

    # For upload/verify/all commands: check Qdrant credentials
    if command in ("upload", "verify", "all"):
        try:
            get_qdrant_config()
        except ValueError:
            errors.append(
                "Qdrant credentials not configured.\n"
                "  1. Create account at https://cloud.qdrant.io/\n"
                "  2. Copy .env.example to .env\n"
                "  3. Add QDRANT_URL and QDRANT_API_KEY"
            )

    # For upload command: check embeddings parquet exists
    if command == "upload":
        # Will be checked per-model in upload function
        pass

    if errors:
        for error in errors:
            logger.error(f"Prerequisite check failed:\n  {error}")
        return False

    return True


def cmd_embed(args):
    """Handle embed command."""
    logger = setup_logging(f"embed_{args.model}", args.verbose)
    logger.info(f"Embedding command: model={args.model}, limit={args.limit}")

    # Check prerequisites
    if not validate_prerequisites("embed", logger):
        return 1

    conn = init_db(DB_PATH)

    exit_code = 0
    try:
        if args.model == "both":
            stats = embed_all_models(limit=args.limit, conn=conn)
            for model_name, model_stats in stats.items():
                if model_stats.get("status") == "completed":
                    logger.info(f"{model_name}: {model_stats['embeddings_generated']} embeddings")
                else:
                    logger.error(f"{model_name}: {model_stats.get('error', 'failed')}")
                    exit_code = 1
        else:
            stats = embed_chunks(model_name=args.model, limit=args.limit, conn=conn)
            if stats.get("status") == "completed":
                logger.info(f"Completed: {stats['embeddings_generated']} embeddings")
            else:
                logger.error(f"Embedding failed: {stats.get('error', 'unknown error')}")
                exit_code = 1

    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        exit_code = 1

    finally:
        conn.close()

    return exit_code


def cmd_upload(args):
    """Handle upload command."""
    logger = setup_logging(f"upload_{args.model}", args.verbose)
    logger.info(f"Upload command: model={args.model}, recreate={args.recreate}")

    # Check prerequisites
    if not validate_prerequisites("upload", logger):
        return 1

    conn = init_db(DB_PATH)

    exit_code = 0
    try:
        if args.model == "both":
            stats = upload_all_models(recreate_collections=args.recreate, conn=conn)
            for model_name, model_stats in stats.items():
                if model_stats.get("status") == "completed":
                    logger.info(f"{model_name}: {model_stats['uploaded_vectors']} vectors uploaded")
                else:
                    logger.error(f"{model_name}: {model_stats.get('error', 'failed')}")
                    exit_code = 1
        else:
            stats = upload_embeddings(
                model_name=args.model,
                recreate_collection=args.recreate,
                conn=conn
            )
            if stats.get("status") == "completed":
                logger.info(f"Completed: {stats['uploaded_vectors']} vectors uploaded")
            else:
                logger.error(f"Upload failed: {stats.get('error', 'unknown error')}")
                exit_code = 1

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        exit_code = 1

    finally:
        conn.close()

    return exit_code


def cmd_all(args):
    """Handle all command (embed + upload)."""
    logger = setup_logging("all", args.verbose)
    logger.info(f"Full pipeline: limit={args.limit}")

    # Check prerequisites
    if not validate_prerequisites("all", logger):
        return 1

    conn = init_db(DB_PATH)

    try:
        # Step 1: Embed all models
        logger.info("=" * 60)
        logger.info("STEP 1: Generating embeddings")
        logger.info("=" * 60)
        embed_stats = embed_all_models(limit=args.limit, conn=conn)

        for model_name, stats in embed_stats.items():
            if stats.get("status") != "completed":
                logger.error(f"Embedding failed for {model_name}, skipping upload")
                return 1

        # Step 2: Upload all models
        logger.info("=" * 60)
        logger.info("STEP 2: Uploading to Qdrant")
        logger.info("=" * 60)
        upload_stats = upload_all_models(conn=conn)

        # Check for upload failures
        exit_code = 0
        for model_name, stats in upload_stats.items():
            if stats.get("status") != "completed":
                logger.error(f"Upload failed for {model_name}: {stats.get('error', 'unknown')}")
                exit_code = 1

        # Summary
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE" if exit_code == 0 else "PIPELINE COMPLETED WITH ERRORS")
        logger.info("=" * 60)
        for model_name in MODELS:
            e_stats = embed_stats.get(model_name, {})
            u_stats = upload_stats.get(model_name, {})
            logger.info(f"{model_name}:")
            logger.info(f"  Embeddings: {e_stats.get('embeddings_generated', 0)}")
            logger.info(f"  Uploaded: {u_stats.get('uploaded_vectors', 0)}")
            logger.info(f"  Collection: {u_stats.get('collection_name', 'N/A')}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1

    finally:
        conn.close()

    return exit_code


def cmd_verify(args):
    """Handle verify command."""
    logger = setup_logging(f"verify_{args.model}", args.verbose)

    # Check prerequisites
    if not validate_prerequisites("verify", logger):
        return 1

    if args.model == "both":
        models = list(MODELS.keys())
    else:
        models = [args.model]

    all_ok = True
    for model_name in models:
        result = verify_collection(model_name)

        if result.get("status") == "ok":
            logger.info(
                f"{model_name}: OK - {result['actual_count']} vectors "
                f"(expected {result['expected_count']})"
            )
        elif result.get("status") == "mismatch":
            logger.warning(
                f"{model_name}: MISMATCH - {result['actual_count']} vectors "
                f"(expected {result['expected_count']})"
            )
            all_ok = False
        else:
            logger.error(f"{model_name}: ERROR - {result.get('error', 'unknown')}")
            all_ok = False

    return 0 if all_ok else 1


def cmd_stats(args):
    """Handle stats command."""
    print("\n" + "=" * 60)
    print("Embedding Pipeline Statistics")
    print("=" * 60)

    # Parquet stats
    print("\nLocal Embeddings (Parquet):")
    for model_name in MODELS:
        stats = get_parquet_stats(model_name)
        if stats.get("exists"):
            print(f"  {model_name}:")
            print(f"    Path: {stats['path']}")
            print(f"    Vectors: {stats['num_rows']:,}")
            print(f"    Dimension: {stats.get('embedding_dimension', 'N/A')}")
            print(f"    Size: {stats['size_mb']:.1f} MB")
        else:
            print(f"  {model_name}: Not found")

    # Qdrant stats
    print("\nQdrant Collections:")
    for model_name in MODELS:
        result = verify_collection(model_name)
        if result.get("exists"):
            print(f"  {model_name}:")
            print(f"    Collection: {result['collection_name']}")
            print(f"    Vectors: {result['actual_count']:,}")
            print(f"    Status: {result['status']}")
        else:
            print(f"  {model_name}: {result.get('error', 'Not found')}")

    # Database stats
    print("\nDatabase Run History:")
    conn = init_db(DB_PATH)
    try:
        stats = queries.get_embedding_stats(conn)

        if stats.get("embedding_runs"):
            print("  Embedding runs:")
            for model, run_stats in stats["embedding_runs"].items():
                print(f"    {model}: {run_stats['completed']} completed, "
                      f"{run_stats['total_embeddings'] or 0} total embeddings")

        if stats.get("upload_runs"):
            print("  Upload runs:")
            for model, run_stats in stats["upload_runs"].items():
                print(f"    {model}: {run_stats['completed']} completed, "
                      f"{run_stats['total_uploaded'] or 0} total uploaded")

        if stats.get("errors"):
            print("  Errors:")
            for run_type, count in stats["errors"].items():
                print(f"    {run_type}: {count}")
    finally:
        conn.close()

    print()
    return 0


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RiskRADAR Embedding Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  embed MODEL     Generate embeddings (minilm|mika|both)
  upload MODEL    Upload to Qdrant (minilm|mika|both)
  all             Embed + upload both models
  verify MODEL    Verify Qdrant collection (minilm|mika|both)
  stats           Show embedding statistics

Examples:
  python -m embeddings.cli embed both --limit 100
  python -m embeddings.cli upload minilm
  python -m embeddings.cli all
  python -m embeddings.cli stats
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Embed command
    embed_parser = subparsers.add_parser("embed", help="Generate embeddings")
    embed_parser.add_argument(
        "model",
        choices=["minilm", "mika", "both"],
        help="Model to use"
    )
    embed_parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit chunks to process (for testing)"
    )
    embed_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging"
    )
    embed_parser.set_defaults(func=cmd_embed)

    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload to Qdrant")
    upload_parser.add_argument(
        "model",
        choices=["minilm", "mika", "both"],
        help="Model to upload"
    )
    upload_parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate collection"
    )
    upload_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging"
    )
    upload_parser.set_defaults(func=cmd_upload)

    # All command
    all_parser = subparsers.add_parser("all", help="Embed + upload both models")
    all_parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit chunks to process"
    )
    all_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging"
    )
    all_parser.set_defaults(func=cmd_all)

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify Qdrant collection")
    verify_parser.add_argument(
        "model",
        choices=["minilm", "mika", "both"],
        help="Model to verify"
    )
    verify_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging"
    )
    verify_parser.set_defaults(func=cmd_verify)

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show statistics")
    stats_parser.set_defaults(func=cmd_stats)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
