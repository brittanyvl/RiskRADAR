"""
extraction/processing/chunk.py
------------------------------
Pass 2: Chunk documents into search-ready segments.

Creates overlapping, section-aware chunks for vector search:
1. Load documents.jsonl
2. Detect section headers
3. Split into sentences
4. Accumulate sentences until ~600 tokens
5. Create chunk when approaching 700 token limit
6. Start next chunk with 20% overlap from previous
7. Append relevant footnotes
8. Write all chunks to chunks.jsonl

CLI Entry Point for full pipeline:
  py -m extraction.processing.chunk pages     # Pass 0
  py -m extraction.processing.chunk documents # Pass 1
  py -m extraction.processing.chunk chunks    # Pass 2
  py -m extraction.processing.chunk all       # All passes
  py -m extraction.processing.chunk stats     # Show statistics
"""

import argparse
import json
import logging
import re
import statistics
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from riskradar.config import DB_PATH
from sqlite.connection import init_db
from sqlite import queries

from .consolidate import load_documents_jsonl, get_consolidation_stats
from .consolidate_pages import consolidate_pages, get_consolidation_stats as get_pages_stats
from .consolidate import consolidate_all
from .footnote_parse import Footnote, append_footnotes_to_chunk
from .section_detect import detect_sections
from .tokenizer import (
    count_tokens,
    get_config as get_tokenizer_config,
    get_overlap_tokens,
    is_in_target_range,
    is_over_target,
    TOKENIZER_CONFIG,
)


logger = logging.getLogger(__name__)


# Pipeline version
# v2: Hierarchical sections, 400 token minimum, 25% overlap, section prefix
PIPELINE_VERSION = "5.0.0"

# Sentence splitting pattern
# v2: Protect section numbers like "1." or "1.1" from being split
# Old pattern: r"(?<=[.!?])\s+(?=[A-Z])" - would split "1. The Accident"
# New pattern: requires letter OR 2+ digits before punctuation
# This preserves "1. The Accident" while splitting "Boeing 737. The aircraft"
# Note: Python lookbehinds need fixed width, so we use specific digit counts
SENTENCE_REGEX = re.compile(
    r"(?<=[a-zA-Z][.!?])\s+(?=[A-Z])"   # Letter + punct + space + uppercase
    r"|(?<=\d\d[.!?])\s+(?=[A-Z])"      # 2 digits + punct (e.g., "17.")
    r"|(?<=\d\d\d[.!?])\s+(?=[A-Z])"    # 3 digits + punct (e.g., "737.")
    r"|(?<=\d\d\d\d[.!?])\s+(?=[A-Z])"  # 4 digits + punct (e.g., "2000.")
)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    if not text:
        return []
    sentences = SENTENCE_REGEX.split(text)
    return [s.strip() for s in sentences if s.strip()]


def get_page_range(
    start: int,
    end: int,
    page_boundaries: list[dict]
) -> tuple[int, int, list[int]]:
    """Get page range for a text span."""
    pages = []
    for boundary in page_boundaries:
        if boundary["end"] > start and boundary["start"] < end:
            pages.append(boundary["page"])
    if not pages:
        return 0, 0, []
    return min(pages), max(pages), sorted(pages)


def chunk_document(
    document: dict,
    footnote_map: dict[str, Footnote]
) -> list[dict]:
    """
    Chunk a document into search-ready segments.

    v2 improvements:
    - Forward borrowing: chunks below min_tokens borrow from next section
    - Cross-section overlap: overlap carries across section boundaries
    - 400 token minimum enforced
    - 25% overlap (up from 20%)
    """
    report_id = document["report_id"]
    full_text = document["full_text"]
    page_boundaries = document.get("page_boundaries", [])
    page_sources = document.get("page_sources", [])

    if not full_text:
        return []

    # Detect sections
    sections, detection_method = detect_sections(full_text)

    # Build page source lookup
    page_source_map = {ps["page"]: ps for ps in page_sources}

    # Chunking parameters from config
    min_tokens = TOKENIZER_CONFIG["chunk_min_tokens"]  # v2: enforced minimum (400)
    target_tokens = TOKENIZER_CONFIG["chunk_target_tokens"]  # target (600)
    max_tokens = TOKENIZER_CONFIG["chunk_max_tokens"]  # hard max (800)
    overlap_tokens = get_overlap_tokens()

    # Build unified sentence list with section metadata
    # This enables forward borrowing across section boundaries
    all_sentences = []
    for section in sections:
        section_text = full_text[section.start:section.end]
        sentences = split_sentences(section_text)
        for sentence in sentences:
            # Find actual position in full_text
            sentence_start = full_text.find(sentence, section.start)
            all_sentences.append({
                "text": sentence,
                "tokens": count_tokens(sentence),
                "section_name": section.name,
                "section_number": section.number,
                "char_start": sentence_start,
            })

    if not all_sentences:
        return []

    chunks = []
    chunk_sequence = 0
    i = 0  # Current sentence index

    overlap_text = ""
    overlap_token_count = 0

    while i < len(all_sentences):
        current_sentences = []
        current_tokens = 0
        chunk_sections = set()  # Track sections this chunk spans
        first_section_name = None
        first_section_number = None

        # Accumulate sentences until we hit target/max or run out
        while i < len(all_sentences):
            sentence_info = all_sentences[i]
            sentence_tokens = sentence_info["tokens"]

            # Chunking decision logic:
            # 1. If adding would exceed max_tokens, must break (unless chunk is empty)
            # 2. If we've hit minimum AND adding would exceed target, prefer to break
            # 3. If under minimum, keep adding until we hit max_tokens

            if current_sentences:  # Only consider breaking if we have content
                would_exceed_max = current_tokens + sentence_tokens > max_tokens
                would_exceed_target = current_tokens + sentence_tokens > target_tokens
                met_minimum = current_tokens >= min_tokens

                if would_exceed_max:
                    break  # Hard stop at max
                if met_minimum and would_exceed_target:
                    break  # Soft stop at target once minimum is met

            current_sentences.append(sentence_info)
            current_tokens += sentence_tokens
            chunk_sections.add(sentence_info["section_name"])

            if first_section_name is None:
                first_section_name = sentence_info["section_name"]
                first_section_number = sentence_info["section_number"]

            i += 1

        # No more sentences accumulated - done
        if not current_sentences:
            break

        # Forward borrowing: if we're below minimum and there are more sentences,
        # continue accumulating (handled in the loop above)
        # If we're still below minimum at end of document, that's acceptable

        # Build chunk text
        chunk_text = " ".join(s["text"] for s in current_sentences)
        if overlap_text:
            chunk_text = overlap_text + " " + chunk_text

        # v2: Add section prefix for embedding context
        # Format: [SECTION_NAME] content...
        # This helps embedding models understand the structural context
        if first_section_name:
            section_prefix = f"[{first_section_name}] "
            chunk_text = section_prefix + chunk_text

        # Calculate positions
        chunk_start = current_sentences[0]["char_start"]
        chunk_end = current_sentences[-1]["char_start"] + len(current_sentences[-1]["text"])
        page_start, page_end, page_list = get_page_range(chunk_start, chunk_end, page_boundaries)

        # Determine source
        sources = set()
        for p in page_list:
            if p in page_source_map:
                sources.add(page_source_map[p].get("source", "unknown"))
        text_source = "mixed" if len(sources) > 1 else (
            sources.pop() if sources else document.get("primary_source", "unknown")
        )

        # Source quality
        source_quality = {}
        if page_list:
            alpha_ratios = [
                page_source_map[p].get("alphabetic_ratio")
                for p in page_list
                if p in page_source_map and page_source_map[p].get("alphabetic_ratio")
            ]
            if alpha_ratios:
                source_quality["min_alphabetic_ratio"] = min(alpha_ratios)
            ocr_confs = [
                page_source_map[p].get("ocr_confidence")
                for p in page_list
                if p in page_source_map and page_source_map[p].get("ocr_confidence")
            ]
            if ocr_confs:
                source_quality["avg_ocr_confidence"] = sum(ocr_confs) / len(ocr_confs)

        # Append footnotes
        final_text, appended_footnotes = append_footnotes_to_chunk(chunk_text, footnote_map)

        # Determine if chunk spans multiple sections
        spans_multiple = len(chunk_sections) > 1
        section_list = sorted(chunk_sections)

        chunk = {
            "chunk_id": f"{report_id}_chunk_{chunk_sequence:04d}",
            "report_id": report_id,
            "chunk_sequence": chunk_sequence,
            "page_start": page_start,
            "page_end": page_end,
            "page_list": page_list,
            "char_start": chunk_start,
            "char_end": chunk_end,
            "section_name": first_section_name,  # Primary section
            "section_number": first_section_number,
            "section_detection_method": detection_method,
            "chunk_text": final_text,
            "token_count": count_tokens(final_text),
            "overlap_tokens": overlap_token_count,
            "text_source": text_source,
            "page_sources": [page_source_map.get(p, {"page": p}) for p in page_list],
            "source_quality": source_quality,
            "has_footnotes": len(appended_footnotes) > 0,
            "footnotes": [
                {"marker": f.marker, "text": f.text} for f in appended_footnotes
            ] if appended_footnotes else None,
            "quality_flags": ["spans_multiple_sections"] if spans_multiple else [],
            "sections_spanned": section_list if spans_multiple else None,  # v2: track multi-section chunks
            "pipeline_version": PIPELINE_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        chunks.append(chunk)
        chunk_sequence += 1

        # Prepare overlap for next chunk (v2: carries across sections)
        overlap_sentences = []
        overlap_count = 0
        for s in reversed(current_sentences):
            s_tokens = s["tokens"]
            if overlap_count + s_tokens <= overlap_tokens:
                overlap_sentences.insert(0, s["text"])
                overlap_count += s_tokens
            else:
                break
        overlap_text = " ".join(overlap_sentences)
        overlap_token_count = overlap_count

    return chunks


def chunk_all(
    documents_path: Path,
    output_path: Path,
    limit: int | None = None,
    conn=None
) -> dict:
    """
    Chunk all documents into a single chunks.jsonl file.

    Args:
        documents_path: Path to documents.jsonl
        output_path: Path to output chunks.jsonl
        limit: Optional limit on number of reports
        conn: Optional database connection for run tracking

    Returns:
        Dict with chunking statistics
    """
    close_conn = False
    if conn is None:
        conn = init_db(DB_PATH)
        close_conn = True

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load all documents
    logger.info(f"Loading documents from {documents_path}")
    documents = load_documents_jsonl(documents_path)
    report_ids = list(documents.keys())

    if limit:
        report_ids = report_ids[:limit]

    logger.info(f"Found {len(report_ids)} documents to chunk")

    # Create run record
    run_id = None
    try:
        run_id = queries.create_chunking_run(
            conn,
            run_type="chunks",
            config_json=json.dumps({
                "tokenizer": get_tokenizer_config(),
                "pipeline_version": PIPELINE_VERSION,
            })
        )
    except Exception as e:
        logger.warning(f"Failed to create run record: {e}")

    # Statistics
    stats = {
        "total_documents": len(report_ids),
        "processed": 0,
        "failed": 0,
        "total_chunks": 0,
        "total_tokens": 0,
        "chunks_in_range": 0,
        "chunks_over": 0,
        "chunks_under": 0,
        "section_pattern_match": 0,
        "section_paragraph_fallback": 0,
        "section_no_structure": 0,
        "chunks_per_doc": [],
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        # Write all chunks to single file
        with open(output_path, "w", encoding="utf-8") as f:
            for i, report_id in enumerate(report_ids):
                try:
                    document = documents[report_id]

                    # Build footnote map
                    footnotes = document.get("footnotes", [])
                    footnote_map = {}
                    for fn in footnotes:
                        footnote_map[fn["marker"]] = Footnote(
                            marker=fn["marker"],
                            text=fn["text"],
                            page=fn.get("page")
                        )

                    # Chunk document
                    chunks = chunk_document(document, footnote_map)

                    # Write chunks
                    for chunk in chunks:
                        f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

                    # Update stats
                    stats["processed"] += 1
                    stats["total_chunks"] += len(chunks)
                    stats["chunks_per_doc"].append(len(chunks))

                    for chunk in chunks:
                        token_count = chunk["token_count"]
                        stats["total_tokens"] += token_count

                        if is_in_target_range(token_count):
                            stats["chunks_in_range"] += 1
                        elif is_over_target(token_count):
                            stats["chunks_over"] += 1
                        else:
                            stats["chunks_under"] += 1

                        method = chunk.get("section_detection_method", "")
                        if method == "pattern_match":
                            stats["section_pattern_match"] += 1
                        elif method == "paragraph_fallback":
                            stats["section_paragraph_fallback"] += 1
                        else:
                            stats["section_no_structure"] += 1

                    if (i + 1) % 50 == 0:
                        logger.info(f"Processed {i + 1}/{len(report_ids)} documents, {stats['total_chunks']} chunks")

                except Exception as e:
                    logger.error(f"Failed to chunk {report_id}: {e}")
                    stats["failed"] += 1

                    if run_id:
                        try:
                            queries.log_chunking_error(
                                conn,
                                run_id=run_id,
                                report_id=report_id,
                                error_type="chunking_error",
                                error_message=str(e),
                                stack_trace=traceback.format_exc(),
                            )
                        except Exception:
                            pass

        stats["completed_at"] = datetime.now(timezone.utc).isoformat()

        # Update run record
        if run_id:
            try:
                queries.update_chunking_run(
                    conn,
                    run_id=run_id,
                    status="completed",
                    documents_created=stats["processed"],
                    chunks_created=stats["total_chunks"],
                    error_count=stats["failed"],
                )
            except Exception as e:
                logger.warning(f"Failed to update run record: {e}")

        logger.info(f"Chunking complete: {stats['processed']} documents, {stats['total_chunks']} chunks")

    finally:
        if close_conn:
            conn.close()

    return stats


def run_full_pipeline(
    extraction_base: Path,
    limit: int | None = None
) -> dict:
    """Run the complete chunking pipeline (all passes)."""
    json_data = extraction_base / "json_data"
    pages_path = json_data / "pages.jsonl"
    documents_path = json_data / "documents.jsonl"
    chunks_path = json_data / "chunks.jsonl"

    all_stats = {}

    # Pass 0: Consolidate pages
    logger.info("=" * 60)
    logger.info("Pass 0: Consolidating pages to pages.jsonl")
    logger.info("=" * 60)
    all_stats["pages"] = consolidate_pages(
        output_path=pages_path,
        extraction_base=extraction_base,
        limit=limit
    )

    # Pass 1: Consolidate to documents
    logger.info("=" * 60)
    logger.info("Pass 1: Consolidating pages to documents.jsonl")
    logger.info("=" * 60)
    all_stats["documents"] = consolidate_all(
        pages_path=pages_path,
        output_path=documents_path,
        limit=limit
    )

    # Pass 2: Chunk documents
    logger.info("=" * 60)
    logger.info("Pass 2: Chunking documents to chunks.jsonl")
    logger.info("=" * 60)
    all_stats["chunks"] = chunk_all(
        documents_path=documents_path,
        output_path=chunks_path,
        limit=limit
    )

    return all_stats


def get_chunks_stats(chunks_path: Path) -> dict:
    """Get statistics from chunks.jsonl."""
    if not chunks_path.exists():
        return {"exists": False}

    stats = {
        "exists": True,
        "total_chunks": 0,
        "total_tokens": 0,
        "unique_reports": set(),
        "token_counts": [],
        "by_source": Counter(),
        "by_section_method": Counter(),
        "chunks_with_footnotes": 0,
    }

    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    chunk = json.loads(line)
                    stats["total_chunks"] += 1
                    stats["total_tokens"] += chunk.get("token_count", 0)
                    stats["unique_reports"].add(chunk.get("report_id"))
                    stats["token_counts"].append(chunk.get("token_count", 0))
                    stats["by_source"][chunk.get("text_source", "unknown")] += 1
                    stats["by_section_method"][chunk.get("section_detection_method", "unknown")] += 1
                    if chunk.get("has_footnotes"):
                        stats["chunks_with_footnotes"] += 1
                except json.JSONDecodeError:
                    pass

    stats["unique_reports"] = len(stats["unique_reports"])
    return stats


def log_analytics(stats: dict, chunks_path: Path):
    """Log comprehensive analytics report."""
    logger.info("=" * 70)
    logger.info("CHUNK ANALYTICS REPORT")
    logger.info("=" * 70)

    chunk_stats = get_chunks_stats(chunks_path)
    if not chunk_stats.get("exists"):
        logger.warning("No chunks.jsonl found for analytics")
        return

    token_counts = chunk_stats["token_counts"]

    logger.info(f"Total chunks: {chunk_stats['total_chunks']:,}")
    logger.info(f"Total documents: {chunk_stats['unique_reports']}")
    logger.info(f"Total tokens: {chunk_stats['total_tokens']:,}")

    if token_counts:
        logger.info(f"Avg tokens/chunk: {statistics.mean(token_counts):.1f}")
        logger.info(f"Median tokens/chunk: {statistics.median(token_counts):.1f}")
        logger.info(f"Min tokens: {min(token_counts)}")
        logger.info(f"Max tokens: {max(token_counts)}")

        # v2: Use config values for thresholds
        min_t = TOKENIZER_CONFIG["chunk_min_tokens"]  # 400
        max_t = TOKENIZER_CONFIG["chunk_max_tokens"]  # 800
        in_range = sum(1 for t in token_counts if min_t <= t <= max_t)
        under = sum(1 for t in token_counts if t < min_t)
        over = sum(1 for t in token_counts if t > max_t)
        total = len(token_counts)

        logger.info(f"Token distribution:")
        logger.info(f"  Under {min_t}: {under:,} ({100*under/total:.1f}%)")
        logger.info(f"  In range ({min_t}-{max_t}): {in_range:,} ({100*in_range/total:.1f}%)")
        logger.info(f"  Over {max_t}: {over:,} ({100*over/total:.1f}%)")

    logger.info(f"Text source distribution: {dict(chunk_stats['by_source'])}")
    logger.info(f"Section detection: {dict(chunk_stats['by_section_method'])}")
    logger.info(f"Chunks with footnotes: {chunk_stats['chunks_with_footnotes']}")

    logger.info(f"Qdrant planning:")
    logger.info(f"  Single model: {chunk_stats['total_chunks']:,} vectors")
    logger.info(f"  Two models: {chunk_stats['total_chunks'] * 2:,} vectors")
    logger.info(f"  Free tier headroom: {1_000_000 - chunk_stats['total_chunks'] * 2:,}")


def print_stats(extraction_base: Path):
    """Print pipeline statistics."""
    json_data = extraction_base / "json_data"
    pages_path = json_data / "pages.jsonl"
    documents_path = json_data / "documents.jsonl"
    chunks_path = json_data / "chunks.jsonl"

    print("\n" + "=" * 60)
    print("Chunking Pipeline Statistics")
    print("=" * 60)

    # Pages stats
    print("\nPass 0: pages.jsonl")
    pages_stats = get_pages_stats(pages_path)
    if pages_stats.get("exists"):
        print(f"  Total lines: {pages_stats['total_lines']:,}")
        print(f"  Unique reports: {pages_stats['unique_reports']}")
        print(f"  Embedded pages: {pages_stats['embedded_count']:,}")
        print(f"  OCR pages: {pages_stats['ocr_count']:,}")
        print(f"  File size: {pages_stats['file_size_bytes'] / 1024 / 1024:.1f} MB")
    else:
        print("  Not found - run 'pages' command first")

    # Documents stats
    print("\nPass 1: documents.jsonl")
    docs_stats = get_consolidation_stats(documents_path)
    if docs_stats.get("exists"):
        print(f"  Document count: {docs_stats['document_count']}")
        print(f"  Total tokens: {docs_stats['total_tokens']:,}")
        print(f"  Total pages: {docs_stats['total_pages']:,}")
        print(f"  By source: {docs_stats['by_source']}")
    else:
        print("  Not found - run 'documents' command first")

    # Chunks stats
    print("\nPass 2: chunks.jsonl")
    chunk_stats = get_chunks_stats(chunks_path)
    if chunk_stats.get("exists"):
        token_counts = chunk_stats["token_counts"]
        # v2: Use config values
        min_t = TOKENIZER_CONFIG["chunk_min_tokens"]  # 400
        max_t = TOKENIZER_CONFIG["chunk_max_tokens"]  # 800
        in_range = sum(1 for t in token_counts if min_t <= t <= max_t)
        over = sum(1 for t in token_counts if t > max_t)
        under = sum(1 for t in token_counts if t < min_t)
        total = len(token_counts)

        print(f"  Total chunks: {chunk_stats['total_chunks']:,}")
        print(f"  Total tokens: {chunk_stats['total_tokens']:,}")
        print(f"  Avg tokens/chunk: {statistics.mean(token_counts):.0f}" if token_counts else "")
        print(f"  In range ({min_t}-{max_t}): {in_range:,} ({100*in_range/total:.1f}%)" if total else "")
        print(f"  Over {max_t}: {over:,} ({100*over/total:.1f}%)" if total else "")
        print(f"  Under {min_t}: {under:,} ({100*under/total:.1f}%)" if total else "")
        print(f"\n  Qdrant Planning:")
        print(f"    Current chunks: {chunk_stats['total_chunks']:,}")
        print(f"    If both models: {chunk_stats['total_chunks'] * 2:,}")
        print(f"    Free tier limit: 1,000,000")
        print(f"    Headroom: {1_000_000 - chunk_stats['total_chunks'] * 2:,}")
    else:
        print("  Not found - run 'chunks' command first")

    print()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Chunking pipeline for RiskRADAR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  pages      Pass 0: Consolidate JSON files to pages.jsonl
  documents  Pass 1: Consolidate pages to documents.jsonl
  chunks     Pass 2: Chunk documents to chunks.jsonl
  all        Run full pipeline (all three passes)
  stats      Show pipeline statistics
        """
    )
    parser.add_argument(
        "command",
        choices=["pages", "documents", "chunks", "all", "stats"],
        help="Pipeline command to run"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of reports to process"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging"
    )

    args = parser.parse_args()

    # Determine paths
    extraction_base = Path(__file__).parent.parent
    project_root = extraction_base.parent
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Setup logging - both console and file
    log_level = logging.DEBUG if args.verbose else logging.INFO
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_file = logs_dir / f"chunk_{args.command}_{timestamp}.log"

    # Configure root logger with both handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    ))
    root_logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    root_logger.addHandler(file_handler)

    logger.info(f"Logging to {log_file}")

    json_data = extraction_base / "json_data"
    pages_path = json_data / "pages.jsonl"
    documents_path = json_data / "documents.jsonl"
    chunks_path = json_data / "chunks.jsonl"

    if args.command == "pages":
        stats = consolidate_pages(
            output_path=pages_path,
            extraction_base=extraction_base,
            limit=args.limit
        )
        logger.info(f"Wrote {stats['written_pages']} pages to {pages_path}")

    elif args.command == "documents":
        if not pages_path.exists():
            logger.error(f"{pages_path} not found. Run 'pages' first.")
            return 1
        stats = consolidate_all(
            pages_path=pages_path,
            output_path=documents_path,
            limit=args.limit
        )
        logger.info(f"Processed {stats['processed']} documents to {documents_path}")

    elif args.command == "chunks":
        if not documents_path.exists():
            logger.error(f"{documents_path} not found. Run 'documents' first.")
            return 1
        stats = chunk_all(
            documents_path=documents_path,
            output_path=chunks_path,
            limit=args.limit
        )
        logger.info(f"Created {stats['total_chunks']} chunks in {chunks_path}")
        log_analytics(stats, chunks_path)

    elif args.command == "all":
        stats = run_full_pipeline(
            extraction_base=extraction_base,
            limit=args.limit
        )
        logger.info("Full pipeline complete!")
        logger.info(f"  Pages written: {stats['pages']['written_pages']}")
        logger.info(f"  Documents processed: {stats['documents']['processed']}")
        logger.info(f"  Chunks created: {stats['chunks']['total_chunks']}")
        log_analytics(stats, chunks_path)

    elif args.command == "stats":
        print_stats(extraction_base)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
