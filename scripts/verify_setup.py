"""
scripts/verify_setup.py
-----------------------
Verify RiskRADAR project setup and dependencies.

Checks:
- Python version >= 3.9
- Virtual environment is active
- Required packages can be imported
- NAS connectivity (if configured)
- Qdrant credentials (if configured)
- System dependencies (Tesseract for OCR)

Usage:
    python -m scripts.verify_setup
    python scripts/verify_setup.py
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def print_header(title: str):
    """Print a section header."""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(check: str, passed: bool, detail: str = ""):
    """Print a check result."""
    status = "[OK]" if passed else "[FAIL]"
    msg = f"  {status} {check}"
    if detail:
        msg += f" - {detail}"
    print(msg)
    return passed


def check_python_version() -> bool:
    """Check Python version >= 3.9."""
    version = sys.version_info
    passed = version >= (3, 9)
    detail = f"{version.major}.{version.minor}.{version.micro}"
    return print_result("Python version >= 3.9", passed, detail)


def check_venv_active() -> bool:
    """Check if running in a virtual environment."""
    in_venv = (
        hasattr(sys, "real_prefix") or
        (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
    )
    detail = sys.prefix if in_venv else "Not in venv!"
    return print_result("Virtual environment active", in_venv, detail)


def check_venv_python() -> bool:
    """Check that we're using the venv Python, not system 'py' launcher."""
    exe_path = Path(sys.executable)
    is_venv_python = "venv" in str(exe_path).lower()
    detail = str(exe_path)
    return print_result("Using venv Python (not 'py')", is_venv_python, detail)


def check_import(module_name: str, package_name: str = None) -> bool:
    """Check if a module can be imported."""
    display_name = package_name or module_name
    try:
        __import__(module_name)
        return print_result(f"Import {display_name}", True)
    except ImportError as e:
        return print_result(f"Import {display_name}", False, str(e))


def check_project_imports() -> list[bool]:
    """Check project-specific imports work."""
    results = []

    # Core dependencies
    results.append(check_import("dotenv", "python-dotenv"))
    results.append(check_import("selenium"))
    results.append(check_import("fitz", "pymupdf"))
    results.append(check_import("pytesseract"))
    results.append(check_import("pdf2image"))
    results.append(check_import("duckdb"))
    results.append(check_import("qdrant_client", "qdrant-client"))
    results.append(check_import("sentence_transformers", "sentence-transformers"))
    results.append(check_import("tiktoken"))
    results.append(check_import("pandas"))
    results.append(check_import("pyarrow"))

    # Project modules
    results.append(check_import("riskradar.config", "riskradar"))
    results.append(check_import("sqlite.connection", "sqlite module"))
    results.append(check_import("scraper.browser", "scraper"))
    results.append(check_import("extraction.processing.extract", "extraction"))
    results.append(check_import("embeddings.cli", "embeddings"))
    results.append(check_import("analytics.cli", "analytics"))
    results.append(check_import("eval.benchmark", "eval"))

    return results


def check_nas_connectivity() -> bool:
    """Check NAS path is accessible."""
    try:
        from riskradar.config import NAS_PATH
        exists = NAS_PATH.exists()
        if exists:
            # Count PDF files
            pdf_count = len(list(NAS_PATH.glob("*.pdf")))
            detail = f"{NAS_PATH} ({pdf_count} PDFs)"
        else:
            detail = f"{NAS_PATH} not accessible"
        return print_result("NAS connectivity", exists, detail)
    except Exception as e:
        return print_result("NAS connectivity", False, str(e))


def check_database() -> bool:
    """Check SQLite database exists and has data."""
    try:
        from riskradar.config import DB_PATH
        if not DB_PATH.exists():
            return print_result("SQLite database", False, f"{DB_PATH} not found")

        from sqlite.connection import init_db
        conn = init_db(DB_PATH)
        count = conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
        conn.close()
        return print_result("SQLite database", True, f"{count} reports in {DB_PATH.name}")
    except Exception as e:
        return print_result("SQLite database", False, str(e))


def check_qdrant_credentials() -> bool:
    """Check Qdrant credentials are configured."""
    url = os.getenv("QDRANT_URL")
    key = os.getenv("QDRANT_API_KEY")

    if url and key:
        # Mask the key
        masked = key[:4] + "..." + key[-4:] if len(key) > 8 else "***"
        return print_result("Qdrant credentials", True, f"URL set, key={masked}")
    else:
        missing = []
        if not url:
            missing.append("QDRANT_URL")
        if not key:
            missing.append("QDRANT_API_KEY")
        return print_result("Qdrant credentials", False, f"Missing: {', '.join(missing)}")


def check_tesseract() -> bool:
    """Check Tesseract OCR is installed."""
    tesseract_path = shutil.which("tesseract")
    if tesseract_path:
        # Get version
        try:
            result = subprocess.run(
                ["tesseract", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            version = result.stdout.split("\n")[0] if result.stdout else "unknown"
            return print_result("Tesseract OCR", True, version)
        except Exception:
            return print_result("Tesseract OCR", True, tesseract_path)
    else:
        return print_result(
            "Tesseract OCR",
            False,
            "Not found in PATH. Install from https://github.com/tesseract-ocr/tesseract"
        )


def check_poppler() -> bool:
    """Check Poppler (pdf2image dependency) is installed."""
    # pdf2image uses pdftoppm from poppler
    poppler_path = shutil.which("pdftoppm")
    if poppler_path:
        return print_result("Poppler (pdftoppm)", True, poppler_path)
    else:
        return print_result(
            "Poppler (pdftoppm)",
            False,
            "Not found. Required for OCR. Install poppler-utils."
        )


def check_parquet_files() -> bool:
    """Check if Parquet files exist for analytics."""
    try:
        project_root = Path(__file__).parent.parent
        parquet_dir = project_root / "analytics" / "data"

        expected = ["pages.parquet", "documents.parquet", "chunks.parquet"]
        found = [f for f in expected if (parquet_dir / f).exists()]

        if len(found) == len(expected):
            return print_result("Analytics Parquet files", True, f"{len(found)}/{len(expected)} files")
        elif found:
            return print_result(
                "Analytics Parquet files",
                False,
                f"Missing: {set(expected) - set(found)}. Run: python -m analytics.convert"
            )
        else:
            return print_result(
                "Analytics Parquet files",
                False,
                "Not found. Run: python -m analytics.convert"
            )
    except Exception as e:
        return print_result("Analytics Parquet files", False, str(e))


def main():
    """Run all verification checks."""
    print()
    print("RiskRADAR Setup Verification")
    print("=" * 60)

    all_results = []
    critical_failed = False

    # Python environment
    print_header("Python Environment")
    all_results.append(check_python_version())
    all_results.append(check_venv_active())
    all_results.append(check_venv_python())

    # If not in venv, warn strongly
    if not all_results[-1]:
        print()
        print("  WARNING: You may be using 'py' instead of 'python'.")
        print("  Make sure venv is activated: venv\\Scripts\\activate")
        critical_failed = True

    # Dependencies
    print_header("Python Dependencies")
    import_results = check_project_imports()
    all_results.extend(import_results)

    # System dependencies
    print_header("System Dependencies")
    all_results.append(check_tesseract())
    all_results.append(check_poppler())

    # Project resources
    print_header("Project Resources")
    all_results.append(check_database())
    all_results.append(check_nas_connectivity())
    all_results.append(check_parquet_files())

    # Credentials (optional)
    print_header("Credentials (Optional)")
    all_results.append(check_qdrant_credentials())

    # Summary
    print_header("Summary")
    passed = sum(all_results)
    total = len(all_results)
    failed = total - passed

    print(f"  Passed: {passed}/{total}")
    print(f"  Failed: {failed}/{total}")

    if failed == 0:
        print()
        print("  All checks passed! Project is ready to use.")
        return 0
    elif critical_failed:
        print()
        print("  CRITICAL: Fix environment issues before proceeding.")
        return 1
    else:
        print()
        print("  Some checks failed. Review the output above.")
        print("  Most issues can be fixed by installing missing dependencies.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
