"""
extraction/scripts/verify_ocr.py
---------------------------------
Verify that all OCR dependencies are correctly installed.

Checks:
- pymupdf (fitz)
- pytesseract
- Tesseract OCR executable
- pdf2image
- Poppler
- pandas
"""

import sys


def check_pymupdf():
    """Check if pymupdf is installed."""
    try:
        import fitz
        version = fitz.version
        print(f"[OK] pymupdf (fitz) installed: {version}")
        return True
    except ImportError as e:
        print(f"[FAIL] pymupdf not found: {e}")
        print("  Install: pip install pymupdf")
        return False


def check_tesseract():
    """Check if pytesseract and Tesseract OCR are installed."""
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print(f"[OK] Tesseract OCR found: {version}")
        return True
    except ImportError:
        print("[FAIL] pytesseract not found")
        print("  Install: pip install pytesseract")
        return False
    except pytesseract.TesseractNotFoundError:
        print("[FAIL] Tesseract OCR executable not found")
        print("  Windows: choco install tesseract")
        print("  Or download: https://github.com/tesseract-ocr/tesseract")
        print("  Or run: scripts/install_tesseract.ps1")
        return False


def check_pdf2image():
    """Check if pdf2image and poppler are installed."""
    import shutil
    import subprocess

    try:
        from pdf2image import convert_from_path  # noqa: F401

        print("[OK] pdf2image installed")

        # Check if pdfinfo executable is available (Poppler)
        pdfinfo_path = shutil.which("pdfinfo")
        if pdfinfo_path:
            try:
                result = subprocess.run(
                    [pdfinfo_path, "-v"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                # pdfinfo -v outputs to stderr
                version_line = result.stderr.strip().split("\n")[0]
                print(f"[OK] Poppler found: {version_line}")
                return True
            except Exception as e:
                print(f"[WARN] Poppler found but error running: {e}")
                return False
        else:
            print("[WARN] Poppler not found in PATH")
            print("  Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases")
            print("  Extract and add bin/ to PATH")
            return False

    except ImportError as e:
        print(f"[FAIL] pdf2image not found: {e}")
        print("  Install: pip install pdf2image")
        return False


def check_pillow():
    """Check if Pillow is installed."""
    try:
        from PIL import Image
        import PIL
        print(f"[OK] Pillow installed: {PIL.__version__}")
        return True
    except ImportError as e:
        print(f"[FAIL] Pillow not found: {e}")
        print("  Install: pip install Pillow")
        return False


def check_pandas():
    """Check if pandas is installed."""
    try:
        import pandas as pd
        print(f"[OK] pandas installed: {pd.__version__}")
        return True
    except ImportError as e:
        print(f"[FAIL] pandas not found: {e}")
        print("  Install: pip install pandas")
        return False


def main():
    """Run all dependency checks."""
    print("=" * 60)
    print("RiskRADAR Phase 3 - OCR Dependency Verification")
    print("=" * 60)
    print()

    checks = [
        check_pymupdf(),
        check_tesseract(),
        check_pdf2image(),
        check_pillow(),
        check_pandas(),
    ]

    print()
    print("=" * 60)

    if all(checks):
        print("[OK] All OCR dependencies installed correctly!")
        print()
        print("You can now run:")
        print("  riskradar extract initial --limit 1  # Test extraction on 1 report")
        print("=" * 60)
        return 0
    else:
        print("[FAIL] Some dependencies are missing - see errors above")
        print()
        print("To install all Python dependencies:")
        print("  pip install -r requirements.txt")
        print()
        print("To install Tesseract (Windows):")
        print("  powershell -ExecutionPolicy Bypass -File extraction/scripts/install_tesseract.ps1")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
