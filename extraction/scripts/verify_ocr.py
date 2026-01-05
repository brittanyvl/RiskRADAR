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
        print(f"✓ pymupdf (fitz) installed: {version}")
        return True
    except ImportError as e:
        print(f"✗ pymupdf not found: {e}")
        print("  Install: pip install pymupdf")
        return False


def check_tesseract():
    """Check if pytesseract and Tesseract OCR are installed."""
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print(f"✓ Tesseract OCR found: {version}")
        return True
    except ImportError:
        print("✗ pytesseract not found")
        print("  Install: pip install pytesseract")
        return False
    except pytesseract.TesseractNotFoundError:
        print("✗ Tesseract OCR executable not found")
        print("  Windows: choco install tesseract")
        print("  Or download: https://github.com/tesseract-ocr/tesseract")
        print("  Or run: scripts/install_tesseract.ps1")
        return False


def check_pdf2image():
    """Check if pdf2image and poppler are installed."""
    try:
        from pdf2image import convert_from_path
        from pdf2image.exceptions import PDFInfoNotInstalledError

        # Try to check if poppler is available
        # This will fail if poppler is not in PATH
        try:
            import pdf2image.pdfinfo_from_path
            print("✓ pdf2image installed")
            print("✓ Poppler found (pdf2image backend)")
            return True
        except (PDFInfoNotInstalledError, Exception):
            print("✓ pdf2image installed")
            print("⚠ Poppler may not be installed or not in PATH")
            print("  Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases")
            print("  Extract and add bin/ to PATH")
            return False

    except ImportError as e:
        print(f"✗ pdf2image not found: {e}")
        print("  Install: pip install pdf2image")
        return False


def check_pillow():
    """Check if Pillow is installed."""
    try:
        from PIL import Image
        import PIL
        print(f"✓ Pillow installed: {PIL.__version__}")
        return True
    except ImportError as e:
        print(f"✗ Pillow not found: {e}")
        print("  Install: pip install Pillow")
        return False


def check_pandas():
    """Check if pandas is installed."""
    try:
        import pandas as pd
        print(f"✓ pandas installed: {pd.__version__}")
        return True
    except ImportError as e:
        print(f"✗ pandas not found: {e}")
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
        print("✓ All OCR dependencies installed correctly!")
        print()
        print("You can now run:")
        print("  riskradar extract initial --limit 1  # Test extraction on 1 report")
        print("=" * 60)
        return 0
    else:
        print("✗ Some dependencies are missing - see errors above")
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
