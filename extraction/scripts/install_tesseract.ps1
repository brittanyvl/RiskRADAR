# install_tesseract.ps1
# ----------------------
# Automated Tesseract OCR installation for Windows using Chocolatey.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File extraction/scripts/install_tesseract.ps1
#
# What it does:
# 1. Checks if Chocolatey is installed (Windows package manager)
# 2. Installs Chocolatey if not present
# 3. Installs Tesseract OCR via Chocolatey
# 4. Adds Tesseract to PATH for current session
#
# Requirements:
# - Windows PowerShell 5.1+ or PowerShell Core 7+
# - Administrator privileges (for Chocolatey installation)
# - Internet connection

Write-Host "=" * 60
Write-Host "RiskRADAR Phase 3 - Tesseract OCR Installation"
Write-Host "=" * 60
Write-Host ""

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "⚠ WARNING: Not running as Administrator" -ForegroundColor Yellow
    Write-Host "  Chocolatey installation requires admin privileges." -ForegroundColor Yellow
    Write-Host "  Please run PowerShell as Administrator and try again." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To run as admin:" -ForegroundColor Cyan
    Write-Host "  1. Right-click PowerShell" -ForegroundColor Cyan
    Write-Host "  2. Select 'Run as Administrator'" -ForegroundColor Cyan
    Write-Host "  3. Re-run this script" -ForegroundColor Cyan
    Write-Host ""
    exit 1
}

# Step 1: Check if Chocolatey is installed
Write-Host "Step 1: Checking for Chocolatey..." -ForegroundColor Cyan

if (Get-Command choco -ErrorAction SilentlyContinue) {
    $chocoVersion = choco --version
    Write-Host "✓ Chocolatey already installed: v$chocoVersion" -ForegroundColor Green
} else {
    Write-Host "✗ Chocolatey not found" -ForegroundColor Yellow
    Write-Host "  Installing Chocolatey..." -ForegroundColor Cyan

    try {
        # Set TLS 1.2 for security
        [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072

        # Download and execute Chocolatey installer
        Set-ExecutionPolicy Bypass -Scope Process -Force
        Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

        Write-Host "✓ Chocolatey installed successfully" -ForegroundColor Green

        # Refresh environment variables
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    } catch {
        Write-Host "✗ Failed to install Chocolatey: $_" -ForegroundColor Red
        Write-Host ""
        Write-Host "Manual installation:" -ForegroundColor Yellow
        Write-Host "  Visit: https://chocolatey.org/install" -ForegroundColor Yellow
        exit 1
    }
}

Write-Host ""

# Step 2: Check if Tesseract is already installed
Write-Host "Step 2: Checking for Tesseract OCR..." -ForegroundColor Cyan

$tesseractInstalled = $false
$tesseractPaths = @(
    "C:\Program Files\Tesseract-OCR\tesseract.exe",
    "C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
)

foreach ($path in $tesseractPaths) {
    if (Test-Path $path) {
        $tesseractInstalled = $true
        Write-Host "✓ Tesseract already installed at: $path" -ForegroundColor Green

        # Get version
        try {
            $version = & $path --version 2>&1 | Select-Object -First 1
            Write-Host "  Version: $version" -ForegroundColor Green
        } catch {
            Write-Host "  (Unable to determine version)" -ForegroundColor Yellow
        }
        break
    }
}

if (-not $tesseractInstalled) {
    Write-Host "✗ Tesseract not found" -ForegroundColor Yellow
    Write-Host "  Installing Tesseract via Chocolatey..." -ForegroundColor Cyan

    try {
        # Install Tesseract (includes English language data by default)
        choco install tesseract -y

        Write-Host "✓ Tesseract installed successfully" -ForegroundColor Green

        # Refresh environment variables
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

        # Verify installation
        $tesseractPath = "C:\Program Files\Tesseract-OCR\tesseract.exe"
        if (Test-Path $tesseractPath) {
            $version = & $tesseractPath --version 2>&1 | Select-Object -First 1
            Write-Host "  Installed version: $version" -ForegroundColor Green
        }
    } catch {
        Write-Host "✗ Failed to install Tesseract: $_" -ForegroundColor Red
        Write-Host ""
        Write-Host "Manual installation:" -ForegroundColor Yellow
        Write-Host "  1. Download from: https://github.com/UB-Mannheim/tesseract/wiki" -ForegroundColor Yellow
        Write-Host "  2. Run installer" -ForegroundColor Yellow
        Write-Host "  3. Add to PATH: C:\Program Files\Tesseract-OCR" -ForegroundColor Yellow
        exit 1
    }
}

Write-Host ""

# Step 3: Add Tesseract to PATH for current session
Write-Host "Step 3: Configuring PATH..." -ForegroundColor Cyan

$tesseractDir = "C:\Program Files\Tesseract-OCR"
if (Test-Path $tesseractDir) {
    if ($env:Path -notlike "*$tesseractDir*") {
        $env:Path += ";$tesseractDir"
        Write-Host "✓ Added Tesseract to PATH for current session" -ForegroundColor Green
    } else {
        Write-Host "✓ Tesseract already in PATH" -ForegroundColor Green
    }
} else {
    Write-Host "⚠ Tesseract directory not found at expected location" -ForegroundColor Yellow
    Write-Host "  You may need to manually add Tesseract to your PATH" -ForegroundColor Yellow
}

Write-Host ""

# Step 4: Verify installation
Write-Host "Step 4: Verifying installation..." -ForegroundColor Cyan

try {
    # Try to run tesseract command
    $null = tesseract --version 2>&1
    Write-Host "✓ Tesseract command is accessible" -ForegroundColor Green
} catch {
    Write-Host "⚠ Cannot execute tesseract command" -ForegroundColor Yellow
    Write-Host "  You may need to restart your terminal or computer" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=" * 60
Write-Host "Installation Complete!" -ForegroundColor Green
Write-Host "=" * 60
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Close and reopen your terminal (to refresh PATH)" -ForegroundColor White
Write-Host "  2. Run: python extraction/scripts/verify_ocr.py" -ForegroundColor White
Write-Host "  3. If all checks pass, run: riskradar extract initial --limit 1" -ForegroundColor White
Write-Host ""
