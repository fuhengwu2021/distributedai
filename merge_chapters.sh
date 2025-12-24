#!/bin/bash

# Script to merge all chapter PDFs into a single book.pdf
# Usage: ./merge_chapters.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUTPUT_FILE="book.pdf"
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

echo "📚 Merging all chapter PDFs into $OUTPUT_FILE..."

# Check if required tools are available
if command -v pdftk &> /dev/null; then
    USE_PDFTK=true
elif command -v gs &> /dev/null; then
    USE_GS=true
elif python3 -c "import pypdf" 2>/dev/null; then
    USE_PYTHON=true
else
    echo "❌ Error: No PDF merging tool found."
    echo "   Please install one of: pdftk, ghostscript (gs), or pypdf (pip install pypdf)"
    exit 1
fi

# Function to check if PDF exists and is valid
check_pdf() {
    local pdf_file="$1"
    if [ ! -f "$pdf_file" ]; then
        echo "⚠️  Warning: $pdf_file not found, skipping..."
        return 1
    fi
    if [ ! -s "$pdf_file" ]; then
        echo "⚠️  Warning: $pdf_file is empty, skipping..."
        return 1
    fi
    return 0
}

# Collect all chapter PDFs in order
PDF_FILES=()

# Chapters 1-11
for i in {1..11}; do
    # Find chapter PDF file (look for chapterN.pdf in chapterN-* directories)
    PDF_PATH=$(find . -path "*/chapter${i}-*/chapter${i}.pdf" -type f 2>/dev/null | head -1)
    
    if [ -n "$PDF_PATH" ] && check_pdf "$PDF_PATH"; then
        PDF_FILES+=("$PDF_PATH")
        echo "  ✓ Found: $PDF_PATH"
    else
        echo "  ✗ Chapter $i PDF not found"
    fi
done

# Chapter 12 (if exists)
PDF_PATH=$(find . -path "*/chapter12-*/chapter12.pdf" -type f 2>/dev/null | head -1)
if [ -n "$PDF_PATH" ] && check_pdf "$PDF_PATH"; then
    PDF_FILES+=("$PDF_PATH")
    echo "  ✓ Found: $PDF_PATH"
fi

# Chapter X
if check_pdf "chapterx/chapterx.pdf"; then
    PDF_FILES+=("chapterx/chapterx.pdf")
    echo "  ✓ Found: chapterx/chapterx.pdf"
fi

if [ ${#PDF_FILES[@]} -eq 0 ]; then
    echo "❌ Error: No chapter PDFs found!"
    exit 1
fi

echo ""
echo "📄 Found ${#PDF_FILES[@]} chapter PDFs to merge:"
for pdf in "${PDF_FILES[@]}"; do
    echo "   - $pdf"
done
echo ""

# Merge PDFs based on available tool
if [ "$USE_PDFTK" = true ]; then
    echo "🔧 Using pdftk to merge PDFs..."
    pdftk "${PDF_FILES[@]}" cat output "$OUTPUT_FILE"
    
elif [ "$USE_GS" = true ]; then
    echo "🔧 Using Ghostscript to merge PDFs..."
    gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -sOutputFile="$OUTPUT_FILE" "${PDF_FILES[@]}"
    
elif [ "$USE_PYTHON" = true ]; then
    echo "🔧 Using Python pypdf to merge PDFs..."
    python3 << EOF
from pypdf import PdfWriter
import sys

writer = PdfWriter()

pdf_files = ${PDF_FILES[@]@Q}
for pdf_file in pdf_files:
    try:
        print(f"  Adding: {pdf_file}")
        writer.append(pdf_file)
    except Exception as e:
        print(f"  ⚠️  Warning: Failed to add {pdf_file}: {e}", file=sys.stderr)

print(f"\n💾 Writing merged PDF to: $OUTPUT_FILE")
with open("$OUTPUT_FILE", "wb") as output_file:
    writer.write(output_file)

print("✅ Successfully merged all PDFs!")
EOF
fi

if [ -f "$OUTPUT_FILE" ] && [ -s "$OUTPUT_FILE" ]; then
    FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
    echo ""
    echo "✅ Successfully created $OUTPUT_FILE ($FILE_SIZE)"
    echo "   Merged ${#PDF_FILES[@]} chapter PDFs"
else
    echo "❌ Error: Failed to create $OUTPUT_FILE"
    exit 1
fi
