#!/bin/bash

# Script to convert markdown files to PDF using pandoc
# Usage:
#   ./convert_to_pdf.sh                    # Convert all chapter chapterX.md files
#   ./convert_to_pdf.sh 1                  # Convert chapter 1
#   ./convert_to_pdf.sh 2                  # Convert chapter 2
#   ./convert_to_pdf.sh chapter1           # Convert specific chapter (alternative)
#   ./convert_to_pdf.sh chapter1-introduction-to-modern-distributed-ai  # Full chapter name

set -e  # Exit on error

# Get the script directory (root of the project)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Function to convert a markdown file to PDF
convert_md_to_pdf() {
    local md_file="$1"
    local pdf_file="${md_file%.md}.pdf"
    local dir_name="$(dirname "$md_file")"
    local md_basename="$(basename "$md_file")"
    local pdf_basename="$(basename "$pdf_file")"
    
    if [ ! -f "$md_file" ]; then
        echo "❌ Error: File not found: $md_file"
        return 1
    fi
    
    # Convert to absolute paths
    local abs_md_file="$(cd "$(dirname "$md_file")" && pwd)/$md_basename"
    local abs_pdf_file="$(cd "$(dirname "$md_file")" && pwd)/$pdf_basename"
    
    echo "📄 Converting: $md_file"
    echo "   → $pdf_file"
    
    # Change to the chapter directory so relative image paths work correctly
    cd "$(dirname "$abs_md_file")"
    
    # Try different PDF engines in order of preference
    # Use basenames since we're now in the chapter directory
    # Capture output to filter warnings but show errors
    local pandoc_output=""
    # Create LaTeX header with image size control
    # Keep images at original size to prevent blurriness from over-scaling
    # Only scale down if image exceeds page width
    local latex_header=$(cat <<'EOF'
\usepackage{microtype}
\sloppy
\setlength{\emergencystretch}{3em}
\setlength{\tolerance}{1000}
\allowdisplaybreaks
\usepackage{float}
\floatplacement{figure}{H}
\usepackage{graphicx}
% Control image scaling: keep images at original size unless they exceed page width
% This prevents small images from being over-scaled and becoming blurry
\makeatletter
% Only scale down if image is larger than linewidth, otherwise keep original size
\def\maxwidth{\ifdim\Gin@nat@width>\linewidth\linewidth\else\Gin@nat@width\fi}
\def\maxheight{\ifdim\Gin@nat@height>\textheight\textheight\else\Gin@nat@height\fi}
\makeatother
\setkeys{Gin}{width=\maxwidth,height=\maxheight,keepaspectratio}
EOF
)
    if pandoc_output=$(pandoc "$md_basename" -o "$pdf_basename" --pdf-engine=xelatex -V geometry:margin=1in --highlight-style=tango -H <(echo "$latex_header") 2>&1); then
        # Filter out font-related warnings but keep image warnings
        echo "$pandoc_output" | grep -E "\[WARNING\].*image|\[WARNING\].*resource" || true
        echo "✅ Successfully converted using xelatex"
        cd "$SCRIPT_DIR"
        return 0
    elif pandoc_output=$(pandoc "$md_basename" -o "$pdf_basename" --pdf-engine=pdflatex -V geometry:margin=1in --highlight-style=tango -V 'tolerance=1000' -V 'emergencystretch=3em' -H <(echo "$latex_header") 2>&1); then
        echo "$pandoc_output" | grep -E "\[WARNING\].*image|\[WARNING\].*resource" || true
        echo "✅ Successfully converted using pdflatex"
        cd "$SCRIPT_DIR"
        return 0
    elif pandoc_output=$(pandoc "$md_basename" -o "$pdf_basename" -V geometry:margin=1in --highlight-style=tango -V 'tolerance=1000' -V 'emergencystretch=3em' -H <(echo "$latex_header") 2>&1); then
        echo "$pandoc_output" | grep -E "\[WARNING\].*image|\[WARNING\].*resource" || true
        echo "✅ Successfully converted using default engine"
        cd "$SCRIPT_DIR"
        return 0
    else
        # Show all output if conversion failed
        echo "$pandoc_output"
        echo "❌ Failed to convert: $md_file"
        cd "$SCRIPT_DIR"
        return 1
    fi
}

# If a chapter name is provided as argument
if [ $# -gt 0 ]; then
    CHAPTER_ARG="$1"
    
    # Check if argument is a simple number (e.g., "1", "2", "10")
    if [[ "$CHAPTER_ARG" =~ ^[0-9]+$ ]]; then
        # Find chapter directory starting with "chapter" + number
        FOUND_DIR=$(find . -maxdepth 1 -type d -name "chapter${CHAPTER_ARG}-*" | head -1)
        if [ -n "$FOUND_DIR" ] && [ -f "$FOUND_DIR/chapter${CHAPTER_ARG}.md" ]; then
            convert_md_to_pdf "$FOUND_DIR/chapter${CHAPTER_ARG}.md"
        else
            echo "❌ Error: Could not find chapter $CHAPTER_ARG"
            echo "   Available chapters:"
            find . -maxdepth 1 -type d -name "chapter*" | sed 's|^\./||' | sort
            exit 1
        fi
    # Check if it's a direct directory path
    elif [ -d "$CHAPTER_ARG" ]; then
        # Try to find chapterX.md file in the directory
        CHAPTER_NUM=$(echo "$CHAPTER_ARG" | grep -oP 'chapter\K\d+' || echo "")
        if [ -n "$CHAPTER_NUM" ] && [ -f "$CHAPTER_ARG/chapter${CHAPTER_NUM}.md" ]; then
            convert_md_to_pdf "$CHAPTER_ARG/chapter${CHAPTER_NUM}.md"
        elif [ -f "$CHAPTER_ARG/main.md" ]; then
            # Fallback to main.md for backward compatibility
            convert_md_to_pdf "$CHAPTER_ARG/main.md"
        else
            echo "❌ Error: Could not find chapter file in $CHAPTER_ARG"
            exit 1
        fi
    # Check if it starts with "chapter" followed by a number
    elif [[ "$CHAPTER_ARG" =~ ^chapter[0-9]+ ]]; then
        # Extract chapter number
        CHAPTER_NUM=$(echo "$CHAPTER_ARG" | grep -oP '\d+')
        # Find chapter directory matching the pattern
        FOUND_DIR=$(find . -maxdepth 1 -type d -name "${CHAPTER_ARG}*" | head -1)
        if [ -n "$FOUND_DIR" ] && [ -f "$FOUND_DIR/chapter${CHAPTER_NUM}.md" ]; then
            convert_md_to_pdf "$FOUND_DIR/chapter${CHAPTER_NUM}.md"
        else
            echo "❌ Error: Could not find chapter matching '$CHAPTER_ARG'"
            echo "   Available chapters:"
            find . -maxdepth 1 -type d -name "chapter*" | sed 's|^\./||' | sort
            exit 1
        fi
    else
        # Try to find by pattern match
        FOUND_DIR=$(find . -maxdepth 1 -type d -name "*${CHAPTER_ARG}*" | head -1)
        if [ -n "$FOUND_DIR" ]; then
            CHAPTER_NUM=$(echo "$FOUND_DIR" | grep -oP 'chapter\K\d+' || echo "")
            if [ -n "$CHAPTER_NUM" ] && [ -f "$FOUND_DIR/chapter${CHAPTER_NUM}.md" ]; then
                convert_md_to_pdf "$FOUND_DIR/chapter${CHAPTER_NUM}.md"
            elif [ -f "$FOUND_DIR/main.md" ]; then
                # Fallback to main.md for backward compatibility
                convert_md_to_pdf "$FOUND_DIR/main.md"
            else
                echo "❌ Error: Could not find chapter file in '$FOUND_DIR'"
                exit 1
            fi
        else
            echo "❌ Error: Could not find chapter matching '$CHAPTER_ARG'"
            echo "   Available chapters:"
            find . -maxdepth 1 -type d -name "chapter*" | sed 's|^\./||' | sort
            exit 1
        fi
    fi
else
    # No argument provided, convert all chapter chapterX.md files
    echo "🔄 Converting all chapter chapterX.md files to PDF..."
    echo ""
    
    SUCCESS=0
    FAILED=0
    
    # Find all chapter directories with chapterX.md files
    while IFS= read -r -d '' md_file; do
        if convert_md_to_pdf "$md_file"; then
            ((SUCCESS++))
        else
            ((FAILED++))
        fi
        echo ""
    done < <(find . -maxdepth 2 -name "chapter[0-9]*.md" -type f -print0 | sort -z)
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 Summary:"
    echo "   ✅ Successful: $SUCCESS"
    echo "   ❌ Failed: $FAILED"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
fi
