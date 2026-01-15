#!/bin/bash

# Script to merge all chapter markdown files and generate a single PDF book with automatic TOC
# Usage: ./build_book_from_markdown.sh [--style circle|square]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUTPUT_FILE="book.pdf"
TEMP_DIR="/tmp/ai4math"
# Create temp directory if it doesn't exist
mkdir -p "$TEMP_DIR"
# Don't auto-cleanup temp dir on exit - keep it for debugging
# trap "rm -rf $TEMP_DIR" EXIT
echo "   Temp directory: $TEMP_DIR (preserved for debugging)"

# Parse chapter style argument (default: circle)
CHAPTER_STYLE="circle"
if [ "$1" = "--style" ] || [ "$1" = "-s" ]; then
    if [ "$2" = "square" ]; then
        CHAPTER_STYLE="square"
    elif [ "$2" = "circle" ]; then
        CHAPTER_STYLE="circle"
    fi
elif [ "$1" = "--style=square" ] || [ "$1" = "-s=square" ]; then
    CHAPTER_STYLE="square"
elif [ "$1" = "--style=circle" ] || [ "$1" = "-s=circle" ]; then
    CHAPTER_STYLE="circle"
fi

echo "📚 Merging all chapter markdown files into $OUTPUT_FILE..."
echo "   Chapter number style: $CHAPTER_STYLE"

# Function to find chapter markdown file
find_chapter_md() {
    local chapter_num="$1"
    if [ "$chapter_num" = "x" ]; then
        find . -path "*/chapterx/chapterx.md" -type f 2>/dev/null | head -1
    else
        find . -path "*/chapter${chapter_num}-*/chapter${chapter_num}.md" -type f 2>/dev/null | head -1
    fi
}

# Function to extract chapter number from file path
get_chapter_number() {
    local md_file="$1"
    local basename_file="$(basename "$md_file")"
    # Extract chapter number: chapter1.md -> 1, chapterx.md -> x
    if [[ "$basename_file" =~ chapter([0-9]+)\.md ]]; then
        echo "${BASH_REMATCH[1]}"
    elif [[ "$basename_file" =~ chapterx\.md ]]; then
        echo "x"
    else
        # Fallback: try to extract from directory name
        local dir_name="$(dirname "$md_file")"
        if [[ "$dir_name" =~ chapter([0-9]+)- ]]; then
            echo "${BASH_REMATCH[1]}"
        elif [[ "$dir_name" =~ /chapterx ]]; then
            echo "x"
        else
            echo "unknown"
        fi
    fi
}

# Collect all chapter markdown files in order
MD_FILES=()

# Check for test mode (only process first 2 chapters)
TEST_MODE=${TEST_MODE:-false}
MAX_CHAPTERS=${MAX_CHAPTERS:-21}

# Chapters 1-21 (or limited by MAX_CHAPTERS)
for i in $(seq 1 $MAX_CHAPTERS); do
    MD_PATH=$(find_chapter_md "$i")
    if [ -n "$MD_PATH" ] && [ -f "$MD_PATH" ]; then
        MD_FILES+=("$MD_PATH")
        echo "  ✓ Found: $MD_PATH"
    else
        echo "  ✗ Chapter $i markdown not found"
    fi
done

# Chapter X (Appendix) - include if INCLUDE_APPENDIX is set to "true" or if MAX_CHAPTERS >= 21
# Default: include appendix if MAX_CHAPTERS >= 21, but can be overridden with INCLUDE_APPENDIX=true
INCLUDE_APPENDIX=${INCLUDE_APPENDIX:-auto}
if [ "$INCLUDE_APPENDIX" = "true" ] || ([ "$INCLUDE_APPENDIX" = "auto" ] && [ "$MAX_CHAPTERS" -ge 21 ]); then
    MD_PATH=$(find_chapter_md "x")
    if [ -n "$MD_PATH" ] && [ -f "$MD_PATH" ]; then
        MD_FILES+=("$MD_PATH")
        echo "  ✓ Found: $MD_PATH (Appendix)"
    else
        echo "  ✗ Chapter X (Appendix) markdown not found"
    fi
fi

if [ ${#MD_FILES[@]} -eq 0 ]; then
    echo "❌ Error: No chapter markdown files found!"
    exit 1
fi

echo ""
echo "📄 Found ${#MD_FILES[@]} chapter markdown files to merge"
echo ""

# Run Python files in img directories to generate images
echo "🖼️  Running Python files in chapter img directories to generate images..."

# Determine Python command: use conda environment if available, otherwise use python3
CONDA_ENV="usao"
if command -v conda &> /dev/null; then
    # Check if conda environment exists
    if conda env list | grep -q "^${CONDA_ENV}\s"; then
        PYTHON_CMD="conda run -n ${CONDA_ENV} python"
        echo "   Using conda environment: ${CONDA_ENV}"
    else
        echo "   ⚠️  Warning: Conda environment '${CONDA_ENV}' not found, using system python3"
        PYTHON_CMD="python3"
    fi
else
    echo "   ⚠️  Warning: Conda not found, using system python3"
    PYTHON_CMD="python3"
fi

python_count=0
python_success=0
python_failed=0

for md_file in "${MD_FILES[@]}"; do
    chapter_dir="$(dirname "$md_file")"
    abs_chapter_dir="$(cd "$chapter_dir" && pwd)"
    chapter_num=$(get_chapter_number "$md_file")
    
    img_dir="$abs_chapter_dir/img"
    if [ -d "$img_dir" ]; then
        # Find all Python files in the img directory
        while IFS= read -r -d '' py_file; do
            py_basename="$(basename "$py_file")"
            py_name="${py_basename%.py}"  # Remove .py extension
            
            
            # Special handling for files that generate multiple images
            if [ "$py_basename" = "gram_matrix.py" ]; then
                png_file1="$img_dir/gram_matrix1.png"
                png_file2="$img_dir/gram_matrix2.png"
                if [ -f "$png_file1" ] && [ -f "$png_file2" ]; then
                    echo "  Skipping: $py_file (gram_matrix1.png and gram_matrix2.png already exist)"
                    continue
                fi
            elif [ "$py_basename" = "householder_qr_demo.py" ]; then
                png_file="$img_dir/${py_name}.png"
                if [ -f "$png_file" ]; then
                    echo "  Skipping: $py_file (${py_name}.png already exists)"
                    continue
                fi
            else
                png_file="$img_dir/${py_name}.png"
                # Check if PNG already exists
                if [ -f "$png_file" ]; then
                    echo "  Skipping: $py_file (${py_name}.png already exists)"
                    continue
                fi
            fi
            
            # Run the Python file if PNG doesn't exist
            python_count=$((python_count + 1))
            echo "  Running: $py_file"
            
            # Change to img directory so relative paths work correctly
            cd "$img_dir"
            # Capture stderr to show errors if execution fails
            error_output=$($PYTHON_CMD "$py_basename" 2>&1 >/dev/null)
            exit_code=$?
            if [ $exit_code -eq 0 ]; then
                python_success=$((python_success + 1))
                echo "    ✓ Success"
            else
                python_failed=$((python_failed + 1))
                echo "    ⚠️  Warning: Failed to run $py_file (exit code: $exit_code)"
                if [ -n "$error_output" ]; then
                    echo "      Error: $error_output" | head -3 | sed 's/^/        /'
                fi
            fi
            cd "$SCRIPT_DIR"
        done < <(find "$img_dir" -maxdepth 1 -name "*.py" -type f -print0 2>/dev/null)
    fi
done

if [ $python_count -gt 0 ]; then
    echo "  Summary: $python_success/$python_count Python files executed successfully"
    if [ $python_failed -gt 0 ]; then
        echo "  ⚠️  $python_failed Python file(s) failed (check output above)"
    fi
    echo ""
fi

# Create merged markdown file
MERGED_MD="$TEMP_DIR/book_merged.md"

# Add YAML metadata block at the beginning
cat > "$MERGED_MD" << 'EOF'
---
title: ""
subtitle: ""
author: ""
date: ""
documentclass: book
toc: true
toc-depth: 3
numbersections: true
lang: ""
---

EOF

# Function to find part PDF file
find_part_pdf() {
    local part_num="$1"
    find . -path "*/chapter*/part${part_num}.pdf" -type f 2>/dev/null | head -1
}

# Function to get part number for a chapter
get_part_number() {
    local chapter_num="$1"
    # Part I: chapters 1-10
    if [ "$chapter_num" -ge 1 ] && [ "$chapter_num" -le 10 ]; then
        echo "1"
    # Part II: chapters 11-12
    elif [ "$chapter_num" -ge 11 ] && [ "$chapter_num" -le 12 ]; then
        echo "2"
    # Part III: chapters 13-16
    elif [ "$chapter_num" -ge 13 ] && [ "$chapter_num" -le 16 ]; then
        echo "3"
    # Part IV: chapters 17-21
    elif [ "$chapter_num" -ge 17 ] && [ "$chapter_num" -le 21 ]; then
        echo "4"
    else
        echo ""
    fi
}

# Merge all chapter files with page breaks, renaming image paths to avoid conflicts
first_chapter=true
last_part_num=""
for md_file in "${MD_FILES[@]}"; do
    echo "  Adding: $md_file"
    
    # Extract chapter number
    chapter_num=$(get_chapter_number "$md_file")
    
    # Get part number for this chapter (only for numeric chapters)
    part_num=""
    if [ "$chapter_num" != "x" ] && [ "$chapter_num" != "unknown" ]; then
        part_num=$(get_part_number "$chapter_num")
    fi
    
    # Check if we're starting a new part
    if [ -n "$part_num" ] && [ "$part_num" != "$last_part_num" ]; then
        # Find the part PDF file
        part_pdf=$(find_part_pdf "$part_num")
        if [ -n "$part_pdf" ] && [ -f "$part_pdf" ]; then
            echo "  ✓ Found part $part_num PDF: $part_pdf"
            # Convert to absolute path
            part_pdf_abs="$(cd "$(dirname "$part_pdf")" && pwd)/$(basename "$part_pdf")"
            
            # Extract part title from the part markdown file
            part_md="${part_pdf%.pdf}.md"
            if [ -f "$part_md" ]; then
                part_info=$(python3 - "$part_md" "$part_num" << 'PYTHON_PART_EOF'
import re
import sys

part_md = sys.argv[1]
part_num = sys.argv[2]
try:
    with open(part_md, 'r', encoding='utf-8') as f:
        content = f.read()
    # Extract part title: # Part X: Title
    match = re.search(r'^# Part ([IVX]+): (.+)$', content, re.MULTILINE)
    if match:
        part_roman = match.group(1).strip()
        part_name = match.group(2).strip()
        # Escape LaTeX special characters for use in \addcontentsline
        # Escape backslashes first, then other special characters
        part_name = part_name.replace('\\', r'\textbackslash{}')
        part_name = part_name.replace('{', r'\{')
        part_name = part_name.replace('}', r'\}')
        part_name = part_name.replace('&', r'\&')
        part_name = part_name.replace('%', r'\%')
        part_name = part_name.replace('#', r'\#')
        part_name = part_name.replace('^', r'\textasciicircum{}')
        part_name = part_name.replace('_', r'\_')
        part_name = part_name.replace('$', r'\$')
        # Output: part_roman, part_name (tab-separated)
        print(f"{part_roman}\t{part_name}")
    else:
        # Map numeric part_num to Roman numeral
        roman_map = {'1': 'I', '2': 'II', '3': 'III', '4': 'IV'}
        part_roman = roman_map.get(part_num, part_num)
        print(f"{part_roman}\tPart {part_num}")
except Exception as e:
    # Map numeric part_num to Roman numeral
    roman_map = {'1': 'I', '2': 'II', '3': 'III', '4': 'IV'}
    part_roman = roman_map.get(part_num, part_num)
    print(f"{part_roman}\tPart {part_num}")
PYTHON_PART_EOF
)
                # Parse the output (tab-separated: part_roman, part_name)
                IFS=$'\t' read -r part_roman part_name <<< "$part_info"
                # Create bookmark/TOC entry: "PART I - xxx"
                part_title="PART $part_roman - $part_name"
            else
                # Map numeric part_num to Roman numeral for fallback
                case "$part_num" in
                    1) part_roman="I" ;;
                    2) part_roman="II" ;;
                    3) part_roman="III" ;;
                    4) part_roman="IV" ;;
                    *) part_roman="$part_num" ;;
                esac
                part_title="PART $part_roman"
            fi
            
            # Insert part PDF with proper bookmarks and TOC
            echo "" >> "$MERGED_MD"
            echo "\`\`\`{=latex}" >> "$MERGED_MD"
            echo "\\newpage" >> "$MERGED_MD"
            echo "\\phantomsection" >> "$MERGED_MD"
            echo "\\addcontentsline{toc}{part}{$part_title}" >> "$MERGED_MD"
            echo "\\includepdf[pages=-,pagecommand={\\thispagestyle{empty}}]{$part_pdf_abs}" >> "$MERGED_MD"
            echo "\`\`\`" >> "$MERGED_MD"
            echo "" >> "$MERGED_MD"
            last_part_num="$part_num"
        else
            echo "  ⚠️  Warning: Part $part_num PDF not found"
        fi
    fi
    
    # Add a page break before each chapter (except the first)
    if [ "$first_chapter" = false ]; then
        echo "" >> "$MERGED_MD"
        echo "\\newpage" >> "$MERGED_MD"
        echo "" >> "$MERGED_MD"
    fi
    first_chapter=false
    
    # For appendix, add \appendix command and create a centered title page
    # This prevents "Chapter 2" from appearing and allows normal section numbering
    if [ "$chapter_num" = "x" ]; then
        echo "" >> "$MERGED_MD"
        echo "\`\`\`{=latex}" >> "$MERGED_MD"
        echo "\\appendix" >> "$MERGED_MD"
        echo "\\setcounter{section}{0}" >> "$MERGED_MD"
        echo "\\renewcommand{\\thesection}{\\arabic{section}}" >> "$MERGED_MD"
        echo "\\renewcommand{\\thesubsection}{\\arabic{section}.\\arabic{subsection}}" >> "$MERGED_MD"
        echo "\\newpage" >> "$MERGED_MD"
        echo "\\thispagestyle{empty}" >> "$MERGED_MD"
        echo "\\vspace*{\\fill}" >> "$MERGED_MD"
        echo "\\begin{center}" >> "$MERGED_MD"
        echo "{\\color{chapterblue}\\fontsize{48}{58}\\selectfont\\bfseries Appendix}" >> "$MERGED_MD"
        echo "\\end{center}" >> "$MERGED_MD"
        echo "\\vspace*{\\fill}" >> "$MERGED_MD"
        echo "\\newpage" >> "$MERGED_MD"
        echo "\\bookmarksetup{startatroot}" >> "$MERGED_MD"
        echo "\\phantomsection" >> "$MERGED_MD"
        echo "\\pdfbookmark[0]{Appendix}{appendix}" >> "$MERGED_MD"
        echo "\`\`\`" >> "$MERGED_MD"
        echo "" >> "$MERGED_MD"
    fi
    
    # Process chapter content: 
    # 1. Extract Code Summary, quote, title, subtitle for chapter title page
    # 2. Generate chapter title page LaTeX code
    # 3. Remove "Code Summary" section from content (it will be in title page)
    # 4. Rename img/ paths to img/chapterN_ to avoid conflicts
    # Special handling for appendix (chapter_num = "x"): don't generate chaptertitlepage, but still process content
    if [ "$chapter_num" != "unknown" ]; then
        # Extract chapter info and generate title page with Code Summary
        extract_script="$SCRIPT_DIR/scripts/extract_chapter_title_quote.py"
        process_includes_script="$SCRIPT_DIR/scripts/process_includes.py"
        
        # First, process conditional includes to expand Code Summary content
        temp_with_includes="$TEMP_DIR/chapter_${chapter_num}_with_includes.md"
        md_file_for_processing="$md_file"
        if [ -f "$process_includes_script" ] && grep -q "<!-- include:" "$md_file" 2>/dev/null; then
            # Load config if available
            config_file="$SCRIPT_DIR/peanut.config"
            config_args=()
            if [ -f "$config_file" ]; then
                config_args=("--config" "$config_file")
            fi
            # Process includes
            python3 "$process_includes_script" --process-file "$md_file" "$temp_with_includes" "${config_args[@]}" >/dev/null 2>&1
            if [ -f "$temp_with_includes" ] && [ -s "$temp_with_includes" ]; then
                md_file_for_processing="$temp_with_includes"
            fi
        fi
        
        if [ -f "$extract_script" ]; then
            
            # Generate LaTeX title page with quote and code summary
            title_page_latex=$(python3 << PYTHON_TITLE_EOF
import re
import sys
from pathlib import Path

md_file = "$md_file_for_processing"
chapter_num = "${chapter_num}"

# Read markdown file (with includes processed if available)
content = Path(md_file).read_text(encoding='utf-8')
lines = content.split('\n')

# Extract chapter title and subtitle
chapter_num_match = re.search(r'^#\s+Chapter\s+(\d+):\s*(.+)$', content, re.MULTILINE)
if chapter_num_match:
    title = chapter_num_match.group(2).strip()
    # Remove Pandoc attributes like {-} or {.unnumbered} or {width=0.3cm} from title
    title = re.sub(r'\s*\{[^}]*\}\s*$', '', title).strip()
else:
    title = ""

subtitle_match = re.search(r'^\*\s*(.+?)\s*\*$', content, re.MULTILINE)
subtitle = subtitle_match.group(1).strip() if subtitle_match else ""

# Extract quote
quote_pattern = r'^>\s*(?:"(.+)"|(.+))$'
author_pattern = r'^-\s*(.+)$'
quote_text = ""
quote_author = ""

for i, line in enumerate(lines):
    quote_match = re.match(quote_pattern, line)
    if quote_match:
        quote_text = (quote_match.group(1) or quote_match.group(2) or "").strip()
        if quote_text and not quote_text.startswith('NOTES:') and not quote_text.startswith('NOTEE'):
            if i + 1 < len(lines):
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines):
                    author_match = re.match(author_pattern, lines[j])
                    if author_match:
                        quote_author = author_match.group(1).strip()
                        break

# Extract code summary
code_summary = ""
code_summary_start = None
for i, line in enumerate(lines):
    if re.match(r'^\*\*Code Summary\*\*', line, re.IGNORECASE):
        code_summary_start = i
        break

if code_summary_start is not None:
    summary_lines = []
    i = code_summary_start + 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith('-'):
            summary_lines.append(line.strip())
        elif line.strip() and not line.strip().startswith('#'):
            break
        elif not line.strip():
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].startswith('#'):
                break
        else:
            break
        i += 1
    if summary_lines:
        code_summary = '\n'.join(summary_lines)

# Import and use the generate function
sys.path.insert(0, "$SCRIPT_DIR/scripts")
from extract_chapter_title_quote import generate_latex_title_page_with_quote, format_title_for_latex

if title and chapter_num:
    # For appendix (chapter_num = "x"), don't generate chaptertitlepage
    # Appendices should use \appendix command in LaTeX, not chaptertitlepage
    if chapter_num == "x":
        # Don't generate title page for appendix - it will be handled differently
        pass
    else:
        # Pass raw title to generate_latex_title_page_with_quote - it will format internally
        # This matches convert_to_pdf.sh's behavior (via process_file_for_pdf)
        latex_code = generate_latex_title_page_with_quote(chapter_num, title, subtitle, quote_text, quote_author, code_summary)
        # The function already returns LaTeX code wrapped in markdown code blocks, so we can use it directly
        print(latex_code)
PYTHON_TITLE_EOF
)
            # Don't clean up temp_with_includes yet - we need it for content processing
            if [ -n "$title_page_latex" ]; then
                # Add title page LaTeX code to merged markdown
                # The latex_code already includes markdown code block markers
                printf '%s\n' "$title_page_latex" >> "$MERGED_MD"
                echo "" >> "$MERGED_MD"
            fi
        fi
        
        # Use Python to process the file: 
        # 1. Remove chapter title, subtitle, quote (already in title page)
        # 2. Remove Code Summary section (already in title page)
        # 3. Rename image paths to avoid conflicts
        # Write to temporary file first, then append to merged file
        temp_processed="$TEMP_DIR/chapter_${chapter_num}_processed.md"
        python3 << PYTHON_EOF
import re
import sys

chapter_num = "${chapter_num}"
# Use the file with includes processed (if available)
md_file_to_process = "$md_file_for_processing"
with open(md_file_to_process, 'r', encoding='utf-8') as f:
    lines = f.readlines()

output_lines = []
i = 0
chapter_pattern = r'^(#\s+Chapter\s+\d+:.*)$'
appendix_pattern = r'^#\s+Appendix\s*$'
subtitle_pattern = r'^\*\s*(.+?)\s*\*$'
quote_pattern = r'^>\s*(?:"(.+)"|(.+))$'
author_pattern = r'^-\s*(.+)$'

while i < len(lines):
    line = lines[i]
    
    # For appendix, skip the "# Appendix" heading entirely
    # This prevents "Appendix A" from appearing and sections start immediately
    if chapter_num == "x" and re.match(appendix_pattern, line):
        i += 1
        # Skip empty lines after appendix heading
        while i < len(lines) and not lines[i].strip():
            i += 1
        continue
    
    # Skip chapter title (already in title page)
    if re.match(chapter_pattern, line):
        i += 1
        # Skip empty lines after chapter title
        while i < len(lines) and not lines[i].strip():
            i += 1
        # Skip subtitle (already in title page)
        if i < len(lines) and re.match(subtitle_pattern, lines[i]):
            i += 1
        # Skip empty lines after subtitle
        while i < len(lines) and not lines[i].strip():
            i += 1
        # Skip quote block (already in title page)
        if i < len(lines):
            quote_match = re.match(quote_pattern, lines[i])
            if quote_match:
                quote_text = (quote_match.group(1) or quote_match.group(2) or "").strip()
                if quote_text and not quote_text.startswith('NOTES:') and not quote_text.startswith('NOTEE'):
                    i += 1
                    # Skip empty lines
                    while i < len(lines) and not lines[i].strip():
                        i += 1
                    # Skip author line
                    if i < len(lines) and re.match(author_pattern, lines[i]):
                        i += 1
                    # Skip empty lines after quote
                    while i < len(lines) and not lines[i].strip():
                        i += 1
        continue
    
    # Skip Code Summary section (already in title page)
    if re.match(r'^\*\*Code Summary\*\*', line, re.IGNORECASE):
        i += 1
        # Skip empty line if present
        while i < len(lines) and not lines[i].strip():
            i += 1
        # Skip all summary lines (starting with -)
        while i < len(lines) and lines[i].strip().startswith('-'):
            i += 1
        # Skip empty lines after summary
        while i < len(lines) and not lines[i].strip():
            i += 1
        continue
    
    # Skip standalone subtitle (already in title page)
    if re.match(subtitle_pattern, line):
        i += 1
        # Skip empty lines after subtitle
        while i < len(lines) and not lines[i].strip():
            i += 1
        continue
    
    # Process the line: rename image paths while preserving all attributes
    chapter_prefix = f"chapter{chapter_num}_"
    # Match img/xxx.png or img/xxx.jpg etc. in markdown image syntax
    # This regex only matches the path part (img/...), preserving any attributes after it
    # Pattern: ![alt](img/path.png){attributes} or ![alt](img/path.png)
    line = re.sub(r'\(img/([^)]+)\)', lambda m: f'(img/{chapter_prefix}{m.group(1)})', line)
    # Remove {-} from chapter titles to ensure proper numbering in merged book
    if line.startswith('# Chapter'):
        line = re.sub(r'\s*\{-\}\s*$', '', line)
    output_lines.append(line)
    
    i += 1

# Write to file instead of stdout
with open("$temp_processed", 'w', encoding='utf-8') as f:
    f.write(''.join(output_lines))
PYTHON_EOF
        # Append processed content to merged file
        if [ -f "$temp_processed" ]; then
            cat "$temp_processed" >> "$MERGED_MD"
            rm -f "$temp_processed"
        else
            echo "⚠️  Warning: Failed to process $md_file, appending original"
            cat "$md_file" >> "$MERGED_MD"
        fi
        
        # Clean up temporary file with includes
        if [ -f "$temp_with_includes" ]; then
            rm -f "$temp_with_includes"
        fi
    else
        # Fallback: just append without renaming if chapter number can't be determined
        cat "$md_file" >> "$MERGED_MD"
    fi
    echo "" >> "$MERGED_MD"
    echo "" >> "$MERGED_MD"
done

echo ""
echo "📝 Merged markdown file created: $MERGED_MD"
echo ""

# Function to get template name from peanut.config
get_template_name() {
    local config_file="$SCRIPT_DIR/peanut.config"
    if [ -f "$config_file" ]; then
        # Use Python to read JSON config and extract template name
        local template_name=$(python3 -c "
import json
import sys
try:
    with open('$config_file', 'r') as f:
        config = json.load(f)
        template = config.get('template', 'default.tpl')
        print(template)
except Exception as e:
    print('default.tpl', file=sys.stderr)
    sys.exit(0)
" 2>/dev/null)
        # If template name doesn't end with .tpl, add it
        if [[ ! "$template_name" =~ \.tpl$ ]]; then
            template_name="${template_name}.tpl"
        fi
        echo "$template_name"
    else
        echo "default.tpl"
    fi
}

# Generate LaTeX header using the same method as convert_to_pdf.sh
# Get template name from peanut.config (defaults to default.tpl)
template_name=$(get_template_name)
# Use book template for merged book
template_file="$SCRIPT_DIR/templates/book/$template_name"
if [ ! -f "$template_file" ]; then
    echo "❌ Error: Template file not found: $template_file"
    echo "   Available templates: $(ls -1 "$SCRIPT_DIR/templates"/*.tpl 2>/dev/null | xargs -n1 basename | tr '\n' ' ' || echo 'none')"
    exit 1
fi

python_script="$SCRIPT_DIR/scripts/generate_latex_header.py"
if [ ! -f "$python_script" ]; then
    echo "❌ Error: Python script not found: $python_script"
    exit 1
fi

# Use book title for header
header_title_text="Math for AI/ML"
latex_header=$(python3 "$python_script" "$template_file" "$header_title_text" "$CHAPTER_STYLE" 2>&1)
python_exit_code=$?

if [ $python_exit_code -ne 0 ] || [ -z "$latex_header" ]; then
    echo "⚠️  Warning: Python script failed, using fallback method"
    # Fallback header would go here, but for now just exit
    echo "❌ Error: Could not generate LaTeX header"
    exit 1
fi

# Write header to temporary file
header_file="$TEMP_DIR/book_header.tex"
echo "$latex_header" > "$header_file"

# Build Lua filters list (same as convert_to_pdf.sh)
lua_filters=""
if [ -f "$SCRIPT_DIR/scripts/reset_section_counter.lua" ]; then
    lua_filters="--lua-filter=$SCRIPT_DIR/scripts/reset_section_counter.lua"
fi
if [ -f "$SCRIPT_DIR/scripts/code_line_numbers.lua" ]; then
    if [ -n "$lua_filters" ]; then
        lua_filters="$lua_filters --lua-filter=$SCRIPT_DIR/scripts/code_line_numbers.lua"
    else
        lua_filters="--lua-filter=$SCRIPT_DIR/scripts/code_line_numbers.lua"
    fi
fi
if [ -f "$SCRIPT_DIR/scripts/note_sections.lua" ]; then
    if [ -n "$lua_filters" ]; then
        lua_filters="$lua_filters --lua-filter=$SCRIPT_DIR/scripts/note_sections.lua"
    else
        lua_filters="--lua-filter=$SCRIPT_DIR/scripts/note_sections.lua"
    fi
fi
if [ -f "$SCRIPT_DIR/scripts/table_beautify.lua" ]; then
    if [ -n "$lua_filters" ]; then
        lua_filters="$lua_filters --lua-filter=$SCRIPT_DIR/scripts/table_beautify.lua"
    else
        lua_filters="--lua-filter=$SCRIPT_DIR/scripts/table_beautify.lua"
    fi
fi
if [ -f "$SCRIPT_DIR/scripts/image_attributes.lua" ]; then
    if [ -n "$lua_filters" ]; then
        lua_filters="$lua_filters --lua-filter=$SCRIPT_DIR/scripts/image_attributes.lua"
    else
        lua_filters="--lua-filter=$SCRIPT_DIR/scripts/image_attributes.lua"
    fi
fi
if [ -f "$SCRIPT_DIR/scripts/equation_numbering.lua" ]; then
    if [ -n "$lua_filters" ]; then
        lua_filters="$lua_filters --lua-filter=$SCRIPT_DIR/scripts/equation_numbering.lua"
    else
        lua_filters="--lua-filter=$SCRIPT_DIR/scripts/equation_numbering.lua"
    fi
fi

# For images, we need to tell pandoc where to look
# Collect all chapter directories and common directories for resource path
resource_paths=()
# Add root directory first
resource_paths+=("$SCRIPT_DIR")
# Add each chapter directory
for md_file in "${MD_FILES[@]}"; do
    chapter_dir=$(cd "$(dirname "$md_file")" && pwd)
    resource_paths+=("$chapter_dir")
done
# Add img directory if it exists
if [ -d "$SCRIPT_DIR/img" ]; then
    resource_paths+=("$SCRIPT_DIR/img")
fi

# Build resource path argument (colon-separated for pandoc)
# Remove duplicates while preserving order
unique_paths=()
for path in "${resource_paths[@]}"; do
    is_duplicate=false
    for unique_path in "${unique_paths[@]}"; do
        if [ "$path" = "$unique_path" ]; then
            is_duplicate=true
            break
        fi
    done
    if [ "$is_duplicate" = false ]; then
        unique_paths+=("$path")
    fi
done

# Build colon-separated resource path
resource_path_arg="--resource-path=$(IFS=:; echo "${unique_paths[*]}")"

echo "🔧 Converting merged markdown to PDF with automatic TOC..."
echo "   Using Lua filters: $(echo $lua_filters | wc -w) filters"
echo ""

# First generate LaTeX, then beautify tables, then compile to PDF
temp_tex_file="$TEMP_DIR/book_merged.tex"

# Check for custom cover.pdf first (from Canva or other design tools)
# If cover.pdf exists, use it directly; otherwise use generated cover-background
cover_bg=""
cover_is_full_page=false

if [ -f "$SCRIPT_DIR/cover.pdf" ]; then
    echo "  Found custom cover.pdf (from Canva/design tool)"
    cover_bg="$SCRIPT_DIR/cover.pdf"
    cover_is_full_page=true
elif [ -f "$SCRIPT_DIR/img/cover-background.pdf" ] || [ -f "$SCRIPT_DIR/img/cover-background.png" ] || [ -f "$SCRIPT_DIR/img/cover-background.jpg" ]; then
    echo "  Using generated cover background..."
    if [ -f "$SCRIPT_DIR/img/cover-background.pdf" ]; then
        cover_bg="$SCRIPT_DIR/img/cover-background.pdf"
    elif [ -f "$SCRIPT_DIR/img/cover-background.png" ]; then
        cover_bg="$SCRIPT_DIR/img/cover-background.png"
    elif [ -f "$SCRIPT_DIR/img/cover-background.jpg" ]; then
        cover_bg="$SCRIPT_DIR/img/cover-background.jpg"
    fi
else
    # Generate cover background if it doesn't exist
    cover_gen_script="$SCRIPT_DIR/scripts/generate_cover_background.py"
    if [ -f "$cover_gen_script" ]; then
        echo "  Generating cover background image..."
        if python3 "$cover_gen_script" 2>/dev/null; then
            echo "  ✓ Cover background generated"
            if [ -f "$SCRIPT_DIR/img/cover-background.pdf" ]; then
                cover_bg="$SCRIPT_DIR/img/cover-background.pdf"
            fi
        else
            echo "  ⚠️  Warning: Failed to generate cover background (continuing anyway)"
        fi
    fi
fi

# Add cover page if cover file exists
if [ -n "$cover_bg" ]; then
    echo "  Adding cover page..."
    
    # Use absolute path for the cover file
    cover_bg_abs="$(cd "$(dirname "$cover_bg")" && pwd)/$(basename "$cover_bg")"
    # Create a temporary cover page markdown
    cover_md="$TEMP_DIR/cover_page.md"
    
    if [ "$cover_is_full_page" = true ]; then
        # For full-page cover PDF (from Canva), insert it directly using pdfpages
        {
            echo '```{=latex}'
            echo "\\phantomsection"
            echo "\\pdfbookmark[0]{Cover}{cover}"
            echo "\\includepdf[pages={1},pagecommand={\\thispagestyle{empty}}]{$cover_bg_abs}"
            echo '```'
            echo ""
        } > "$cover_md"
    else
        # For background image, use coverpagewithbackground command
        {
            echo '```{=latex}'
            echo "\\phantomsection"
            echo "\\pdfbookmark[0]{Cover}{cover}"
            echo "\\coverpagewithbackground{$cover_bg_abs}"
            echo '```'
            echo ""
        } > "$cover_md"
    fi
    
    # Prepend cover page to merged markdown
    cat "$cover_md" "$MERGED_MD" > "$MERGED_MD.tmp"
    mv "$MERGED_MD.tmp" "$MERGED_MD"
fi

# Add preface page if preface.md exists
preface_file="$SCRIPT_DIR/preface.md"
if [ -f "$preface_file" ]; then
    echo "  Adding preface page..."
    # Create a temporary preface markdown with proper formatting
    preface_md="$TEMP_DIR/preface_page.md"
    
    # Process preface content: skip YAML frontmatter and the first "# Preface" heading
    temp_preface_content="$TEMP_DIR/preface_content.md"
    python3 << PYTHON_PREFACE_EOF
import re
import sys

preface_file = "$preface_file"
output_file = "$temp_preface_content"

with open(preface_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

output_lines = []
skip_yaml = False
yaml_started = False
skip_first_heading = True
i = 0

while i < len(lines):
    line = lines[i]
    
    # Skip YAML frontmatter
    if line.strip() == '---':
        if not yaml_started:
            yaml_started = True
            skip_yaml = True
            i += 1
            continue
        else:
            skip_yaml = False
            i += 1
            continue
    
    if skip_yaml:
        i += 1
        continue
    
    # Skip the first "# Preface" heading (we add our own title)
    if skip_first_heading and re.match(r'^#\s+Preface\s*$', line):
        skip_first_heading = False
        i += 1
        # Skip empty line after heading if present
        if i < len(lines) and not lines[i].strip():
            i += 1
        continue
    
    # Pass through all other content (markdown will be processed by Pandoc)
    output_lines.append(line)
    i += 1

# Write the processed content
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(''.join(output_lines))
PYTHON_PREFACE_EOF
    
    # Create preface markdown - add Preface in frontmatter with unnumbered sections
    # Use \chapter* for unnumbered chapter in frontmatter, then manually add to TOC
    cat > "$preface_md" << PREFACE_EOF
\`\`\`{=latex}
\\frontmatter
\\setcounter{secnumdepth}{0}
\\phantomsection
\\pdfbookmark[0]{Preface}{preface}
\\addcontentsline{toc}{chapter}{Preface}
\`\`\`

\`\`\`{=latex}
\\chapter*{Preface}
\`\`\`

PREFACE_EOF
    
    # Add the preface content as regular markdown (Pandoc will process it)
    cat "$temp_preface_content" >> "$preface_md"
    echo "" >> "$preface_md"
    echo "\`\`\`{=latex}" >> "$preface_md"
    echo "\\setcounter{secnumdepth}{3}" >> "$preface_md"
    echo "\`\`\`" >> "$preface_md"
    
    # Insert preface right after cover page in the merged markdown
    # The cover is prepended, so we need to insert preface after it but before chapters
    if [ -n "$cover_bg" ]; then
        # Find where the cover block ends (after ```) and insert preface there
        MERGED_MD="$MERGED_MD" PREFACE_MD="$preface_md" python3 << 'PYTHON_INSERT_PREFACE_EOF'
import sys
import re
import os

merged_file = os.environ.get('MERGED_MD')
preface_file = os.environ.get('PREFACE_MD')

with open(merged_file, 'r', encoding='utf-8') as f:
    merged_content = f.read()

with open(preface_file, 'r', encoding='utf-8') as f:
    preface_content = f.read()

# Find the end of the cover LaTeX block
# Cover block pattern: ```{=latex}\n\coverpagewithbackground{...}\n```
# Escape backticks for bash heredoc
backtick = '`'
cover_end_pattern = backtick + backtick + backtick + r'\{=latex\}.*?' + backtick + backtick + backtick
match = re.search(cover_end_pattern, merged_content, re.DOTALL)
if match:
    # Insert preface right after the cover block ends
    insert_pos = match.end()
    # Add newline if needed
    if merged_content[insert_pos:insert_pos+1] != '\n':
        preface_content = preface_content + '\n'
    new_content = merged_content[:insert_pos] + '\n' + preface_content + merged_content[insert_pos:]
else:
    # Fallback: if cover pattern not found, prepend preface (shouldn't happen)
    new_content = preface_content + '\n' + merged_content

with open(merged_file, 'w', encoding='utf-8') as f:
    f.write(new_content)
PYTHON_INSERT_PREFACE_EOF
    else
        # Preface goes at the beginning if no cover
        cat "$preface_md" "$MERGED_MD" > "$MERGED_MD.tmp"
        mv "$MERGED_MD.tmp" "$MERGED_MD"
    fi
fi

# Convert to LaTeX first
# Use book document class for merged chapters with TOC
# --standalone is needed to generate complete document including title page
# Change to temp directory so relative image paths work correctly (like convert_to_pdf.sh)
# But header_file path needs to be absolute or relative to temp dir
cd "$TEMP_DIR"
header_file_basename="$(basename "$header_file")"
if pandoc_output=$(pandoc "$(basename "$MERGED_MD")" -o "$(basename "$temp_tex_file")" \
    --standalone \
    --from=markdown+raw_tex+link_attributes \
    $lua_filters \
    $resource_path_arg \
    --to=latex \
    --top-level-division=chapter \
    --toc \
    --toc-depth=3 \
    --number-sections \
    -V geometry:margin=1in \
    -V secnumdepth=3 \
    --syntax-highlighting=tango \
    -V "chapternumberstyle=$CHAPTER_STYLE" \
    -H "$header_file" \
    2>&1); then
    
    # Check if LaTeX file was created
    if [ ! -f "$(basename "$temp_tex_file")" ]; then
        echo "❌ Error: LaTeX file was not created: $(basename "$temp_tex_file")"
        echo "Pandoc output: $pandoc_output"
        cd "$SCRIPT_DIR"
        exit 1
    fi
    
    # Return to script directory before beautifying tables
    cd "$SCRIPT_DIR"
    
    # Save a copy before beautify for comparison
    cp "$temp_tex_file" "$temp_tex_file.before_beautify" 2>/dev/null || true
    
    # Beautify tables in the generated LaTeX (use absolute path)
    if [ -f "$SCRIPT_DIR/scripts/beautify_tables.py" ]; then
        echo "  Beautifying tables..."
        python3 "$SCRIPT_DIR/scripts/beautify_tables.py" "$temp_tex_file" || true
        # Verify file still exists after beautify
        if [ ! -f "$temp_tex_file" ]; then
            echo "⚠️  Warning: LaTeX file missing after beautify, restoring from backup"
            cp "$temp_tex_file.before_beautify" "$temp_tex_file" 2>/dev/null || true
        fi
    fi
    
    # Fix order: Insert cover page and preface before TOC
    # ROOT CAUSE: Cover page and preface are added to markdown before TOC, but Pandoc places TOC first
    # We need to manually reorder: cover -> preface -> TOC -> chapters
    if [ -n "$cover_bg" ] || [ -f "$preface_file" ]; then
        echo "  Reordering: cover page and preface before TOC..."
        cover_bg_abs=""
        if [ -n "$cover_bg" ]; then
            cover_bg_abs="$(cd "$(dirname "$cover_bg")" && pwd)/$(basename "$cover_bg")"
        fi
        # Check if cover.pdf is a full-page cover (from Canva) or background image
        cover_is_full_page=false
        if [ -n "$cover_bg_abs" ]; then
            # Check if it's the root cover.pdf (full page) or img/cover-background.pdf (background)
            if [[ "$cover_bg_abs" == *"/cover.pdf" ]] && [[ "$cover_bg_abs" != *"/img/"* ]]; then
                cover_is_full_page=true
            fi
        fi
        
        # Call the Python script to reorder cover and preface
        reorder_script="$SCRIPT_DIR/scripts/reorder_cover_preface.py"
        if [ ! -f "$reorder_script" ]; then
            echo "❌ Error: Python script not found: $reorder_script"
            exit 1
        fi
        
        # Build arguments for the Python script
        reorder_args=("$temp_tex_file")
        if [ -n "$cover_bg_abs" ]; then
            reorder_args+=("--cover-bg" "$cover_bg_abs")
            if [ "$cover_is_full_page" = true ]; then
                reorder_args+=("--cover-is-full-page")
            fi
        fi
        if [ -f "$preface_file" ]; then
            reorder_args+=("--preface-file" "$preface_file")
        fi
        
        python3 "$reorder_script" "${reorder_args[@]}"
    fi
    
    # Fix TOC: Add \newpage after \tableofcontents
    # ROOT CAUSE: TOC and wrapfigure on the same page causes wrapfigure to not work
    echo "  Adding page break after TOC..."
    python3 << PYTHON_EOF
import sys
import os
tex_file = "$temp_tex_file"
if not os.path.exists(tex_file):
    print(f"  Error: File not found: {tex_file}", file=sys.stderr)
    sys.exit(1)

with open(tex_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
i = 0
fixed = False

while i < len(lines):
    line = lines[i]
    new_lines.append(line)
    
    # If this is tableofcontents, check if next line is newpage
    if '\\\\tableofcontents' in line:
        # Check next line
        if i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if '\\\\newpage' not in next_line and '\\\\clearpage' not in next_line:
                # Add newpage (use actual newline character, not literal \n)
                new_lines.append('\\\\newpage\n')
                fixed = True
                print("  Added \\\\newpage after TOC", file=sys.stderr)
    
    i += 1

if fixed:
    with open(tex_file, 'w', encoding='utf-8') as f:
        f.write(''.join(new_lines))
else:
    print("  TOC already has page break", file=sys.stderr)
PYTHON_EOF
    
    # Fix section numbering: reset section counter before first \section after each chaptertitlepage
    # ROOT CAUSE: In book class, section numbering should restart at 1 for each chapter
    # But since we use custom chaptertitlepage instead of \chapter, we need to manually reset
    # Reset must be before the first \section command, after \newpage
    echo "  Fixing section numbering for book class..."
    python3 << PYTHON_EOF
import sys
import os
tex_file = "$temp_tex_file"
if not os.path.exists(tex_file):
    print(f"  Error: File not found: {tex_file}", file=sys.stderr)
    sys.exit(1)

with open(tex_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
i = 0
fixed_count = 0
last_was_chapter_title = False

while i < len(lines):
    line = lines[i]
    
    # Check if this is the end of chaptertitlepage
    if '\\\\end{chaptertitlepage}' in line:
        new_lines.append(line)
        last_was_chapter_title = True
        i += 1
        # Skip \newpage and blank lines, but remove any existing setcounter commands
        while i < len(lines):
            if '\\\\newpage' in lines[i] or not lines[i].strip():
                new_lines.append(lines[i])
                i += 1
            elif 'setcounter{section}' in lines[i] or 'setcounter{subsection}' in lines[i]:
                # Skip existing setcounter commands (we'll add them in the right place)
                i += 1
            else:
                break
    # Check if this is the first \section after a chaptertitlepage
    elif last_was_chapter_title and r'\section{' in line:
        # Insert reset commands before the first section
        new_lines.append(r'\setcounter{section}{0}' + '\n')
        new_lines.append(r'\setcounter{subsection}{0}' + '\n')
        new_lines.append(line)
        last_was_chapter_title = False
        fixed_count += 1
        i += 1
    else:
        new_lines.append(line)
        i += 1

if fixed_count > 0:
    with open(tex_file, 'w', encoding='utf-8') as f:
        f.write(''.join(new_lines))
    print(f"  Fixed section counter reset before {fixed_count} first section(s)", file=sys.stderr)
else:
    print("  Section counter reset already correct", file=sys.stderr)
PYTHON_EOF
    
    # Fix wrapfigure placement: ensure wrapfigure has blank lines before and after
    # Based on comparison: working single chapter has blank lines around wrapfigure
    echo "  Fixing wrapfigure placement..."
    python3 << PYTHON_EOF
import sys
import os
tex_file = "$temp_tex_file"
if not os.path.exists(tex_file):
    print(f"  Error: File not found: {tex_file}", file=sys.stderr)
    sys.exit(1)

with open(tex_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
i = 0
fixed_count = 0

while i < len(lines):
    line = lines[i]
    
    # If this is the start of wrapfigure
    if '\\\\begin{wrapfigure}' in line:
        # Ensure blank line before wrapfigure (if previous line is not blank and not empty)
        if new_lines and new_lines[-1].strip() != '':
            new_lines.append('\\\\n')
            fixed_count += 1
        
        # Add wrapfigure
        new_lines.append(line)
        i += 1
        # Collect rest of wrapfigure
        while i < len(lines) and '\\\\end{wrapfigure}' not in lines[i]:
            new_lines.append(lines[i])
            i += 1
        if i < len(lines):
            new_lines.append(lines[i])  # Add \end{wrapfigure}
        i += 1
        
        # Ensure blank line after wrapfigure
        if i < len(lines) and lines[i].strip() != '':
            new_lines.append('\\\\n')
            fixed_count += 1
        
        # Add next line
        if i < len(lines):
            new_lines.append(lines[i])
            i += 1
    else:
        new_lines.append(line)
        i += 1

if fixed_count > 0:
    with open(tex_file, 'w', encoding='utf-8') as f:
        f.write(''.join(new_lines))
    print(f"  Added {fixed_count} blank line(s) around wrapfigure", file=sys.stderr)
else:
    print("  Wrapfigure placement already correct", file=sys.stderr)
PYTHON_EOF
    
    # Fix alt attribute in includegraphics: remove alt={...} from \includegraphics options
    # ROOT CAUSE: Pandoc 3.8+ automatically adds alt attribute from markdown alt text,
    # but LaTeX \includegraphics doesn't support the alt key, causing "Package keyval Error: alt undefined"
    echo "  Removing alt attribute from includegraphics commands..."
    python3 << PYTHON_EOF
import sys
import os
import re
tex_file = "$temp_tex_file"
if not os.path.exists(tex_file):
    print(f"  Error: File not found: {tex_file}", file=sys.stderr)
    sys.exit(1)

with open(tex_file, 'r', encoding='utf-8') as f:
    content = f.read()

original_content = content

# Use string replacement approach to avoid regex escape issues
# Find all includegraphics commands and process them
lines = content.split('\n')
fixed_lines = []
includegraphics_cmd = r'\includegraphics['
for line in lines:
    if includegraphics_cmd in line and 'alt=' in line:
        # Find the start of options bracket
        start_idx = line.find(includegraphics_cmd)
        if start_idx != -1:
            bracket_start = start_idx + len(includegraphics_cmd)
            # Find matching closing bracket
            bracket_count = 1
            i = bracket_start
            while i < len(line) and bracket_count > 0:
                if line[i] == '[':
                    bracket_count += 1
                elif line[i] == ']':
                    bracket_count -= 1
                i += 1
            if bracket_count == 0:
                # Extract options
                options = line[bracket_start:i-1]
                # Remove alt={...} pattern
                fixed_options = re.sub(r',?\s*alt\s*=\s*\{[^}]*\},?', '', options)
                # Clean up commas
                fixed_options = re.sub(r',\s*,', ',', fixed_options)
                fixed_options = re.sub(r'^\s*,', '', fixed_options)
                fixed_options = re.sub(r',\s*$', '', fixed_options)
                # Reconstruct line
                new_line = line[:bracket_start] + fixed_options + line[i-1:]
                fixed_lines.append(new_line)
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)
    else:
        fixed_lines.append(line)

fixed_content = '\n'.join(fixed_lines)

if fixed_content != original_content:
    with open(tex_file, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    count = len(re.findall(r'alt\s*=\s*\{', original_content)) - len(re.findall(r'alt\s*=\s*\{', fixed_content))
    print(f"  ✓ Removed {count} alt attribute(s) from includegraphics commands", file=sys.stderr)
else:
    print("  No alt attributes found in includegraphics", file=sys.stderr)
PYTHON_EOF

    
    # Create symlinks to img directories for LaTeX compilation
    # 1. Symlink root img to parent of temp directory (for ../img/note32x32.png in template)
    # 2. Symlink chapter images with chapterN_ prefix to match renamed paths in merged markdown
    TEMP_PARENT="$(dirname "$TEMP_DIR")"
    # Convert to absolute paths
    ABS_SCRIPT_DIR="$(cd "$SCRIPT_DIR" && pwd)"
    
    if [ -d "$ABS_SCRIPT_DIR/img" ]; then
        echo "  Creating symlinks to img directories for LaTeX compilation..."
        # Symlink root img to parent for ../img paths (use absolute path)
        ln -sf "$ABS_SCRIPT_DIR/img" "$TEMP_PARENT/img" 2>/dev/null || true
    fi
    
    # Create a merged img directory in temp with symlinks to all images
    # Root img files don't need renaming (they use ../img/ paths in template)
    # Chapter img files need chapterN_ prefix to match renamed paths in merged markdown
    mkdir -p "$TEMP_DIR/img"
    
    # Symlink all images from root img (no prefix needed, for ../img/ paths)
    if [ -d "$ABS_SCRIPT_DIR/img" ]; then
        for img_file in "$ABS_SCRIPT_DIR/img"/*; do
            if [ -f "$img_file" ]; then
                ln -sf "$img_file" "$TEMP_DIR/img/$(basename "$img_file")" 2>/dev/null || true
            fi
        done
    fi
    
    # Symlink all images from chapter img directories with chapterN_ prefix
    # This matches the renamed paths in the merged markdown (img/chapterN_image.png)
    # Return to script directory first to resolve relative paths correctly
    cd "$SCRIPT_DIR"
    for md_file in "${MD_FILES[@]}"; do
        chapter_dir="$(dirname "$md_file")"
        abs_chapter_dir="$(cd "$chapter_dir" && pwd)"
        chapter_num=$(get_chapter_number "$md_file")
        
        if [ -d "$abs_chapter_dir/img" ] && [ "$chapter_num" != "unknown" ]; then
            for img_file in "$abs_chapter_dir/img"/*; do
                if [ -f "$img_file" ]; then
                    # Use chapterN_ prefix to match renamed paths in merged markdown
                    new_name="chapter${chapter_num}_$(basename "$img_file")"
                    ln -sf "$img_file" "$TEMP_DIR/img/$new_name" 2>/dev/null || true
                fi
            done
        fi
    done
    
    # Compile LaTeX to PDF (run twice for proper cross-references and TOC)
    echo "  Compiling LaTeX to PDF (first pass)..."
    log_file="$TEMP_DIR/$(basename "$temp_tex_file" .tex).log"
    tex_basename="$(basename "$temp_tex_file")"
    
    # Change to temp directory so relative paths (like ../img) resolve correctly
    cd "$TEMP_DIR"
    
    # Clean any existing auxiliary files from previous runs to avoid corruption issues
    rm -f "$(basename "$temp_tex_file" .tex)".{aux,log,out,toc} 2>/dev/null || true
    
    # Run xelatex and capture output
    if xelatex -interaction=nonstopmode "$tex_basename" > "xelatex_first.log" 2>&1; then
        echo "  Compiling LaTeX to PDF (second pass for TOC and cross-references)..."
        xelatex -interaction=nonstopmode "$tex_basename" > "xelatex_second.log" 2>&1 || true
        
        # Change back to script directory
        cd "$SCRIPT_DIR"
        
        # Change back to script directory (if not already done)
        cd "$SCRIPT_DIR"
        
        # Move the generated PDF to the target location
        generated_pdf="$TEMP_DIR/$(basename "$temp_tex_file" .tex).pdf"
        if [ -f "$generated_pdf" ]; then
            mv "$generated_pdf" "$OUTPUT_FILE"
            FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
            echo ""
            echo "✅ Successfully created $OUTPUT_FILE ($FILE_SIZE)"
            echo "   Merged ${#MD_FILES[@]} chapters with automatic TOC"
            echo "   TOC includes page numbers based on the merged book"
        else
            echo "❌ Error: PDF was not generated"
            echo ""
            echo "📋 Last 50 lines of LaTeX log:"
            tail -50 "$log_file" 2>/dev/null || echo "   (Log file not found)"
            echo ""
            echo "   Full log file: $log_file"
            echo "   Temp directory preserved: $TEMP_DIR"
            exit 1
        fi
    else
        # Change back to script directory
        cd "$SCRIPT_DIR"
        
        echo "❌ Error: xelatex compilation failed"
        echo ""
        echo "📋 LaTeX errors from log file:"
        if [ -f "$log_file" ]; then
            # Extract error messages from log
            grep -A 5 "! " "$log_file" | head -30 || tail -50 "$log_file"
        else
            echo "   (Log file not found, showing xelatex output:)"
            cat "$TEMP_DIR/xelatex_first.log" 2>/dev/null || echo "   (No output captured)"
        fi
        echo ""
        echo "   Full log file: $log_file"
        echo "   Temp directory preserved: $TEMP_DIR"
        echo "   You can inspect the LaTeX file: $temp_tex_file"
        exit 1
    fi
else
    echo "❌ Error: Pandoc conversion failed"
    echo "$pandoc_output"
    exit 1
fi

# Clean up temporary LaTeX files (but keep temp dir for debugging)
# Only clean up on success
if [ -f "$OUTPUT_FILE" ]; then
    echo ""
    echo "🧹 Cleaning up temporary files..."
    # Clean up auxiliary files but keep .tex and .md for debugging
    rm -f "$TEMP_DIR"/*.{aux,log,out,toc} 2>/dev/null || true
    echo "   Temp directory: $TEMP_DIR (preserved for debugging if needed)"
else
    echo ""
    echo "⚠️  Temp directory preserved for debugging: $TEMP_DIR"
fi
