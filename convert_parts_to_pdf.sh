#!/bin/bash

# Script to convert part markdown files to PDF using pandoc
# Usage:
#   ./convert_parts_to_pdf.sh                    # Convert all part*.md files
#   ./convert_parts_to_pdf.sh 1                  # Convert part1.md
#   ./convert_parts_to_pdf.sh part1              # Convert specific part (alternative)

set -e  # Exit on error

# Get the script directory (root of the project)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Function to convert markdown to LaTeX formatting
markdown_to_latex() {
    local text="$1"
    # Convert markdown bold **text** to LaTeX \textbf{text}
    # Need to be careful with nested formatting
    text=$(echo "$text" | sed 's/\*\*\([^*]*\)\*\*/\\textbf{\1}/g')
    # Convert markdown italic *text* to LaTeX \textit{text} (but avoid conflicts)
    text=$(echo "$text" | sed 's/\([^\\]\)\*\([^*]*\)\*\([^}]\)/\1\\textit{\2}\3/g')
    echo "$text"
}

# Function to escape LaTeX special characters while preserving LaTeX commands
escape_latex_preserve_commands() {
    local text="$1"
    # First, convert any remaining markdown to LaTeX
    text=$(markdown_to_latex "$text")
    
    # Since we're inserting into LaTeX, we need to escape special chars
    # BUT preserve LaTeX commands. Use Python for reliable handling.
    printf '%s' "$text" | python3 <<'PYTHON_ESCAPE'
import re
import sys

text = sys.stdin.read()

# Protect LaTeX commands with placeholders
placeholders = {}
counter = 0

def protect(match):
    global counter
    placeholder = f"__LATEX{counter}__"
    placeholders[placeholder] = match.group(0)
    counter += 1
    return placeholder

# Protect \textbf{...} and \textit{...} commands
text = re.sub(r'\\(?:textbf|textit)\{[^}]+\}', protect, text)

# Now escape special characters (but NOT backslashes - needed for LaTeX)
text = text.replace('$', '\\$')
text = text.replace('&', '\\&')
text = text.replace('%', '\\%')
text = text.replace('#', '\\#')
text = text.replace('^', '\\textasciicircum{}')
# Escape underscores and braces that are NOT in placeholders
text = text.replace('_', '\\_')
text = text.replace('{', '\\{')
text = text.replace('}', '\\}')

# Restore LaTeX commands (they have proper backslashes and braces)
for placeholder, cmd in sorted(placeholders.items(), key=lambda x: -len(x[0])):
    text = text.replace(placeholder, cmd)

print(text, end='')
PYTHON_ESCAPE
}

# Function to escape LaTeX special characters (for text without LaTeX commands)
escape_latex() {
    echo "$1" | sed 's/\\/\\\\/g; s/{/\\{/g; s/}/\\}/g; s/\$/\\$/g; s/&/\\&/g; s/%/\\%/g; s/#/\\#/g; s/\^/\\textasciicircum{}/g; s/_/\\_/g'
}

# Function to convert a part markdown file to PDF
convert_part_to_pdf() {
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
    
    # Change to the directory containing the part file
    cd "$(dirname "$abs_md_file")"
    
    # Extract part information from markdown using Python for better parsing
    local part_info=$(python3 <<PYTHON_EOF
import re
import sys

with open('$md_basename', 'r', encoding='utf-8') as f:
    content = f.read()

# Extract part title: # Part X: Title
part_title_match = re.search(r'^# Part ([IVX]+): (.+)$', content, re.MULTILINE)
part_num = part_title_match.group(1) if part_title_match else ""
part_name = part_title_match.group(2) if part_title_match else ""

# Extract chapter range: **Chapters X–Y**
chapter_range_match = re.search(r'\*\*Chapters (.+?)\*\*', content)
chapter_range = chapter_range_match.group(1) if chapter_range_match else ""

# Extract subtitle (italic lines after chapter range, before quote)
subtitle_lines = []
in_subtitle = False
for line in content.split('\n'):
    if line.startswith('**Chapters'):
        in_subtitle = True
        continue
    if line.startswith('>'):
        break
    if in_subtitle and line.strip().startswith('*') and not line.strip().startswith('**'):
        subtitle_lines.append(line.strip().strip('*').strip())
subtitle = ' '.join(subtitle_lines)

# Extract quote (blockquote)
quote_match = re.search(r'^> (.+)$', content, re.MULTILINE)
quote = quote_match.group(1).strip() if quote_match else ""

# Extract body text (paragraphs after quote)
body_lines = []
quote_found = False
for line in content.split('\n'):
    if line.startswith('>'):
        quote_found = True
        continue
    # After quote, collect all non-empty lines that aren't headers or another quote
    if quote_found:
        if line.strip() and not line.startswith('#') and not line.startswith('>'):
            # Keep the line as-is (preserve markdown formatting for later conversion)
            body_lines.append(line.strip())
        elif not line.strip() and body_lines:
            # Empty line after content - add it to preserve paragraph breaks
            body_lines.append('')
body_text = '\n\n'.join(body_lines)

# Output as tab-separated values for bash to parse
print(f"{part_num}\t{part_name}\t{chapter_range}\t{subtitle}\t{quote}\t{body_text}")
PYTHON_EOF
)
    
    # Parse the Python output
    IFS=$'\t' read -r part_num part_name chapter_range subtitle quote body_text <<< "$part_info"
    
    # Escape all text for LaTeX (keep markdown formatting as-is)
    local escaped_part_num=$(escape_latex "$part_num")
    local escaped_part_name=$(escape_latex "$part_name")
    local escaped_chapter_range=$(escape_latex "$chapter_range")
    local escaped_subtitle=$(escape_latex "$subtitle")
    # For quote, just escape special characters (keep ** as-is)
    local escaped_quote=$(escape_latex "$quote")
    
    # Process body text - escape special characters (keep markdown formatting as-is)
    # Use Python to handle paragraph splitting and escaping in one step
    local body_latex=$(printf '%s' "$body_text" | python3 <<'PYTHON_BODY'
import sys

text = sys.stdin.read()

# Function to escape LaTeX special characters
def escape_latex(s):
    s = s.replace('\\', '\\\\')
    s = s.replace('{', '\\{')
    s = s.replace('}', '\\}')
    s = s.replace('$', '\\$')
    s = s.replace('&', '\\&')
    s = s.replace('%', '\\%')
    s = s.replace('#', '\\#')
    s = s.replace('^', '\\textasciicircum{}')
    s = s.replace('_', '\\_')
    return s

# Split into paragraphs
paragraphs = []
current_para = []

for line in text.split('\n'):
    stripped = line.strip()
    if stripped:
        current_para.append(stripped)
    else:
        if current_para:
            para_text = ' '.join(current_para)
            escaped_para = escape_latex(para_text)
            paragraphs.append(escaped_para)
            current_para = []

# Add last paragraph
if current_para:
    para_text = ' '.join(current_para)
    escaped_para = escape_latex(para_text)
    paragraphs.append(escaped_para)

# Join with double newline for LaTeX paragraph breaks
result = '\n\n'.join(paragraphs)
print(result, end='')
PYTHON_BODY
)
    
    # If Python processing failed or returned empty, use fallback
    if [ -z "$body_latex" ] && [ -n "$body_text" ]; then
        # Fallback: simple escape of the entire body text
        body_latex=$(escape_latex "$body_text")
    fi
    
    # Create LaTeX header
    local latex_header=$(cat <<PART_HEADER_EOF
% Unicode support
\ifxetex
  \usepackage{fontspec}
\else
  \usepackage[utf8]{inputenc}
  \usepackage[T1]{fontenc}
  \usepackage{textcomp}
\fi
\usepackage{microtype}
\sloppy
\setlength{\emergencystretch}{3em}
\setlength{\tolerance}{1000}
\allowdisplaybreaks
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{calc}
\usepackage{geometry}
\geometry{margin=1in}
% Define colors
\definecolor{chapterblue}{RGB}{0,102,204}
\definecolor{chapterbluelight}{RGB}{153,204,255}
\definecolor{chaptergray}{RGB}{128,128,128}
\definecolor{dividerred}{RGB}{255,0,0}
% Remove page numbers
\pagestyle{empty}
PART_HEADER_EOF
)
    
    # Create a temporary LaTeX file for the part page
    local temp_tex_file="${md_basename}.part.$$.tex"
    cat > "$temp_tex_file" <<PART_TEX_EOF
\\documentclass[11pt]{article}
\\usepackage[utf8]{inputenc}
\\usepackage[T1]{fontenc}
\\usepackage{graphicx}
\\usepackage{xcolor}
\\usepackage{tikz}
\\usetikzlibrary{calc}
\\usepackage{geometry}
\\geometry{margin=1in}
\\pagestyle{empty}

% Define colors
\\definecolor{chapterblue}{RGB}{0,102,204}
\\definecolor{chapterbluelight}{RGB}{153,204,255}
\\definecolor{chaptergray}{RGB}{128,128,128}

\\begin{document}
\\thispagestyle{empty}
\\vspace*{-1.5cm}
\\begin{center}
  \\vspace{3cm}
  
  % Part number in decorative box
  \\begin{tikzpicture}
    \\coordinate (part-pos) at (0,0);
    % Decorative box background
    \\fill[chapterbluelight!20] (-3cm,-1cm) rectangle (3cm,1cm);
    \\draw[chapterblue,line width=3pt] (-3cm,-1cm) rectangle (3cm,1cm);
    % Part number
    \\node[font=\\fontsize{72}{86}\\selectfont\\bfseries,text=chapterblue] at (part-pos) {$escaped_part_num};
  \\end{tikzpicture}
  
  \\vspace{1.5cm}
  
  % Part title
  {\\color{chapterblue}\\fontsize{32}{40}\\selectfont\\bfseries $escaped_part_name}\\\\[0.8cm]
  
  % Chapter range
  {\\color{chaptergray}\\large\\bfseries Chapters $escaped_chapter_range}\\\\[0.6cm]
  
  % Subtitle
  \\begin{minipage}{0.85\\textwidth}
    \\centering
    {\\itshape\\normalsize\\color{black} $escaped_subtitle}
  \\end{minipage}
  
  \\vspace{0.8cm}
  
  % Quote/philosophy statement
  \\begin{minipage}{0.8\\textwidth}
    \\centering
    \\begin{tikzpicture}
      \\draw[chapterbluelight,line width=0.8pt] (0,0) -- (0.95\\textwidth,0);
      \\fill[chapterbluelight!20] (0.95\\textwidth,0) circle (0.12);
      \\draw[chapterbluelight,line width=0.8pt] (0.95\\textwidth,0) circle (0.12);
    \\end{tikzpicture}
    \\par\\vspace{0.3cm}
    \\noindent
    \\begin{minipage}[t]{0.9\\textwidth}
      \\centering
      \\itshape
      \\fontsize{14}{18}\\selectfont
      \\color{chapterblue}
      \\bfseries
      $escaped_quote
    \\end{minipage}
  \\end{minipage}
  
  \\vspace{1.2cm}
  
  % Body text
  \\begin{minipage}{0.85\\textwidth}
    \\raggedright
    \\normalsize
    \\color{black}
    \\setlength{\\parindent}{0pt}
    \\setlength{\\parskip}{0.6em}
    $body_latex
  \\end{minipage}
  
  \\vfill
\\end{center}
\\end{document}
PART_TEX_EOF
    
    # Cleanup function
    cleanup_all() {
        if [ -n "$temp_tex_file" ] && [ -f "$temp_tex_file" ]; then
            rm -f "$temp_tex_file"
        fi
    }
    
    # Convert LaTeX to PDF
    local current_dir=$(pwd)
    if pdflatex_output=$(cd "$current_dir" && pdflatex -interaction=nonstopmode "$temp_tex_file" 2>&1); then
        # Move PDF to correct location
        local tex_basename=$(basename "$temp_tex_file" .tex)
        if [ -f "${current_dir}/${tex_basename}.pdf" ]; then
            mv "${current_dir}/${tex_basename}.pdf" "$pdf_basename"
            # Clean up aux files
            rm -f "${current_dir}/${tex_basename}.aux" "${current_dir}/${tex_basename}.log"
        fi
        echo "✅ Successfully converted using pdflatex"
        cleanup_all
        return 0
    elif xelatex_output=$(cd "$current_dir" && xelatex -interaction=nonstopmode "$temp_tex_file" 2>&1); then
        local tex_basename=$(basename "$temp_tex_file" .tex)
        if [ -f "${current_dir}/${tex_basename}.pdf" ]; then
            mv "${current_dir}/${tex_basename}.pdf" "$pdf_basename"
            rm -f "${current_dir}/${tex_basename}.aux" "${current_dir}/${tex_basename}.log"
        fi
        echo "✅ Successfully converted using xelatex"
        cleanup_all
        return 0
    else
        echo "$pdflatex_output"
        echo "❌ Failed to convert: $md_file"
        cleanup_all
        return 1
    fi
}

# If a part number is provided as argument
if [ $# -gt 0 ]; then
    PART_ARG="$1"
    
    # Check if it's a simple number (e.g., "1", "2", "3", "4")
    if [[ "$PART_ARG" =~ ^[1-4]$ ]]; then
        # Find part file
        FOUND_FILE=$(find . -maxdepth 2 -name "part${PART_ARG}.md" -type f | head -1)
        if [ -n "$FOUND_FILE" ]; then
            convert_part_to_pdf "$FOUND_FILE"
        else
            echo "❌ Error: Could not find part${PART_ARG}.md"
            echo "   Available parts:"
            find . -maxdepth 2 -name "part*.md" -type f | sed 's|^\./||' | sort
            exit 1
        fi
    # Check if it's "part1", "part2", etc.
    elif [[ "$PART_ARG" =~ ^part[1-4]$ ]]; then
        FOUND_FILE=$(find . -maxdepth 2 -name "${PART_ARG}.md" -type f | head -1)
        if [ -n "$FOUND_FILE" ]; then
            convert_part_to_pdf "$FOUND_FILE"
        else
            echo "❌ Error: Could not find ${PART_ARG}.md"
            exit 1
        fi
    # Check if it's a direct file path
    elif [ -f "$PART_ARG" ]; then
        convert_part_to_pdf "$PART_ARG"
    else
        echo "❌ Error: Could not find part matching '$PART_ARG'"
        echo "   Available parts:"
        find . -maxdepth 2 -name "part*.md" -type f | sed 's|^\./||' | sort
        exit 1
    fi
else
    # No argument provided, convert all part*.md files
    echo "🔄 Converting all part*.md files to PDF..."
    echo ""
    
    SUCCESS=0
    FAILED=0
    
    # Find all part files and convert them
    while IFS= read -r -d '' part_file; do
        if convert_part_to_pdf "$part_file"; then
            ((SUCCESS++))
        else
            ((FAILED++))
        fi
        echo ""
    done < <(find . -maxdepth 2 -name "part*.md" -type f -print0 | sort -z)
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 Summary:"
    echo "   ✅ Successful: $SUCCESS"
    echo "   ❌ Failed: $FAILED"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
fi
