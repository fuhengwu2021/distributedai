#!/bin/bash

# Script to convert markdown files to PDF using pandoc
# Usage:
#   ./convert_to_pdf.sh                    # Convert all chapter chapterX.md files
#   ./convert_to_pdf.sh 1                  # Convert chapter 1
#   ./convert_to_pdf.sh 2                  # Convert chapter 2
#   ./convert_to_pdf.sh chapter1           # Convert specific chapter (alternative)
#   ./convert_to_pdf.sh chapter1-introduction-to-modern-distributed-ai  # Full chapter name
#
# Footnote Support:
#   To add footnotes in markdown, use the following syntax:
#   
#   In the text: [^1] or [^note-label]
#   At the end of the document or section: [^1]: Footnote text here
#   
#   Example:
#     This is some text with a footnote[^1].
#     More text with another footnote[^2].
#     
#     [^1]: This is the first footnote.
#     [^2]: This is the second footnote.
#
#   Footnotes will automatically appear at the bottom of each page in the PDF.

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
    
    # Auto-generate LaTeX chapter title page and quote from Markdown
    # Create a temporary file with LaTeX code prepended if chapter title exists
    local temp_md_file=""
    local original_md_basename="$md_basename"
    
    # Check if file has chapter title pattern: # Chapter N: Title
    if grep -q "^# Chapter [0-9]\+:" "$md_basename"; then
        echo "   🔧 Auto-generating LaTeX chapter title page and quote from Markdown..."
        
        # Use Python script to process the entire file
        if [ -f "$SCRIPT_DIR/scripts/extract_chapter_title_quote.py" ]; then
            temp_md_file="${md_basename}.tmp.$$"
            if python3 "$SCRIPT_DIR/scripts/extract_chapter_title_quote.py" --process-file "$original_md_basename" "$temp_md_file" 2>/dev/null; then
                if [ -f "$temp_md_file" ] && [ -s "$temp_md_file" ]; then
                    md_basename="$temp_md_file"
                else
                    # Fallback: use original file if processing failed
                    rm -f "$temp_md_file"
                    temp_md_file=""
                fi
            else
                # Fallback: use original file if processing failed
                rm -f "$temp_md_file"
                temp_md_file=""
            fi
        fi
    fi
    
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
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{calc}
\usepackage{tcolorbox}
\usepackage{xparse}
% Control image scaling: keep images at original size unless they exceed page width
% This prevents small images from being over-scaled and becoming blurry
\makeatletter
% Only scale down if image is larger than linewidth, otherwise keep original size
\def\maxwidth{\ifdim\Gin@nat@width>\linewidth\linewidth\else\Gin@nat@width\fi}
\def\maxheight{\ifdim\Gin@nat@height>\textheight\textheight\else\Gin@nat@height\fi}
\makeatother
\setkeys{Gin}{width=\maxwidth,height=\maxheight,keepaspectratio}
% Footnote support: ensure footnotes are properly displayed
\usepackage{footnote}
% Customize footnote style and spacing
% Increase space between main text and footnotes
\setlength{\footnotesep}{0.5cm}
% Add space above the footnote rule
\renewcommand{\footnoterule}{\vspace*{8pt}\hrule width 0.4\columnwidth height 0.4pt \vspace*{4pt}}
% Increase space between footnote rule and first footnote
\setlength{\skip\footins}{1.2cm}
% Chapter title page styling
\definecolor{chapterblue}{RGB}{0,102,204}
\definecolor{chapterbluelight}{RGB}{153,204,255}
\definecolor{chaptergray}{RGB}{128,128,128}
\NewDocumentEnvironment{chaptertitlepage}{m m m O{} O{}}{%
  \newpage
  \thispagestyle{empty}
  \vspace*{-2cm}
  \begin{tikzpicture}[remember picture,overlay]
    % Blue square on the top right (4cm x 4cm)
    \coordinate (square-center) at ($(current page.north east) + (-2cm,-2cm)$);
    \fill[chapterblue] ($(square-center) + (-2cm,-2cm)$) rectangle 
      ($(square-center) + (2cm,2cm)$);
    % Chapter number in blue square (centered, smaller font to fit)
    \node[white,font=\fontsize{60}{72}\selectfont\bfseries,anchor=center] 
      at (square-center) {#1};
  \end{tikzpicture}
  \begin{minipage}{1.0\textwidth}
    \vspace{2cm}
    {\color{chaptergray}\large\bfseries Chapter #1}\\[0.4cm]
    {\color{chapterblue}\fontsize{24}{32}\selectfont\bfseries\raggedright #2}\\[0.4cm]
    {\itshape\normalsize #3}\\[0.8cm]
    % Compact quote section (if provided as 4th argument)
    \ifx\relax#4\relax\else
      \vspace{0.2cm}
      \noindent
      \begin{tikzpicture}
        % Decorative horizontal line spanning most of the page width
        \draw[chapterbluelight,line width=0.8pt] (0,0) -- (0.95\textwidth,0);
        % Speech bubble circle at the end
        \fill[chapterbluelight!20] (0.95\textwidth,0) circle (0.12);
        \draw[chapterbluelight,line width=0.8pt] (0.95\textwidth,0) circle (0.12);
      \end{tikzpicture}
      \par\vspace{0.1cm}
      \noindent
      \begin{minipage}[t]{0.95\textwidth}
        \raggedright
        \itshape
        \fontsize{12}{14}\selectfont
        \color{black}
        #4
      \end{minipage}
    \fi
    % Code summary section (if provided as 5th argument)
    \ifx\relax#5\relax\else
      \\[0.6cm]
      \noindent
      \begin{tikzpicture}
        % Decorative horizontal line spanning most of the page width
        \draw[chapterbluelight,line width=0.8pt] (0,0) -- (0.95\textwidth,0);
        % Speech bubble circle at the end
        \fill[chapterbluelight!20] (0.95\textwidth,0) circle (0.12);
        \draw[chapterbluelight,line width=0.8pt] (0.95\textwidth,0) circle (0.12);
      \end{tikzpicture}
      \\[0.3cm]
      \begin{itemize}
      \setlength{\itemsep}{0.4cm}
      \normalsize
      \color{chapterblue}
      #5
      \end{itemize}
    \fi
  \end{minipage}
  \vfill
  \newpage
}{}
EOF
)
    # Cleanup function (defined before use)
    cleanup_temp() {
        if [ -n "$temp_md_file" ] && [ -f "$temp_md_file" ]; then
            rm -f "$temp_md_file"
        fi
    }
    
    if pandoc_output=$(pandoc "$md_basename" -o "$pdf_basename" --pdf-engine=xelatex -V geometry:margin=1in --highlight-style=tango -H <(echo "$latex_header") 2>&1); then
        # Filter out font-related warnings but keep image warnings
        echo "$pandoc_output" | grep -E "\[WARNING\].*image|\[WARNING\].*resource" || true
        echo "✅ Successfully converted using xelatex"
        cleanup_temp
        return 0
    elif pandoc_output=$(pandoc "$md_basename" -o "$pdf_basename" --pdf-engine=pdflatex -V geometry:margin=1in --highlight-style=tango -V 'tolerance=1000' -V 'emergencystretch=3em' -H <(echo "$latex_header") 2>&1); then
        echo "$pandoc_output" | grep -E "\[WARNING\].*image|\[WARNING\].*resource" || true
        echo "✅ Successfully converted using pdflatex"
        cleanup_temp
        return 0
    elif pandoc_output=$(pandoc "$md_basename" -o "$pdf_basename" -V geometry:margin=1in --highlight-style=tango -V 'tolerance=1000' -V 'emergencystretch=3em' -H <(echo "$latex_header") 2>&1); then
        echo "$pandoc_output" | grep -E "\[WARNING\].*image|\[WARNING\].*resource" || true
        echo "✅ Successfully converted using default engine"
        cleanup_temp
        return 0
    else
        # Show all output if conversion failed
        echo "$pandoc_output"
        echo "❌ Failed to convert: $md_file"
        cleanup_temp
        return 1
    fi
}

# If a chapter name is provided as argument
if [ $# -gt 0 ]; then
    CHAPTER_ARG="$1"
    
    # Check if it's the appendix file
    if [[ "$CHAPTER_ARG" == "chapterx/chapterx.md" ]] || [[ "$CHAPTER_ARG" == "chapterx" ]]; then
        if [ -f "chapterx/chapterx.md" ]; then
            convert_md_to_pdf "chapterx/chapterx.md"
        else
            echo "❌ Error: Appendix file not found: chapterx/chapterx.md"
            exit 1
        fi
    # Check if argument is a simple number (e.g., "1", "2", "10")
    elif [[ "$CHAPTER_ARG" =~ ^[0-9]+$ ]]; then
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
    # No argument provided, convert all chapter chapterX.md files and appendix
    echo "🔄 Converting all chapter chapterX.md files and appendix to PDF..."
    echo ""
    
    SUCCESS=0
    FAILED=0
    
    # Convert appendix if it exists
    if [ -f "chapterx/chapterx.md" ]; then
        if convert_md_to_pdf "chapterx/chapterx.md"; then
            ((SUCCESS++))
        else
            ((FAILED++))
        fi
        echo ""
    fi
    
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
