#!/bin/bash

# Script to convert markdown files to PDF using pandoc
# Usage:
#   ./convert_to_pdf.sh                    # Convert all chapter chapterX.md files
#   ./convert_to_pdf.sh 1                  # Convert chapter 1
#   ./convert_to_pdf.sh 2                  # Convert chapter 2
#   ./convert_to_pdf.sh chapter1           # Convert specific chapter (alternative)
#   ./convert_to_pdf.sh chapter1-introduction-to-modern-distributed-ai  # Full chapter name
#
# Chapter Number Style Options:
#   --style circle  or  -s circle          # Quarter circle style (default)
#   --style square  or  -s square          # Square style
#   Example: ./convert_to_pdf.sh 1 --style square
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
#
# Fancy Divider Support:
#   To add a fancy horizontal divider in markdown, use the following syntax:
#   
#   Without icon:
#     \fancydivider                                    # Default blue divider (95% width)
#     \fancydivider[dividerred]                        # Red divider with default width
#     \fancydivider[chapterblue][0.8\textwidth]       # Blue divider with custom width
#   
#   With icon:
#     \fancydividerwithicon{icon.png}                  # With icon (default color/width)
#     \fancydividerwithicon[dividerred]{python-logo.png}  # Red divider with icon
#     \fancydividerwithicon[chapterblue][0.8\textwidth]{icon.svg}  # Full customization
#   
#   Available colors: chapterbluelight (default), chapterblue, dividerred, red, blue, black, etc.
#   Width can be specified as: 0.95\textwidth (default), 0.8\textwidth, \linewidth, etc.
#   Icon formats: PNG, SVG, PDF, JPG (any format supported by LaTeX graphicx package)
#   Icon will be placed on the right side of the line, overlapping the line
#   Icon path should be relative to the markdown file location (same as regular images)
#   
#   Example:
#     Some text here.
#     
#     \fancydivider
#     
#     More text after the divider.
#     
#     \fancydividerwithicon[dividerred]{python-logo.png}
#     
#     Text after red divider with Python icon.
#
# Code Line Annotations Support:
#   Code blocks automatically display circled line numbers (①, ②, ③, etc.) at the
#   end of each line. No manual annotation needed!
#   
#   Simply write your code block normally:
#      ```python
#      print("Hello")
#      print("World")
#      ```
#   
#   Add explanations after the code block:
#      ```python
#      print("Hello")
#      print("World")
#      ```
#      
#      \begin{codeexplanation}
#      \codelineannotation{1}{First line prints Hello}
#      \codelineannotation{2}{Second line prints World}
#      \end{codeexplanation}
#   
#   The circled numbers (①, ②, etc.) will automatically appear at the end of each
#   line inside the code block, and explanations will appear below the code block
#   in italic gray text with proper line breaks.
#
# Image Attributes Support:
#   Images can be controlled with attributes for size, alignment, and placement.
#   Images without attributes work normally (default Pandoc behavior).
#   
#   IMPORTANT: Use class syntax (with dot) for placement modes!
#   
#   Basic size control (applies to any placement mode):
#     ![alt text](image.png){width=50%}
#     ![alt text](image.png){width=5cm}
#     ![alt text](image.png){width=0.8\textwidth}
#     ![alt text](image.png){height=3in}
#     ![alt text](image.png){width=50% height=4cm}
#   
#   Block placement (full-width block, occupies entire row, with optional caption):
#     ![alt text](image.png){.block}
#     ![alt text](image.png){.block width=80% align=center}
#     ![alt text](image.png){.block width=60% align=left}
#     ![alt text](image.png){.block width=70% align=right}
#     Note: Block images are placed in figure environment, alt text becomes caption
#   
#   Inline placement (flows with text, wraps by text):
#     ![alt text](image.png){.inline width=30%}
#     ![alt text](image.png){.inline width=25% align=left}
#     ![alt text](image.png){.inline width=25% align=right}
#     Note: Inline images appear within the text flow
#   
#   Text wrapping (text wraps around image):
#     ![alt text](image.png){.wrap width=40% align=right}
#     ![alt text](image.png){.wrap width=35% align=left}
#     Note: Wrap images allow text to flow around them
#   
#   Default behavior: Images without .block/.inline/.wrap classes use default
#   Pandoc behavior (typically block placement with automatic width scaling).

set -e  # Exit on error

# Global cleanup function to be called on script exit (including errors)
cleanup_on_exit() {
    # Clean up any files containing tmp in their name immediately
    # Search in chapter directories (maxdepth 2) and root (maxdepth 1)
    find . -maxdepth 2 -type f \( -name "*tmp*" -o -name "*.tmp*" \) -delete 2>/dev/null || true
    # Also try with rm as fallback for nested temp files
    find . -maxdepth 3 -type f \( -name "*tmp*" -o -name "*.tmp*" \) -exec rm -f {} \; 2>/dev/null || true
}

# Register cleanup function to run on script exit
trap cleanup_on_exit EXIT

# Get the script directory (root of the project)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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

# Default chapter number style
CHAPTER_STYLE="circle"
# Default include config file
INCLUDE_CONFIG=""

# Parse command line arguments
ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --style|-s)
            CHAPTER_STYLE="$2"
            if [[ "$CHAPTER_STYLE" != "circle" && "$CHAPTER_STYLE" != "square" ]]; then
                echo "❌ Error: Invalid style '$CHAPTER_STYLE'. Use 'circle' or 'square'."
                exit 1
            fi
            shift 2
            ;;
        --include-config|-c)
            INCLUDE_CONFIG="$2"
            shift 2
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

# Restore positional arguments
set -- "${ARGS[@]}"

echo $CHAPTER_STYLE
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
    
    # Process conditional includes first (before chapter title extraction)
    local temp_md_file=""
    local original_md_basename="$md_basename"
    local include_processed_file=""
    
    # Check if file contains include directives
    if grep -q "<!-- include:" "$md_basename"; then
        echo "   🔧 Processing conditional includes..."
        
        # Use Python script to process includes
        if [ -f "$SCRIPT_DIR/scripts/process_includes.py" ]; then
            include_processed_file="${md_basename}.tmp.includes"
            local config_args=()
            if [ -n "$INCLUDE_CONFIG" ]; then
                # Convert to absolute path if relative
                if [[ "$INCLUDE_CONFIG" != /* ]]; then
                    config_args=("--config" "$SCRIPT_DIR/$INCLUDE_CONFIG")
                else
                    config_args=("--config" "$INCLUDE_CONFIG")
                fi
            else
                # Use default config file if no custom config specified
                if [ -f "$SCRIPT_DIR/peanut.config" ]; then
                    config_args=("--config" "$SCRIPT_DIR/peanut.config")
                fi
            fi
            
            python_output=$(python3 "$SCRIPT_DIR/scripts/process_includes.py" --process-file "$original_md_basename" "$include_processed_file" "${config_args[@]}" 2>&1)
            python_exit_code=$?
            if [ -n "$python_output" ]; then
                echo "   $python_output" | sed 's/^/   /'
            fi
            if [ $python_exit_code -eq 0 ]; then
                if [ -f "$include_processed_file" ] && [ -s "$include_processed_file" ]; then
                    md_basename="$include_processed_file"
                    echo "   ✓ Includes processed: $include_processed_file"
                else
                    # Fallback: use original file if processing failed
                    echo "   ⚠️  Warning: Include processing failed, using original file"
                    rm -f "$include_processed_file"
                    include_processed_file=""
                fi
            else
                # Fallback: use original file if processing failed
                echo "   ⚠️  Warning: Include processing failed (exit code $python_exit_code), using original file"
                rm -f "$include_processed_file"
                include_processed_file=""
            fi
        else
            echo "   ⚠️  Warning: process_includes.py not found, skipping include processing"
        fi
    fi
    
    # Check if file contains CODE_EXPLAIN_START blocks
    # Check both the current md_basename and the original file to be safe
    if grep -q "CODE_EXPLAIN_START" "$md_basename" 2>/dev/null || grep -q "CODE_EXPLAIN_START" "$original_md_basename" 2>/dev/null; then
        echo "   🔧 Processing CODE_EXPLAIN blocks..."
        
        # Use Python script to process CODE_EXPLAIN blocks
        if [ -f "$SCRIPT_DIR/scripts/process_code_explain.py" ]; then
            code_explain_processed_file="${md_basename}.tmp.code_explain"
            
            python_output=$(python3 "$SCRIPT_DIR/scripts/process_code_explain.py" --process-file "$md_basename" "$code_explain_processed_file" 2>&1)
            python_exit_code=$?
            if [ -n "$python_output" ]; then
                echo "   $python_output" | sed 's/^/   /'
            fi
            if [ $python_exit_code -eq 0 ]; then
                if [ -f "$code_explain_processed_file" ] && [ -s "$code_explain_processed_file" ]; then
                    # Update md_basename to use the processed file
                    # If we already have an include_processed_file, we need to chain the processing
                    if [ -n "$include_processed_file" ] && [ -f "$include_processed_file" ]; then
                        # Remove the intermediate include file since we're replacing it
                        rm -f "$include_processed_file"
                    fi
                    md_basename="$code_explain_processed_file"
                    include_processed_file="$code_explain_processed_file"
                    echo "   ✓ CODE_EXPLAIN blocks processed: $code_explain_processed_file"
                else
                    # Fallback: use original file if processing failed
                    echo "   ⚠️  Warning: CODE_EXPLAIN processing failed, using original file"
                    rm -f "$code_explain_processed_file"
                fi
            else
                # Fallback: use original file if processing failed
                echo "   ⚠️  Warning: CODE_EXPLAIN processing failed (exit code $python_exit_code), using original file"
                rm -f "$code_explain_processed_file"
            fi
        else
            echo "   ⚠️  Warning: process_code_explain.py not found, skipping CODE_EXPLAIN processing"
        fi
    fi
    
    # Auto-generate LaTeX chapter title page and quote from Markdown
    # Create a temporary file with LaTeX code prepended if chapter title exists
    # Use the include-processed file if available, otherwise original
    local file_for_title_extraction="$md_basename"
    
    # Check if file has chapter title pattern: # Chapter N: Title
    if grep -q "^# Chapter [0-9]\+:" "$file_for_title_extraction"; then
        echo "   🔧 Auto-generating LaTeX chapter title page and quote from Markdown..."
        
        # Use Python script to process the entire file
        if [ -f "$SCRIPT_DIR/scripts/extract_chapter_title_quote.py" ]; then
            temp_md_file="${md_basename}.tmp"
            # Use the include-processed file if available, otherwise original
            local input_for_title="$md_basename"
            # Since we're already in the chapter directory, use relative paths
            # Debug: show what we're calling
            python_output=$(python3 "$SCRIPT_DIR/scripts/extract_chapter_title_quote.py" --process-file "$input_for_title" "$temp_md_file" 2>&1)
            python_exit_code=$?
            if [ -n "$python_output" ]; then
                echo "   Python script output: $python_output"
            fi
            if [ $python_exit_code -eq 0 ]; then
                if [ -f "$temp_md_file" ] && [ -s "$temp_md_file" ]; then
                    md_basename="$temp_md_file"
                    echo "   ✓ Temp file created: $temp_md_file"
                else
                    # Fallback: use original file if processing failed
                    echo "   ⚠️  Warning: Temp file not created (exit code was 0 but file missing), using original file"
                    echo "   Debug: Current dir=$(pwd), temp_file=$temp_md_file, exists=$([ -f "$temp_md_file" ] && echo yes || echo no)"
                    rm -f "$temp_md_file"
                    temp_md_file=""
                fi
            else
                # Fallback: use original file if processing failed
                echo "   ⚠️  Warning: Python script failed (exit code $python_exit_code), using original file"
                rm -f "$temp_md_file"
                temp_md_file=""
            fi
        fi
    fi
    
    # Extract chapter number and title from markdown file for header
    local chapter_number=""
    local chapter_title=""
    local header_title_text="Modern Distributed AI Systems"
    if grep -q "^# Chapter [0-9]\+:" "$original_md_basename"; then
        # Extract chapter number: "Chapter N: Title" -> "N"
        chapter_number=$(grep "^# Chapter [0-9]\+:" "$original_md_basename" | sed 's/^# Chapter \([0-9]\+\).*/\1/' | head -1)
        # Extract chapter title: "Chapter N: Title" -> "Title"
        chapter_title=$(grep "^# Chapter [0-9]\+:" "$original_md_basename" | sed 's/^# Chapter [0-9]\+: *//' | head -1)
        if [ -n "$chapter_title" ] && [ -n "$chapter_number" ]; then
            # Remove Pandoc attributes like {-} or {.unnumbered} from title
            chapter_title=$(echo "$chapter_title" | sed 's/\s*{[^}]*}\s*$//')
            # Escape LaTeX special characters in chapter title
            local escaped_title=$(echo "$chapter_title" | sed 's/\\/\\textbackslash{}/g; s/{/\\{/g; s/}/\\}/g; s/\$/\\$/g; s/&/\\&/g; s/%/\\%/g; s/#/\\#/g; s/\^/\\textasciicircum{}/g; s/_/\\_/g')
            # Format as "Chapter X - Title"
            header_title_text="Chapter $chapter_number - $escaped_title"
        fi
    fi
    
    # Try different PDF engines in order of preference
    # Use basenames since we're now in the chapter directory
    # Capture output to filter warnings but show errors
    local pandoc_output=""
    # Create LaTeX header with image size control
    # Keep images at original size to prevent blurriness from over-scaling
    # Only scale down if image exceeds page width
    # Use Jinja2 template to generate LaTeX header
    # Get template name from peanut.config (defaults to default.tpl)
    local template_name=$(get_template_name)
    # Use chapter template for single chapters
    local template_file="$SCRIPT_DIR/templates/chapter/$template_name"
    if [ ! -f "$template_file" ]; then
        echo "❌ Error: Template file not found: $template_file"
        echo "   Available templates: $(ls -1 "$SCRIPT_DIR/templates"/*.tpl 2>/dev/null | xargs -n1 basename | tr '\n' ' ' || echo 'none')"
        return 1
    fi
    
    # Generate LaTeX header using template file with Python script
    local python_script="$SCRIPT_DIR/scripts/generate_latex_header.py"
    if [ ! -f "$python_script" ]; then
        echo "❌ Error: Python script not found: $python_script"
        return 1
    fi
    
    local latex_header=$(python3 "$python_script" "$template_file" "$header_title_text" "$CHAPTER_STYLE" "$chapter_number" 2>&1)
    local python_exit_code=$?
    
    if [ $python_exit_code -ne 0 ] || [ -z "$latex_header" ]; then
        echo "⚠️  Warning: Python script failed, using fallback method"
        latex_header=""
    fi
    
    # Fallback: Use quoted heredoc if Python/Jinja2 is not available
    if [ $? -ne 0 ] || [ -z "$latex_header" ]; then
        echo "⚠️  Warning: Jinja2 template rendering failed, using fallback method"
        local latex_header=$(cat <<'STATIC_EOF'
% Unicode support - use fontspec for xelatex, inputenc for pdflatex
\ifxetex
  \usepackage{fontspec}
  % Let xelatex use default system fonts (will handle Unicode properly)
\else
  \usepackage[utf8]{inputenc}
  \usepackage[T1]{fontenc}
  \usepackage{textcomp}
\fi
\usepackage{microtype}
\sloppy
\setlength{\emergencystretch}{3em}
\tolerance=1000
\allowdisplaybreaks
\usepackage{amsmath}  % Required for \eqref command
\usepackage{float}
\floatplacement{figure}{H}
\usepackage{graphicx}
\usepackage{wrapfig}  % For text wrapping around images
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{calc}
% Use tcolorbox to wrap listings with rounded corners (better than mdframed)
% tcolorbox provides reliable rounded corners support
% Load with [most] option to get all libraries including listings
\usepackage[most]{tcolorbox}
\tcbuselibrary{listings}
\usepackage{xparse}
\usepackage{enumitem}
\usepackage{etoolbox}
% Define colors before using them in listings
\definecolor{chapterblue}{RGB}{0,102,204}
\definecolor{chapterbluelight}{RGB}{153,204,255}
\definecolor{chaptergray}{RGB}{128,128,128}
\definecolor{dividerred}{RGB}{255,0,0}
% Code block styling with line number annotations using listings package
\usepackage{listings}
% Use soul package for text highlighting
\usepackage{soul}
% Use xstring for string comparison in chapter title page style
\usepackage{xstring}
% Enhanced syntax highlighting colors
\definecolor{codekeyword}{RGB}{0,102,204}
\definecolor{codecomment}{RGB}{128,128,128}
\definecolor{codestring}{RGB}{0,128,0}
\definecolor{codenumber}{RGB}{128,0,128}
\definecolor{codefunction}{RGB}{0,0,255}
% Define custom style for circled line numbers using simpler approach
% We'll add line numbers manually using a post-processing approach
\lstset{
  basicstyle=\ttfamily\normalsize,
  breaklines=true,
  breakatwhitespace=true,
  showstringspaces=false,
  frame=none,
  backgroundcolor=\color{gray!5},
  % Enhanced syntax highlighting
  commentstyle=\color{codecomment}\itshape,
  keywordstyle=\color{codekeyword}\bfseries,
  stringstyle=\color{codestring},
  numberstyle=\color{codenumber},
  identifierstyle=\color{black},
  % Python-specific highlighting
  emph={True,False,None,self},
  emphstyle=\color{codekeyword}\bfseries,
  % Function names
  morekeywords={print,import,from,def,class,if,else,elif,for,while,return,try,except,finally,with,as,pass,break,continue,lambda,del,global,nonlocal,yield,assert,raise},
  % Better string handling
  string=[s]{"}{"},
  string=[s]{'}{'},
  % Better comment handling
  comment=[l]{\#},
  % Ensure symmetric padding inside mdframed
  xleftmargin=0pt,
  xrightmargin=0pt,
  aboveskip=0pt,
  belowskip=0pt,
  % Remove extra spacing that might cause asymmetry
  lineskip=0pt,
  % Ensure consistent line height
  basewidth=0.5em,
}
% Bash-specific highlighting style
\lstdefinestyle{bashstyle}{%
  language=bash,
  basicstyle=\ttfamily\normalsize,
  breaklines=true,
  breakatwhitespace=true,
  showstringspaces=false,
  frame=none,
  backgroundcolor=\color{gray!5},
  keywordstyle=\color{codekeyword}\bfseries,
  commentstyle=\color{codecomment}\itshape,
  stringstyle=\color{codestring},
  identifierstyle=\color{black},
  numberstyle=\color{codenumber},
  % Bash keywords
  morekeywords={if,then,else,elif,fi,for,while,do,done,case,esac,function,export,local,readonly,declare,typeset},
  % Bash built-in commands and ML/AI commands
  morekeywords=[2]{echo,cd,pwd,ls,cat,grep,sed,awk,find,chmod,chown,cp,mv,rm,mkdir,rmdir,touch,ln,ps,kill,env,source,exec,eval,test,torchrun,python,python3,srun,mpirun,horovodrun},
  keywordstyle=[2]=\color{codefunction}\bfseries,
  % Comments
  comment=[l]{\#},
  % Strings
  string=[b]{"},
  string=[b]{'},
  % Highlight numbers
  literate={-}{{\color{codekeyword}\bfseries-}}1
           {--}{{\color{codekeyword}\bfseries--}}2
           {0}{{\color{codenumber}0}}1
           {1}{{\color{codenumber}1}}1
           {2}{{\color{codenumber}2}}1
           {3}{{\color{codenumber}3}}1
           {4}{{\color{codenumber}4}}1
           {5}{{\color{codenumber}5}}1
           {6}{{\color{codenumber}6}}1
           {7}{{\color{codenumber}7}}1
           {8}{{\color{codenumber}8}}1
           {9}{{\color{codenumber}9}}1,
  xleftmargin=0pt,
  xrightmargin=0pt,
  aboveskip=0pt,
  belowskip=0pt,
  numbers=none,
}
% Define tcolorbox style for code blocks with border and rounded corners
% tcolorbox provides reliable rounded corners support
\tcbset{
  codeblockstyle/.style={
    colback=gray!5,
    colframe=chapterbluelight,
    boxrule=0.8pt,
    arc=4pt,
    left=5pt,
    right=10pt,
    top=5pt,
    bottom=5pt,
    before skip=0.5em,
    after skip=0.5em,
    fontupper=\ttfamily\normalsize,
  }
}
% Command to add circled number mark (for use in explanations outside code blocks)
% First argument: style ("normal" or "solid")
% Second argument: line number
% Use lighter colors and position closer to left border
\newcommand{\codelinemark}[2]{%
  \ifstrequal{#1}{solid}{%
    % Solid fill for highlighted lines (dark blue)
    \tikz[baseline=(char.base)]{%
      \node[shape=circle,draw=chapterblue,fill=chapterblue,inner sep=2pt,minimum size=1.2em,font=\scriptsize\bfseries\color{white}] (char) {#2};%
    }%
  }{%
    % Normal style (lighter colors)
    \tikz[baseline=(char.base)]{%
      \node[shape=circle,draw=chapterbluelight,fill=chapterbluelight!20,inner sep=2pt,minimum size=1.2em,font=\scriptsize\bfseries\color{chapterbluelight}] (char) {#2};%
    }%
  }%
}
% Command for code line explanation (used outside code block)
% Use dark blue for consistency with highlighted lines
% Ensure perfect baseline alignment between circle and text
% Use moderate adjustment to find middle ground
\newcommand{\codelineannotation}[2]{%
  \tikz[baseline=-0.35ex]{%
    \node[shape=circle,draw=chapterblue,fill=chapterblue,inner sep=2pt,minimum size=1.2em,font=\scriptsize\bfseries\color{white}] (char) {#1};%
  }%
  \quad\raisebox{-0.35ex}{\color{black}#2}\par%
}
% Environment for code explanations (to be used after code blocks)
\newenvironment{codeexplanation}{%
  \vspace{0.3cm}%
  \noindent%
  \begin{minipage}{\textwidth}%
  \small%
  \color{black}%
  \raggedright%
}{%
  \end{minipage}%
  \vspace{0.3cm}%
}
% Environment for note sections (marked with >NOTES: and >NOTEE:)
\newenvironment{notesection}{%
  \vspace{0.5em}%
  \noindent%
  \begin{tcolorbox}[
    colback=yellow!10,
    colframe=chapterbluelight,
    boxrule=0.8pt,
    arc=4pt,
    left=3pt,
    right=10pt,
    top=8pt,
    bottom=8pt,
    before skip=0pt,
    after skip=0pt
  ]%
  \raisebox{-\height}[0pt][0pt]{%
    \begin{minipage}[t]{0.10\textwidth}%
      \raggedright%
      \includegraphics[width=0.3\linewidth,keepaspectratio]{../img/note32x32.png}%
    \end{minipage}%
  }%
  \hspace{0.00em}%
  \begin{minipage}[t]{0.9\textwidth}%
  \small%
  \color{black}%
  \raggedright%
  \setlength{\leftskip}{-15pt}%
  \setlength{\parindent}{0pt}%
}{%
  \end{minipage}%
  \end{tcolorbox}%
  \vspace{0.5em}%
}
% PDF bookmarks/outline for navigation
\usepackage{hyperref}
\hypersetup{
    colorlinks=true,
    linkcolor=chapterblue,
    urlcolor=chapterblue,
    citecolor=chapterblue,
    bookmarks=true,
    bookmarksopen=true,
    bookmarksopenlevel=2,
    pdfstartview=FitH
}
% Page headers using fancyhdr
\usepackage{fancyhdr}
\pagestyle{fancy}
% Clear default header/footer
\fancyhf{}
% Set header: left side shows chapter number and title (or book title if no chapter), right side shows page number
% Reduce vertical spacing in header by using raisebox to move text down closer to rule
\fancyhead[L]{\raisebox{-6pt}{\color{chaptergray}\small\itshape HEADER_TITLE_PLACEHOLDER}}
\fancyhead[R]{\raisebox{-6pt}{\color{chaptergray}\small\thepage}}
% Set footer: right side shows page number
\fancyfoot[R]{\color{chaptergray}\small\thepage}
% Add a line under the header
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\headrule}{\hbox to\headwidth{\color{chapterbluelight}\leaders\hrule height \headrulewidth\hfill}}
% Add a line above the footer
\renewcommand{\footrulewidth}{0.4pt}
\renewcommand{\footrule}{\hbox to\headwidth{\color{chapterbluelight}\leaders\hrule height \footrulewidth\hfill}}
% Set header and footer height to accommodate the content
% Reduce headheight to bring header text closer to the horizontal line
\setlength{\headheight}{10pt}
\setlength{\footskip}{20pt}
% Control image scaling: keep images at original size unless they exceed page width
% This prevents small images from being over-scaled and becoming blurry
\makeatletter
% Only scale down if image is larger than linewidth, otherwise keep original size
\def\maxwidth{\ifdim\Gin@nat@width>\linewidth\linewidth\else\Gin@nat@width\fi}
\def\maxheight{\ifdim\Gin@nat@height>\textheight\textheight\else\Gin@nat@height\fi}
\makeatother
\setkeys{Gin}{width=\maxwidth,height=\maxheight,keepaspectratio}
% Footnote support: ensure footnotes are properly displayed
% Note: Pandoc 3.8 handles footnotes natively, so \usepackage{footnote} is not needed
% Customize footnote style and spacing
% Increase space between main text and footnotes
\setlength{\footnotesep}{0.5cm}
% Add space above the footnote rule
\renewcommand{\footnoterule}{\vspace*{8pt}\hrule width 0.4\columnwidth height 0.4pt \vspace*{4pt}}
% Increase space between footnote rule and first footnote
\setlength{\skip\footins}{1.2cm}
% Modify subsection numbering to show only subsection number (not section.subsection)
% This makes the first ## heading show as "1" instead of "0.1"
\renewcommand{\thesubsection}{\arabic{subsection}}
% Modify equation numbering to include chapter number (e.g., (15.1) instead of (1))
% Define chapter number variable (will be set from markdown if available)
\newcommand{\mychapternum}{}
% If chapter number is provided, use it; otherwise use subsection number
\ifx\mychapternum\empty
  \renewcommand{\theequation}{\thesubsection.\arabic{equation}}
\else
  \renewcommand{\theequation}{\mychapternum.\arabic{equation}}
\fi
% Chapter title page styling
% (Colors are already defined above, before listings package)
% Fancy divider command for use in markdown
% Usage in markdown: 
%   \fancydivider                                    # Default blue divider (no icon)
%   \fancydivider[color]                            # Custom color, no icon
%   \fancydivider[color][width]                      # Custom color and width, no icon
%   \fancydividerwithicon{icon.png}                  # With icon (default color/width)
%   \fancydividerwithicon[color]{icon.png}           # With icon and custom color
%   \fancydividerwithicon[color][width]{icon.png}    # With icon, color, and width
% Example: 
%   \fancydivider
%   \fancydivider[dividerred]
%   \fancydividerwithicon{python-logo.png}
%   \fancydividerwithicon[dividerred]{python-logo.svg}
%   \fancydividerwithicon[chapterblue][0.8\textwidth]{python-logo.pdf}
% Available colors: chapterbluelight (default), chapterblue, dividerred, red, blue, black, etc.
% Icon formats: PNG, SVG, PDF, JPG (any format supported by graphicx package)
% Icon will be placed on the right side of the line, overlapping the line
% Divider without icon
\NewDocumentCommand{\fancydivider}{O{chapterbluelight} O{0.95\textwidth}}{%
  \par\vspace{0.8cm}%
  \noindent%
  \begin{tikzpicture}%
    \coordinate (line-end) at (#2,0);%
    \draw[#1,line width=0.8pt] (0,0) -- (line-end);%
    \fill[#1!20] (line-end) circle (0.12);%
    \draw[#1,line width=0.8pt] (line-end) circle (0.12);%
  \end{tikzpicture}%
  \par\vspace{0.8cm}%
}
% Divider with icon
% Usage: \fancydividerwithicon[position]{image}
%   position: end (default), begin, center
%   image: automatically uses '../img/' prefix
%   Note: color and width are fixed (chapterbluelight, 0.95\textwidth)
\NewDocumentCommand{\fancydividerwithicon}{O{end} m}{%
  \par\vspace{0.5\baselineskip}%
  \noindent%
  \begin{tikzpicture}[baseline=(current bounding box.south)]%
    \coordinate (lineend) at (0.95\textwidth,0);%
    \ifstrequal{#1}{begin}{%
      % Icon at beginning
      \node[anchor=center,inner sep=0] at (0,0) {%
        \includegraphics[height=0.7cm,keepaspectratio]{../img/#2}%
      };%
      \draw[chapterbluelight,line width=0.8pt] (0.4cm,0) -- (lineend,0);%
    }{%
      \ifstrequal{#1}{center}{%
        % Icon at center - use calc library for coordinate calculation
        \coordinate (linecenter) at ($(0,0)!0.5!(lineend)$);%
        \draw[chapterbluelight,line width=0.8pt] (0,0) -- ($(linecenter) + (-0.4cm,0)$);%
        \draw[chapterbluelight,line width=0.8pt] ($(linecenter) + (0.4cm,0)$) -- (lineend);%
        \node[anchor=center,inner sep=0] at (linecenter) {%
          \includegraphics[height=0.7cm,keepaspectratio]{../img/#2}%
        };%
      }{%
        % Icon at end (default)
        \draw[chapterbluelight,line width=0.8pt] (0,0) -- (lineend,0);%
        \node[anchor=center,inner sep=0] at (lineend) {%
          \includegraphics[height=0.7cm,keepaspectratio]{../img/#2}%
        };%
      }%
    }%
  \end{tikzpicture}%
  \par\vspace{-0.5\baselineskip}%
}
% Chapter number style - generated based on CHAPTER_STYLE variable
\NewDocumentEnvironment{chaptertitlepage}{m m m O{} O{}}{%
  \newpage
  \thispagestyle{empty}
  % Add PDF bookmark for chapter title (for navigation in PDF readers)
  \pdfbookmark[1]{Chapter #1: #2}{chapter#1}
  \vspace*{-2cm}
  \begin{tikzpicture}[remember picture,overlay]
    CHAPTER_NUMBER_STYLE_PLACEHOLDER
    % Chapter number (positioned in the visible area) - use the coordinate defined above
    \node[black,font=\fontsize{60}{72}\selectfont\bfseries,anchor=center] 
      at (chapter-pos) {#1};
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
      \centering
      \begin{itemize}[label=\raisebox{0.2ex}{\textcolor{chapterblue}{$\triangleright$}},leftmargin=*,itemsep=0.15cm,topsep=0pt,partopsep=0pt]
      \normalsize
      \color{chapterblue}
      #5
      \end{itemize}
      \raggedright
    \fi
  \end{minipage}
  \vfill
  \newpage
}{}
STATIC_EOF
)
        # Fallback: Replace placeholder with actual title (already formatted as "Chapter X - Title")
        local escaped_title=$(printf '%s\n' "$header_title_text" | sed 's/[[\.*^$()+?{|]/\\&/g' | sed 's/\\/\\\\/g')
        latex_header=$(echo "$latex_header" | sed "s|HEADER_TITLE_PLACEHOLDER|$escaped_title|g")
        
        # Fallback: Generate chapter number style code based on CHAPTER_STYLE using Python
        local chapter_style_code=$(python3 <<PYTHON_EOF
style = "$CHAPTER_STYLE"
if style == "square":
    print(r"""    % Square style
    \coordinate (square-center) at (\$(current page.north east) + (-2cm,-2cm)\$);
    \coordinate (chapter-pos) at (square-center);
    % Light blue background with darker blue border to make it visible
    \filldraw[fill=chapterbluelight!30,draw=chapterblue,line width=3pt] (\$(square-center) + (-2cm,-2cm)\$) rectangle 
      (\$(square-center) + (2cm,2cm)\$);""")
else:
    # Default: Circle style
    print(r"""    % Circle style (default) - quarter circle
    \coordinate (circle-center) at (current page.north east);
    \coordinate (chapter-pos) at (\$(circle-center) + (-1.5cm,-1.5cm)\$);
    % White background - quarter circle area (no border on top/right edges)
    \fill[white] (\$(circle-center) + (-4cm,0)\$) arc (180:270:4cm) -- (circle-center) -- cycle;
    % Blue border only on the curved arc (not on top/right edges)
    \draw[chapterblue,line width=2pt] (\$(circle-center) + (-4cm,0)\$) arc (180:270:4cm);""")
PYTHON_EOF
)
        # Fallback: Replace the placeholder with the generated code
        latex_header=$(echo "$latex_header" | python3 <<PYTHON_REPLACE
import sys
content = sys.stdin.read()
style_code = '''$chapter_style_code'''
# Replace the placeholder
content = content.replace('CHAPTER_NUMBER_STYLE_PLACEHOLDER', style_code)
print(content, end='')
PYTHON_REPLACE
)
    fi
    # Cleanup function (defined before use)
    cleanup_temp() {
        if [ -n "$temp_md_file" ] && [ -f "$temp_md_file" ]; then
            rm -f "$temp_md_file"
        fi
        if [ -n "$include_processed_file" ] && [ -f "$include_processed_file" ]; then
            rm -f "$include_processed_file"
        fi
    }
    
    # Use Lua filters for resetting section counter, code line numbers, and note sections
    local lua_filters=""
    # Reset section counter filter must run first to insert counter reset before first ## heading
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
    
    # Write header to temporary file to ensure proper handling
    # We're already in the chapter directory, so use relative path
    local header_file="${md_basename}.header.tex"
    echo "$latex_header" > "$header_file"
    # Verify file was created
    if [ ! -f "$header_file" ]; then
        echo "❌ Error: Failed to create header file: $header_file"
        cleanup_all
        return 1
    fi
    
    # Simplified cleanup function - remove everything with .tmp in the name
    cleanup_all() {
        # Get the directory from the original md_file if dir_name is not set
        local cleanup_dir="${dir_name:-$(dirname "$md_file")}"
        if [ -n "$cleanup_dir" ] && [ -d "$cleanup_dir" ]; then
            # Remove all files containing tmp in their name (more aggressive cleanup)
            # Use both find and rm for maximum coverage - check for both *tmp* and *.tmp* patterns
            find "$cleanup_dir" -maxdepth 2 -type f \( -name "*tmp*" -o -name "*.tmp*" \) -delete 2>/dev/null || true
            # Also use rm with glob pattern as fallback
            rm -f "$cleanup_dir"/*tmp* 2>/dev/null || true
            rm -f "$cleanup_dir"/.*tmp* 2>/dev/null || true
            # Also try to remove files based on the original md_file name pattern
            if [ -n "$original_md_basename" ]; then
                local md_base="${original_md_basename%.md}"
                rm -f "$cleanup_dir/${md_base}".tmp* 2>/dev/null || true
                rm -f "$cleanup_dir/.${md_base}".tmp* 2>/dev/null || true
            fi
        fi
    }
    
    # First generate LaTeX, then beautify tables, then compile to PDF
    local temp_tex_file="${md_basename}.temp.tex"
    
    # Verify the markdown file exists before calling Pandoc
    # If temp file was supposed to be used but doesn't exist, fall back to original
    if [ ! -f "$md_basename" ]; then
        if [ -n "$temp_md_file" ] && [ "$md_basename" = "$temp_md_file" ]; then
            echo "   ⚠️  Warning: Temp file not accessible, using original file"
            md_basename="$original_md_basename"
            temp_md_file=""
        else
            echo "❌ Error: Markdown file not found: $md_basename"
            cleanup_all
            return 1
        fi
    fi
    
    # Final check right before Pandoc - if temp file disappeared, use original
    if [ "$md_basename" != "$original_md_basename" ] && [ ! -f "$md_basename" ]; then
        echo "   ⚠️  Warning: Temp file disappeared, using original file"
        md_basename="$original_md_basename"
    fi
    
    if pandoc_output=$(pandoc "$md_basename" -o "$temp_tex_file" --from=markdown+raw_tex+link_attributes $lua_filters --to=latex --number-sections -V geometry:margin=1in --syntax-highlighting=tango -V "chapternumberstyle=$CHAPTER_STYLE" -H "$header_file" 2>&1); then
        # Beautify tables in the generated LaTeX
        if [ -f "$SCRIPT_DIR/scripts/beautify_tables.py" ]; then
            python3 "$SCRIPT_DIR/scripts/beautify_tables.py" "$temp_tex_file" || true
        fi
        # Fix alt attribute in includegraphics: remove alt={...} from \includegraphics options
        # ROOT CAUSE: Pandoc 3.8+ automatically adds alt attribute from markdown alt text,
        # but LaTeX \includegraphics doesn't support the alt key, causing "Package keyval Error: alt undefined"
        python3 << PYTHON_EOF
import sys
import os
import re
tex_file = "$temp_tex_file"
if not os.path.exists(tex_file):
    sys.exit(0)

with open(tex_file, 'r', encoding='utf-8') as f:
    content = f.read()

original_content = content

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
PYTHON_EOF
        # Compile LaTeX to PDF (run twice for proper cross-references)
        local tex_dir="$(dirname "$temp_tex_file")"
        local tex_basename="$(basename "$temp_tex_file" .tex)"
        # Try lualatex first (better emoji support), then xelatex
        if lualatex -interaction=nonstopmode -output-directory="$tex_dir" "$temp_tex_file" >/dev/null 2>&1; then
            # Run second time for cross-references
            lualatex -interaction=nonstopmode -output-directory="$tex_dir" "$temp_tex_file" >/dev/null 2>&1 || true
            # Move the generated PDF to the target location
            local generated_pdf="$tex_dir/${tex_basename}.pdf"
            if [ -f "$generated_pdf" ]; then
                mv "$generated_pdf" "$pdf_basename"
            fi
            # Clean up temporary files
            rm -f "$tex_dir/${tex_basename}".{aux,log,out} 2>/dev/null || true
            rm -f "$temp_tex_file" 2>/dev/null || true
            echo "$pandoc_output" | grep -E "\[WARNING\].*image|\[WARNING\].*resource" || true
            echo "✅ Successfully converted using lualatex"
            cleanup_all
            return 0
        elif xelatex -interaction=nonstopmode -output-directory="$tex_dir" "$temp_tex_file" >/dev/null 2>&1; then
            # Run second time for cross-references
            xelatex -interaction=nonstopmode -output-directory="$tex_dir" "$temp_tex_file" >/dev/null 2>&1 || true
            echo "✅ Successfully converted using xelatex"
            # Move the generated PDF to the target location
            local generated_pdf="$tex_dir/${tex_basename}.pdf"
            if [ -f "$generated_pdf" ]; then
                mv "$generated_pdf" "$pdf_basename"
            fi
            # Clean up temporary files (but keep header_file for now, cleanup_all will handle it)
            rm -f "$tex_dir/${tex_basename}".{aux,log,out} 2>/dev/null || true
            rm -f "$temp_tex_file" 2>/dev/null || true
            echo "$pandoc_output" | grep -E "\[WARNING\].*image|\[WARNING\].*resource" || true
            echo "✅ Successfully converted using xelatex"
            cleanup_all
            return 0
        else
            # xelatex failed, clean up temp_tex_file and auxiliary files before fallback
            rm -f "$tex_dir/${tex_basename}".{aux,log,out} 2>/dev/null || true
            rm -f "$temp_tex_file" 2>/dev/null || true
            # Restore original file for fallback before cleaning up temp file
            if [ "$md_basename" != "$original_md_basename" ]; then
                md_basename="$original_md_basename"
            fi
            # Don't cleanup temp_md_file yet - wait until after fallback attempts
        fi
    else
        # pandoc failed to generate LaTeX, clean up if temp_tex_file was created
        # Don't cleanup header_file yet, fallback path still needs it
        if [ -n "$temp_tex_file" ] && [ -f "$temp_tex_file" ]; then
            rm -f "$temp_tex_file" 2>/dev/null || true
        fi
        # DON'T restore original file - use processed file if available for fallback
        # The processed file has Code Summary removed and title page added
        # Don't cleanup temp_md_file yet - wait until after fallback attempts
    fi
    
    # Fallback: try direct conversion (original method)
    # Only run fallback if PDF was not already generated by LaTeX path
    # Check if PDF exists before running fallback
    if [ ! -f "$pdf_basename" ]; then
        # header_file should still exist at this point
        # Use processed file if available (has Code Summary removed), otherwise original
        # Try lualatex first (better emoji support), then xelatex, then pdflatex
        if pandoc_output=$(pandoc "$md_basename" -o "$pdf_basename" --from=markdown+raw_tex+link_attributes $lua_filters --pdf-engine=lualatex --number-sections -V geometry:margin=1in --syntax-highlighting=tango -V "chapternumberstyle=$CHAPTER_STYLE" -H "$header_file" 2>&1); then
        # Filter out font-related warnings but keep image warnings
        echo "$pandoc_output" | grep -E "\[WARNING\].*image|\[WARNING\].*resource" || true
        echo "✅ Successfully converted using lualatex"
        cleanup_all
        return 0
    elif pandoc_output=$(pandoc "$md_basename" -o "$pdf_basename" --from=markdown+raw_tex+link_attributes $lua_filters --pdf-engine=xelatex --number-sections -V geometry:margin=1in --syntax-highlighting=tango -V "chapternumberstyle=$CHAPTER_STYLE" -H "$header_file" 2>&1); then
        # Filter out font-related warnings but keep image warnings
        echo "$pandoc_output" | grep -E "\[WARNING\].*image|\[WARNING\].*resource" || true
        echo "✅ Successfully converted using xelatex"
        cleanup_all
        return 0
    elif pandoc_output=$(pandoc "$md_basename" -o "$pdf_basename" --from=markdown+raw_tex+link_attributes $lua_filters --pdf-engine=pdflatex --number-sections -V geometry:margin=1in --syntax-highlighting=tango -V 'tolerance=1000' -V 'emergencystretch=3em' -V "chapternumberstyle=$CHAPTER_STYLE" -H "$header_file" 2>&1); then
        echo "$pandoc_output" | grep -E "\[WARNING\].*image|\[WARNING\].*resource" || true
        echo "✅ Successfully converted using pdflatex"
        cleanup_all
        return 0
    elif pandoc_output=$(pandoc "$md_basename" -o "$pdf_basename" --from=markdown+raw_tex+link_attributes $lua_filters --number-sections -V geometry:margin=1in --syntax-highlighting=tango -V 'tolerance=1000' -V 'emergencystretch=3em' -V "chapternumberstyle=$CHAPTER_STYLE" -H "$header_file" 2>&1); then
        echo "$pandoc_output" | grep -E "\[WARNING\].*image|\[WARNING\].*resource" || true
        echo "✅ Successfully converted using default engine"
        cleanup_all
        return 0
    else
        # Show all output if conversion failed
        echo "$pandoc_output"
        echo "❌ Failed to convert: $md_file"
        # Now cleanup temp file since all attempts failed
        cleanup_all
        return 1
    fi
    fi  # Close the "if [ ! -f "$pdf_basename" ]" check for fallback
}

# If a chapter name is provided as argument
if [ $# -gt 0 ]; then
    CHAPTER_ARG="$1"
    
    # Check if it's a direct .md file path
    if [[ "$CHAPTER_ARG" == *.md ]] && [ -f "$CHAPTER_ARG" ]; then
        convert_md_to_pdf "$CHAPTER_ARG"
        exit $?
    fi
    
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
