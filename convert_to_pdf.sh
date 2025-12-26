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
    
    # Extract chapter title from markdown file for header
    local chapter_title=""
    local header_title_text="Modern Distributed AI Systems"
    if grep -q "^# Chapter [0-9]\+:" "$original_md_basename"; then
        # Extract chapter title: "Chapter N: Title" -> "Title"
        chapter_title=$(grep "^# Chapter [0-9]\+:" "$original_md_basename" | sed 's/^# Chapter [0-9]\+: *//' | head -1)
        if [ -n "$chapter_title" ]; then
            # Escape LaTeX special characters in chapter title
            header_title_text=$(echo "$chapter_title" | sed 's/\\/\\textbackslash{}/g; s/{/\\{/g; s/}/\\}/g; s/\$/\\$/g; s/&/\\&/g; s/%/\\%/g; s/#/\\#/g; s/\^/\\textasciicircum{}/g; s/_/\\_/g')
        fi
    fi
    
    # Try different PDF engines in order of preference
    # Use basenames since we're now in the chapter directory
    # Capture output to filter warnings but show errors
    local pandoc_output=""
    # Create LaTeX header with image size control
    # Keep images at original size to prevent blurriness from over-scaling
    # Only scale down if image exceeds page width
    # Use a dynamic header that includes chapter title
    # Use quoted heredoc to prevent $ expansion, then replace placeholder
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
\usepackage{enumitem}
\usepackage{etoolbox}
% Define colors before using them in listings
\definecolor{chapterblue}{RGB}{0,102,204}
\definecolor{chapterbluelight}{RGB}{153,204,255}
\definecolor{chaptergray}{RGB}{128,128,128}
\definecolor{dividerred}{RGB}{255,0,0}
% Code block styling with line number annotations using listings package
\usepackage{listings}
% Use mdframed to wrap listings and add custom line numbers
\usepackage{mdframed}
% Use soul package for text highlighting
\usepackage{soul}
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
% Define mdframed style for code blocks with border and rounded corners
% Note: roundcorner may not work in all mdframed versions, but we keep it for compatibility
\mdfdefinestyle{codeblockstyle}{%
  leftmargin=0pt,
  rightmargin=0pt,
  innerleftmargin=5pt,
  innerrightmargin=10pt,
  innertopmargin=10pt,
  innerbottommargin=5pt,
  skipabove=0.5em,
  skipbelow=0.5em,
  linecolor=chapterbluelight,
  linewidth=0.8pt,
  backgroundcolor=gray!5,
  roundcorner=4pt,
  outerlinewidth=0.8pt,
  % Ensure escapeinside works properly
  hidealllines=false,
  % Ensure symmetric inner padding
  innerlinewidth=0pt,
  outermargin=0pt,
}
% Command to add circled number mark (for use in explanations outside code blocks)
% First argument: style ("normal" or "solid")
% Second argument: line number
% Use lighter colors and position closer to left border
\newcommand{\codelinemark}[2]{%
  \ifstrequal{#1}{solid}{%
    % Solid fill for highlighted lines (dark blue)
    \tikz[baseline=(char.base)]{%
      \node[shape=circle,draw=chapterblue,fill=chapterblue,inner sep=2pt,minimum size=1.2em,font=\tiny\bfseries\color{white}] (char) {#2};%
    }%
  }{%
    % Normal style (lighter colors)
    \tikz[baseline=(char.base)]{%
      \node[shape=circle,draw=chapterbluelight,fill=chapterbluelight!20,inner sep=2pt,minimum size=1.2em,font=\tiny\bfseries\color{chapterbluelight}] (char) {#2};%
    }%
  }%
}
% Command for code line explanation (used outside code block)
% Use dark blue for consistency with highlighted lines
% Ensure perfect baseline alignment between circle and text
% Use moderate adjustment to find middle ground
\newcommand{\codelineannotation}[2]{%
  \tikz[baseline=-0.35ex]{%
    \node[shape=circle,draw=chapterblue,fill=chapterblue,inner sep=2pt,minimum size=1.2em,font=\tiny\bfseries\color{white}] (char) {#1};%
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
% Page headers using fancyhdr
\usepackage{fancyhdr}
\pagestyle{fancy}
% Clear default header/footer
\fancyhf{}
% Set header: left side shows chapter title (or book title if no chapter), right side shows page number
\fancyhead[L]{\color{chaptergray}\small\itshape HEADER_TITLE_PLACEHOLDER}
\fancyhead[R]{\color{chaptergray}\small\thepage}
% Add a line under the header
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\headrule}{\hbox to\headwidth{\color{chapterbluelight}\leaders\hrule height \headrulewidth\hfill}}
% Set header height to accommodate the content
\setlength{\headheight}{14.5pt}
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
\NewDocumentCommand{\fancydividerwithicon}{O{chapterbluelight} O{0.95\textwidth} m}{%
  \par\vspace{0.4cm}%
  \noindent%
  \begin{tikzpicture}%
    \coordinate (line-end) at (#2,0);%
    \draw[#1,line width=0.8pt] (0,0) -- (line-end);%
    \node[anchor=center,inner sep=0] at (line-end) {%
      \includegraphics[height=0.7cm,keepaspectratio]{#3}%
    };%
  \end{tikzpicture}%
  \par\vspace{0.4cm}%
}
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
      \centering
      \begin{itemize}[label=\raisebox{1ex}{\textcolor{chapterblue}{$\triangleright$}},leftmargin=*,itemsep=0.15cm,topsep=0pt,partopsep=0pt]
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
    # Replace placeholder with actual title (escape special chars for sed)
    local escaped_title=$(printf '%s\n' "$header_title_text" | sed 's/[[\.*^$()+?{|]/\\&/g' | sed 's/\\/\\\\/g')
    latex_header=$(echo "$latex_header" | sed "s|HEADER_TITLE_PLACEHOLDER|$escaped_title|g")
    # Cleanup function (defined before use)
    cleanup_temp() {
        if [ -n "$temp_md_file" ] && [ -f "$temp_md_file" ]; then
            rm -f "$temp_md_file"
        fi
    }
    
    # Use Lua filter for code line numbers if available
    local lua_filter=""
    if [ -f "$SCRIPT_DIR/scripts/code_line_numbers.lua" ]; then
        lua_filter="--lua-filter=$SCRIPT_DIR/scripts/code_line_numbers.lua"
    fi
    
    # Write header to temporary file to ensure proper handling
    local header_file="${md_basename}.header.$$.tex"
    echo "$latex_header" > "$header_file"
    
    # Enhanced cleanup function
    cleanup_all() {
        cleanup_temp
        if [ -n "$header_file" ] && [ -f "$header_file" ]; then
            rm -f "$header_file"
        fi
    }
    
    if pandoc_output=$(pandoc "$md_basename" -o "$pdf_basename" --from=markdown+raw_tex $lua_filter --pdf-engine=xelatex -V geometry:margin=1in --highlight-style=tango -H "$header_file" 2>&1); then
        # Filter out font-related warnings but keep image warnings
        echo "$pandoc_output" | grep -E "\[WARNING\].*image|\[WARNING\].*resource" || true
        echo "✅ Successfully converted using xelatex"
        cleanup_all
        return 0
    elif pandoc_output=$(pandoc "$md_basename" -o "$pdf_basename" --from=markdown+raw_tex $lua_filter --pdf-engine=pdflatex -V geometry:margin=1in --highlight-style=tango -V 'tolerance=1000' -V 'emergencystretch=3em' -H "$header_file" 2>&1); then
        echo "$pandoc_output" | grep -E "\[WARNING\].*image|\[WARNING\].*resource" || true
        echo "✅ Successfully converted using pdflatex"
        cleanup_all
        return 0
    elif pandoc_output=$(pandoc "$md_basename" -o "$pdf_basename" --from=markdown+raw_tex $lua_filter -V geometry:margin=1in --highlight-style=tango -V 'tolerance=1000' -V 'emergencystretch=3em' -H "$header_file" 2>&1); then
        echo "$pandoc_output" | grep -E "\[WARNING\].*image|\[WARNING\].*resource" || true
        echo "✅ Successfully converted using default engine"
        cleanup_all
        return 0
    else
        # Show all output if conversion failed
        echo "$pandoc_output"
        echo "❌ Failed to convert: $md_file"
        cleanup_all
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
