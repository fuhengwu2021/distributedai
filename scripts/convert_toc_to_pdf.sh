#!/bin/bash

# Script to convert toc.md to toc.pdf
# Usage: ./scripts/convert_toc_to_pdf.sh [--use-eisvogel]
#
# Options:
#   --use-eisvogel    Use eisvogel template if available (requires eisvogel installed)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TOC_MD="$ROOT_DIR/toc.md"
TOC_PDF="$ROOT_DIR/toc.pdf"
USE_EISVOGEL=false

# Parse arguments
if [ "$1" = "--use-eisvogel" ]; then
    USE_EISVOGEL=true
fi

if [ ! -f "$TOC_MD" ]; then
    echo "❌ Error: toc.md not found at $TOC_MD"
    echo "   Run: python3 scripts/generate_toc.py"
    exit 1
fi

# Generate a LaTeX header for TOC with eisvogel-inspired styling
HEADER_FILE=$(mktemp)
cat > "$HEADER_FILE" << 'EOF'
% LaTeX packages for TOC (inspired by eisvogel template)
% Note: geometry is loaded by pandoc via -V geometry option, don't load it here
\usepackage{fontspec}
\usepackage{xcolor}
\usepackage{titlesec}
\usepackage{setspace}
\usepackage{hyperref}

% Define colors (matching your book's color scheme)
\definecolor{chapterblue}{RGB}{0,102,204}
\definecolor{chapterbluelight}{RGB}{153,204,255}
\definecolor{chaptergray}{RGB}{128,128,128}

% Line spacing (eisvogel uses 1.2)
\setstretch{1.2}

% Paragraph formatting (no indentation, spacing between paragraphs)
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.5em plus 0.1em minus 0.1em}

% Title formatting - Main title
\titleformat{\section}
  {\Large\bfseries\color{chapterblue}}
  {}
  {0em}
  {}
\titlespacing*{\section}{0pt}{1.5em}{0.8em}

% Subsection formatting - Part names
\titleformat{\subsection}
  {\large\bfseries\color{chapterblue}}
  {}
  {0em}
  {}
\titlespacing*{\subsection}{0pt}{1.2em}{0.6em}

% Subsubsection formatting - Chapter entries
\titleformat{\subsubsection}
  {\normalsize\bfseries\color{black}}
  {}
  {0em}
  {}
\titlespacing*{\subsubsection}{0pt}{0.8em}{0.4em}

% Hyperlink setup
\hypersetup{
    colorlinks=true,
    linkcolor=chapterblue,
    urlcolor=chapterblue,
    citecolor=chapterblue
}

% Page setup
\pagestyle{plain}

% Italic text styling for subtitles
\renewcommand{\emph}[1]{\textit{#1}}

EOF

# Convert using pandoc
echo "📄 Converting: toc.md → toc.pdf"

# Check if eisvogel template is available
if [ "$USE_EISVOGEL" = true ]; then
    # Check if eisvogel template exists by trying to access it
    if pandoc --print-default-data-file=templates/eisvogel.latex > /dev/null 2>&1 || \
       [ -f "$HOME/.local/share/pandoc/templates/eisvogel.latex" ] || \
       [ -f "$HOME/.pandoc/templates/eisvogel.latex" ]; then
        echo "   Using eisvogel template"
        if pandoc "$TOC_MD" \
            -o "$TOC_PDF" \
            --from=markdown \
            --template=eisvogel \
            --pdf-engine=xelatex \
            --number-sections \
            -V geometry:margin=2.5cm \
            -V colorlinks=true \
            -V linkcolor=blue \
            2>&1; then
            echo "✅ Successfully generated: toc.pdf (using eisvogel template)"
            rm -f "$HEADER_FILE"
            exit 0
        else
            echo "⚠️  eisvogel template failed, falling back to custom styling"
        fi
    else
        echo "⚠️  eisvogel template not found, using custom styling"
    fi
fi

# Use custom styling (inspired by eisvogel)
if pandoc "$TOC_MD" \
    -o "$TOC_PDF" \
    --from=markdown \
    --pdf-engine=xelatex \
    --number-sections \
    -V geometry:margin=2.5cm \
    -H "$HEADER_FILE" \
    2>&1; then
    echo "✅ Successfully generated: toc.pdf"
    rm -f "$HEADER_FILE"
    exit 0
else
    echo "❌ Failed to convert toc.md to PDF"
    rm -f "$HEADER_FILE"
    exit 1
fi
