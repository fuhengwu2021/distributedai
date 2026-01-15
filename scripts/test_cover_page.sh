#!/bin/bash
# Test script for cover page PDF generation
# 
# This script generates a test PDF with only the cover page to preview different cover styles.
# It automatically generates the cover background if it doesn't exist.
#
# Usage: ./test_cover_page.sh [style]
#   style: classic, modern, minimal, bold (default: classic)
#
# Examples:
#   ./test_cover_page.sh classic    # Generate classic style cover
#   ./test_cover_page.sh modern      # Generate modern style cover
#   ./test_cover_page.sh minimal     # Generate minimal style cover
#   ./test_cover_page.sh bold        # Generate bold style cover
#
# Output: test_cover_<style>.pdf in the project root directory

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Parse style argument
COVER_STYLE="${1:-classic}"
if [[ ! "$COVER_STYLE" =~ ^(classic|modern|minimal|bold)$ ]]; then
    echo "❌ Error: Invalid style '$COVER_STYLE'"
    echo "   Valid styles: classic, modern, minimal, bold"
    exit 1
fi

echo "📄 Testing cover page generation with style: $COVER_STYLE"

# Ensure cover background exists
COVER_BG="$PROJECT_ROOT/img/cover-background.pdf"
if [ ! -f "$COVER_BG" ]; then
    echo "  Generating cover background..."
    python3 "$SCRIPT_DIR/generate_cover_background.py" || {
        echo "❌ Failed to generate cover background"
        exit 1
    }
fi

if [ ! -f "$COVER_BG" ]; then
    echo "❌ Error: Cover background not found at $COVER_BG"
    exit 1
fi

# Create temporary directory
TEMP_DIR="/tmp/cover_test_$$"
mkdir -p "$TEMP_DIR"
trap "rm -rf '$TEMP_DIR'" EXIT

# Use relative path from temp directory to cover background
# This avoids issues with special characters in absolute paths
COVER_BG_REL="$TEMP_DIR/cover-bg-link.pdf"
# Create a symlink or copy to avoid path issues
cp "$COVER_BG" "$COVER_BG_REL" 2>/dev/null || ln -sf "$COVER_BG" "$COVER_BG_REL" 2>/dev/null || {
    # If both fail, use absolute path with proper escaping
    COVER_BG_REL="$COVER_BG"
}

# Create test LaTeX file
TEX_FILE="$TEMP_DIR/test_cover.tex"
OUTPUT_PDF="$PROJECT_ROOT/test_cover_${COVER_STYLE}.pdf"

cat > "$TEX_FILE" << EOF
% Test cover page generation - single page only
\documentclass[a4paper,oneside]{article}
\pagestyle{empty}
% Prevent extra pages
\setlength{\topskip}{0pt}
\setlength{\topmargin}{0pt}
\setlength{\headheight}{0pt}
\setlength{\headsep}{0pt}
\setlength{\footskip}{0pt}
\setlength{\textheight}{\paperheight}
\setlength{\textwidth}{\paperwidth}
\setlength{\oddsidemargin}{0pt}
\setlength{\evensidemargin}{0pt}
\setlength{\marginparwidth}{0pt}
\setlength{\marginparsep}{0pt}
\setlength{\hoffset}{-1in}
\setlength{\voffset}{-1in}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{fontspec}
\usepackage{graphicx}
\usepackage{tikz}
\usetikzlibrary{calc}
\usepackage{eso-pic}  % For background images
\usepackage{xcolor}
\usepackage{etoolbox}  % For \detokenize

% Define colors
\definecolor{chapterblue}{RGB}{0,102,204}
\definecolor{chapterbluelight}{RGB}{153,204,255}
\definecolor{chaptergray}{RGB}{128,128,128}

% Cover page commands (from templates/book/default.tpl)
% Classic style - centered layout
\newcommand{\coverpageclassic}[1]{%
  \thispagestyle{empty}%
  \AddToShipoutPictureBG*{%
    \AtPageLowerLeft{%
      \begin{tikzpicture}[remember picture,overlay]
        \node[anchor=south west,inner sep=0,opacity=0.5] at (current page.south west) {%
          \includegraphics[width=\paperwidth,height=\paperheight,keepaspectratio=false]{#1}%
        };%
      \end{tikzpicture}%
    }%
  }%
  \noindent
  \begin{minipage}{\textwidth}
    \centering
    \vspace*{0.16\paperheight}%
    {\fontsize{68}{82}\selectfont\bfseries\textcolor{black}{Math for AI/ML}}\par\vspace{1.0cm}
    {\fontsize{26}{32}\selectfont\textcolor{black}{A comprehensive mathematics textbook for AI/ML}}\par\vspace{2.5cm}
    {\fontsize{24}{30}\selectfont\textcolor{black}{MockSphere.com}}\par\vspace{1.2cm}
    \vspace*{1.0cm}%
    {\fontsize{20}{26}\selectfont\textcolor{black}{Second Edition}}\par\vspace{0.3cm}
    {\fontsize{22}{28}\selectfont\bfseries\textcolor{black}{2026}}
    \vspace*{0.10\paperheight}%
  \end{minipage}%
}

% Modern style - left-aligned, bold typography
\newcommand{\coverpagemodern}[1]{%
  \thispagestyle{empty}
  \AddToShipoutPictureBG*{%
    \AtPageLowerLeft{%
      \begin{tikzpicture}[remember picture,overlay]
        \node[anchor=south west,inner sep=0,opacity=0.4] at (current page.south west) {%
          \includegraphics[width=\paperwidth,height=\paperheight,keepaspectratio=false]{#1}%
        };%
      \end{tikzpicture}%
    }%
  }%
  \begin{tikzpicture}[remember picture,overlay]
    \node[anchor=north west, xshift=2cm, yshift=-3cm] at (current page.north west) {%
      \begin{minipage}{0.7\textwidth}
        \raggedright
        {\fontsize{72}{86}\selectfont\bfseries\textcolor{black}{Math for AI/ML}}\par\vspace{0.5cm}
        {\fontsize{28}{34}\selectfont\textcolor{black!70}{A comprehensive mathematics textbook for AI/ML}}
      \end{minipage}
    };
    \node[anchor=south west, xshift=2cm, yshift=2.5cm] at (current page.south west) {%
      \begin{minipage}{0.5\textwidth}
        \raggedright
        {\fontsize{22}{28}\selectfont\textcolor{black}{MockSphere.com}}\par\vspace{0.3cm}
        {\fontsize{18}{24}\selectfont\textcolor{black!80}{Second Edition \textbullet\ 2026}}
      \end{minipage}
    };
  \end{tikzpicture}
}

% Minimal style - clean and simple
\newcommand{\coverpageminimal}[1]{%
  \thispagestyle{empty}
  \AddToShipoutPictureBG*{%
    \AtPageLowerLeft{%
      \begin{tikzpicture}[remember picture,overlay]
        \node[anchor=south west,inner sep=0,opacity=0.3] at (current page.south west) {%
          \includegraphics[width=\paperwidth,height=\paperheight,keepaspectratio=false]{#1}%
        };%
      \end{tikzpicture}%
    }%
  }%
  \begin{center}
    \vspace*{0.25\paperheight}%
    {\fontsize{56}{68}\selectfont\bfseries\textcolor{black!90}{Math for AI/ML}}\par\vspace{1.5cm}
    {\fontsize{18}{24}\selectfont\textcolor{black!70}{A comprehensive mathematics textbook for AI/ML}}\par\vspace{4cm}
    {\fontsize{16}{20}\selectfont\textcolor{black!60}{MockSphere.com}}\par\vspace{0.5cm}
    {\fontsize{14}{18}\selectfont\textcolor{black!50}{Second Edition \textbullet\ 2026}}
    \vspace*{0.20\paperheight}%
  \end{center}
}

% Bold style - dramatic and eye-catching
\newcommand{\coverpagebold}[1]{%
  \thispagestyle{empty}
  \AddToShipoutPictureBG*{%
    \AtPageLowerLeft{%
      \begin{tikzpicture}[remember picture,overlay]
        \node[anchor=south west,inner sep=0,opacity=0.6] at (current page.south west) {%
          \includegraphics[width=\paperwidth,height=\paperheight,keepaspectratio=false]{#1}%
        };%
      \end{tikzpicture}%
    }%
  }%
  \begin{center}
    \vspace*{0.12\paperheight}%
    {\fontsize{80}{96}\selectfont\bfseries\textcolor{black}{Math for AI/ML}}\par\vspace{0.8cm}
    {\fontsize{30}{36}\selectfont\bfseries\textcolor{black!85}{A comprehensive mathematics textbook for AI/ML}}\par\vspace{3cm}
    {\fontsize{28}{34}\selectfont\bfseries\textcolor{black}{MockSphere.com}}\par\vspace{1.5cm}
    \vspace*{0.5cm}%
    {\fontsize{24}{30}\selectfont\bfseries\textcolor{black!90}{Second Edition}}\par\vspace{0.2cm}
    {\fontsize{32}{38}\selectfont\bfseries\textcolor{black}{2026}}
    \vspace*{0.08\paperheight}%
  \end{center}
}

\begin{document}
\coverpage${COVER_STYLE}{$(basename "$COVER_BG_REL")}%
\end{document}
EOF

echo "  LaTeX file created: $TEX_FILE"
echo "  Compiling with xelatex..."

# Compile LaTeX (need to be in temp dir for relative paths to work)
cd "$TEMP_DIR"
# Copy cover background to temp dir if using symlink
if [ -L "$COVER_BG_REL" ]; then
    cp "$COVER_BG" "$TEMP_DIR/cover-bg-link.pdf" 2>/dev/null || true
fi

# Compile LaTeX (suppress output, but check if PDF was generated)
xelatex -interaction=nonstopmode -output-directory="$TEMP_DIR" test_cover.tex > /dev/null 2>&1 || true

# Check if PDF was actually generated (xelatex may return error code even if PDF is created)
if [ ! -f "$TEMP_DIR/test_cover.pdf" ]; then
    echo "❌ LaTeX compilation failed. Showing errors:"
    xelatex -interaction=nonstopmode -output-directory="$TEMP_DIR" test_cover.tex
    exit 1
fi

# Move PDF to project root
if [ -f "$TEMP_DIR/test_cover.pdf" ]; then
    mv "$TEMP_DIR/test_cover.pdf" "$OUTPUT_PDF"
    
    # Verify it's a single page
    pages=$(pdfinfo "$OUTPUT_PDF" 2>/dev/null | grep -i pages | awk '{print $2}' || pdftk "$OUTPUT_PDF" dump_data 2>/dev/null | grep -i "numberofpages" | awk '{print $2}' || echo "unknown")
    if [ "$pages" != "1" ] && [ "$pages" != "unknown" ]; then
        echo "  ⚠️  Warning: PDF has $pages pages (expected 1)"
    fi
    
    echo "✅ Cover page PDF generated: $OUTPUT_PDF"
    echo "   Style: $COVER_STYLE"
    echo "   Background: $COVER_BG"
    echo "   Pages: $pages"
else
    echo "❌ Error: PDF not generated"
    exit 1
fi
