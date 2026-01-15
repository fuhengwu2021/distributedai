% Minimal LaTeX template - simpler styling, fewer packages
% Unicode support - use fontspec for xelatex, inputenc for pdflatex
\ifxetex
  \usepackage{fontspec}
\else
  \usepackage[utf8]{inputenc}
  \usepackage[T1]{fontenc}
\fi
\usepackage{microtype}
\sloppy
\setlength{\emergencystretch}{3em}
\tolerance=1000
\allowdisplaybreaks
\usepackage{float}
\floatplacement{figure}{H}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{listings}
\usepackage{fancyhdr}

% Define basic colors
\definecolor{chapterblue}{RGB}{0,102,204}
\definecolor{chaptergray}{RGB}{128,128,128}

% Basic code block styling
\lstset{
  basicstyle=\ttfamily\normalsize,
  breaklines=true,
  breakatwhitespace=true,
  showstringspaces=false,
  frame=single,
  backgroundcolor=\color{gray!10},
  commentstyle=\color{gray}\itshape,
  keywordstyle=\color{chapterblue}\bfseries,
  stringstyle=\color{black},
  numberstyle=\tiny\color{gray},
  numbers=left,
  stepnumber=5,
  numbersep=5pt,
}

% Page headers
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\color{chaptergray}\small\itshape {{header_title}}}
\fancyhead[R]{\color{chaptergray}\small\thepage}
\fancyfoot[R]{\color{chaptergray}\small\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\setlength{\headheight}{14.5pt}
\setlength{\footskip}{14.5pt}

% Image scaling
\makeatletter
\def\maxwidth{\ifdim\Gin@nat@width>\linewidth\linewidth\else\Gin@nat@width\fi}
\def\maxheight{\ifdim\Gin@nat@height>\textheight\textheight\else\Gin@nat@height\fi}
\makeatother
\setkeys{Gin}{width=\maxwidth,height=\maxheight,keepaspectratio}

% Footnotes
\setlength{\footnotesep}{0.5cm}
\renewcommand{\footnoterule}{\vspace*{8pt}\hrule width 0.4\columnwidth height 0.4pt \vspace*{4pt}}
\setlength{\skip\footins}{1.2cm}

% Subsection numbering
\renewcommand{\thesubsection}{\arabic{subsection}}

% Table styling - simple alternating colors
\definecolor{tableyellow}{RGB}{255,255,200}
\definecolor{tablered}{RGB}{255,230,230}
\renewcommand{\arraystretch}{1.2}

% Chapter title page - simple style
% Note: This template uses a simpler chapter title page without fancy decorations
% The placeholders below are required for compatibility with generate_latex_header.py
\NewDocumentEnvironment{chaptertitlepage}{m m m O{} O{}}{%
  \newpage
  \thispagestyle{empty}
  \pdfbookmark[1]{Chapter #1: #2}{chapter#1}
  \vspace*{2cm}
  \begin{center}
    {\color{chaptergray}\large\bfseries Chapter #1}\\[0.5cm]
    {\color{chapterblue}\fontsize{24}{32}\selectfont\bfseries #2}\\[0.5cm]
    {\itshape\normalsize #3}
  \end{center}
  \vfill
  \newpage
}{}
% Placeholders for chapter number style (required by generate_latex_header.py)
% These are replaced by the script but not used in this minimal template
CHAPTER_NUMBER_STYLE_PLACEHOLDER
% Chapter number color placeholder (required by generate_latex_header.py)
\newcommand{\chapternumbercolor}{CHAPTER_NUMBER_COLOR_PLACEHOLDER}
