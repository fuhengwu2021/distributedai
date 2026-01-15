% Unicode support - use fontspec for xelatex/lualatex, inputenc for pdflatex
\ifluatex
  \usepackage{fontspec}
  % LuaLaTeX: Set up emoji fallback font
  \directlua{luaotfload.add_fallback("emojifallback",{"Noto Color Emoji:mode=harf"})}
  % Use default font with emoji fallback
  \setmainfont[RawFeature={fallback=emojifallback}]{Latin Modern Roman}
\else\ifxetex
  \usepackage{fontspec}
  % Let xelatex use default system fonts (will handle Unicode properly)
  % Note: For emoji support, LuaLaTeX with mainfontfallback is recommended
\else
  \usepackage[utf8]{inputenc}
  \usepackage[T1]{fontenc}
  \usepackage{textcomp}
\fi\fi
\usepackage{amsmath}  % Required for \eqref command
\usepackage{amsthm}  % Required for \qed command in proofbox
\usepackage{microtype}
\sloppy
\setlength{\emergencystretch}{3em}
\tolerance=1000
\allowdisplaybreaks
\usepackage{float}
\floatplacement{figure}{H}
\usepackage{graphicx}
\usepackage{wrapfig}  % For text wrapping around images
% Pass table option to xcolor BEFORE any package that loads xcolor (like colortbl)
\PassOptionsToPackage{table}{xcolor}
% Load colortbl first (it will load xcolor with the table option we just passed)
\makeatletter
\@ifpackageloaded{colortbl}{}{\usepackage{colortbl}}
\makeatother
% Now explicitly load xcolor[table] (colortbl already loaded it, but this ensures it's available)
\usepackage[table]{xcolor}  % xcolor with table option for \rowcolors support
% Table styling packages
\usepackage{longtable}  % Required for longtable environments (must be loaded before \AtBeginEnvironment hook)
\usepackage{booktabs}  % Better table lines
\usepackage{arydshln}  % Dashed lines in tables
\usepackage{tikz}
\usetikzlibrary{calc}
\usepackage{eso-pic}  % For background images
% Use tcolorbox to wrap listings with rounded corners (better than mdframed)
% tcolorbox provides reliable rounded corners support
% Load with [most] option to get all libraries including listings
\usepackage[most]{tcolorbox}
\tcbuselibrary{listings}
\tcbuselibrary{breakable}
\usepackage{xparse}
\usepackage{enumitem}
\usepackage{etoolbox}
\usepackage{xstring}  % For string manipulation (cleaning titles for PDF bookmarks)
% Define common math shorthand commands
\newcommand{\R}{\mathbb{R}}
\newcommand{\C}{\mathbb{C}}
\newcommand{\N}{\mathbb{N}}
\newcommand{\Z}{\mathbb{Z}}
% Define colors before using them in listings
\definecolor{chapterblue}{RGB}{0,102,204}
\definecolor{chapterbluelight}{RGB}{153,204,255}
\definecolor{chaptergray}{RGB}{128,128,128}
\definecolor{dividerred}{RGB}{255,0,0}
% Code block styling with line number annotations using listings package
\usepackage{listings}
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
  % Allow lstlisting to break across pages when inside breakable tcolorbox
  aboveskip=0pt,
  belowskip=0pt,
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
    breakable,
    enhanced
  }
}
% Command to add circled number mark (for use in explanations outside code blocks)
% First argument: style ("normal" or "solid")
% Second argument: line number
% Use lighter colors and position closer to left border
\newcommand{\codelinemark}[2]{%
  \ifstrequal{#1}{solid}{%
    % Solid fill for highlighted lines (dark blue circle)
    \tikz[baseline=(char.base)]{%
      \node[shape=circle,draw=chapterblue,fill=chapterblue,inner sep=2pt,minimum size=1.3em,font=\scriptsize\bfseries\color{white},line width=0.8pt] (char) {#2};%
    }%
  }{%
    % Normal style: transparent box with only right border (same as linebar style)
    \tikz[baseline=(box.base)]{%
      \node[draw=none,fill=none,inner sep=2pt,minimum height=1.2em,font=\scriptsize\bfseries\color{chapterbluelight},anchor=base west] (box) {%
        \makebox[1.5em][r]{#2}%
      };%
      % Draw only the right border (vertical line from top to bottom)
      \draw[chapterbluelight,line width=0.8pt] (box.north east) -- (box.south east);
    }%
    \hspace{-1.5pt}%
  }%
}
% Command to add line number in a box (e.g., "13" or " 1")
% First argument: style ("normal" or "solid")
% Second argument: line number
\newcommand{\codelinemarkbar}[2]{%
  \ifstrequal{#1}{solid}{%
    % Solid style for highlighted lines (dark blue background)
    \tikz[baseline=(box.base)]{%
      \node[draw=chapterblue,fill=chapterblue,inner sep=2pt,minimum height=1.2em,font=\scriptsize\bfseries\color{white},anchor=base west] (box) {%
        \makebox[1.5em][r]{#2}%
      };%
    }%
    \hspace{-3pt}%
  }{%
    % Normal style: transparent box with only right border
    \tikz[baseline=(box.base)]{%
      \node[draw=none,fill=none,inner sep=2pt,minimum height=1.2em,font=\scriptsize\bfseries\color{chapterbluelight},anchor=base west] (box) {%
        \makebox[1.5em][r]{#2}%
      };%
      % Draw only the right border (vertical line from top to bottom)
      \draw[chapterbluelight,line width=0.8pt] (box.north east) -- (box.south east);
    }%
    \hspace{-3pt}%
  }%
}
% Command for code line explanation (used outside code block)
% Use dark blue for consistency with highlighted lines
% Ensure perfect baseline alignment between circle and text
\newcommand{\codelineannotation}[2]{%
  \tikz[baseline=(char.north)]{%
    \node[shape=circle,draw=chapterblue,fill=chapterblue,inner sep=2pt,minimum size=1.2em,font=\scriptsize\bfseries\color{white}] (char) {#1};%
  }%
  \quad\begin{minipage}[t]{0.85\textwidth}\raggedright\color{black}#2\end{minipage}\par%
}
% Command for code line explanation with box style (used when #LINEBAR is active)
% Use box style to match the line number style in code blocks
\newcommand{\codelineannotationbar}[2]{%
  \tikz[baseline=(box.north)]{%
    \node[draw=chapterblue,fill=chapterblue,inner sep=2pt,minimum height=1.2em,font=\scriptsize\bfseries\color{white},anchor=north west] (box) {%
      \makebox[1.5em][r]{#1}%
    };%
  }%
  \quad\begin{minipage}[t]{0.85\textwidth}\raggedright\color{black}#2\end{minipage}\par%
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
% Page headers using fancyhdr
\usepackage{fancyhdr}
\pagestyle{fancy}
% Clear default header/footer
\fancyhf{}
% Set header: left side shows chapter number and title (or book title if no chapter), right side shows page number
% Reduce vertical spacing in header by using raisebox to move text down closer to rule
\fancyhead[L]{\raisebox{-6pt}{\color{chaptergray}\small\itshape {{header_title}}}}
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
% Modify subsection numbering to show section.subsection format
% In book class, this will show as 1.1, 1.2, 1.3, etc. (chapter.section.subsection would be too long)
% For article class, this shows as 1.1, 1.2, etc.
\renewcommand{\thesubsection}{\thesection.\arabic{subsection}}
% Modify equation numbering to include chapter number (e.g., (15.1) instead of (1))
% Define chapter number variable (will be set from markdown if available)
\newcommand{\mychapternum}{{{chapter_number}}}
% Check if chapter number is provided (not the special marker)
\makeatletter
\def\@nochapternum{NOSUCHCHAPTERNUMBER}
\ifx\mychapternum\@nochapternum
  % No chapter number provided, use subsection
  \renewcommand{\theequation}{\thesubsection.\arabic{equation}}
\else
  % Chapter number provided, use it
  \renewcommand{\theequation}{\mychapternum.\arabic{equation}}
\fi
\makeatother
% Table styling: alternating row colors and horizontal lines
% Define light yellow and light red for alternating rows
\definecolor{tableyellow}{RGB}{255,255,200}
\definecolor{tablered}{RGB}{255,230,230}
% Apply alternating row colors to all tables
% Use \rowcolors{start}{odd}{even} to set alternating colors
% This will be applied via pandoc's table processing
% For better table appearance, we'll use booktabs style with horizontal lines
\renewcommand{\arraystretch}{1.2}  % Increase row height for better readability
% Configure booktabs for better table lines
\setlength{\heavyrulewidth}{0.08em}  % Top and bottom rule thickness
\setlength{\lightrulewidth}{0.05em}  % Mid rule thickness
\setlength{\cmidrulewidth}{0.05em}   % Partial rule thickness
\setlength{\aboverulesep}{0.5ex}     % Space above rules
\setlength{\belowrulesep}{0.5ex}     % Space below rules
% Automatically apply rowcolors to all longtable and tabular environments
% This uses etoolbox to patch the environments
% Note: etoolbox is already loaded above
% NOTE: The \AtBeginEnvironment hooks are DISABLED because they cause
% "Undefined control sequence" errors with longtable. The beautify_tables.py
% script handles rowcolors insertion directly in the tables, which is more
% reliable and doesn't have timing/package loading issues.
% If beautify_tables.py is not used, uncomment these hooks:
% Patch longtable to add rowcolors
% longtable structure: \toprule, header, \midrule, \endhead, data rows
% \AtBeginEnvironment{longtable}{%
%   \rowcolors{2}{tablered}{tableyellow}%
% }
% Patch tabular to add rowcolors (for inline tables, skip header if present)
% \AtBeginEnvironment{tabular}{%
%   \rowcolors{2}{tablered}{tableyellow}%
% }
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
% Chapter number style - generated based on chapter_style variable
\NewDocumentEnvironment{chaptertitlepage}{m m m O{} O{}}{%
  \newpage
  \thispagestyle{empty}
  % Reset section counters for article class
  % This ensures section numbering starts at 1 (e.g., 1.1, 1.2, not 0.1, 0.2)
  % Note: In article class, we set section to -1 so that the first \section increments it to 0,
  % but LaTeX section numbering uses \arabic{section} which shows 0 as "0", not "1"
  % Actually, we need to set it to 0, and LaTeX will increment it to 1 for the first section
  % But if TOC is present, it might consume the counter. Let reset_section_counter.lua handle it.
  % We don't set it here to avoid conflicts with reset_section_counter.lua
  % Add PDF bookmark for chapter title (for navigation in PDF readers)
  % Clean LaTeX line break commands from title for metadata
  \StrSubstitute{#2}{\\[0.3cm]}{ }[\@cleantitle]
  \StrSubstitute{\@cleantitle}{\\[0.5cm]}{ }[\@cleantitle]
  \StrSubstitute{\@cleantitle}{\\[0.6cm]}{ }[\@cleantitle]
  \pdfbookmark[1]{Chapter #1: \@cleantitle}{chapter#1}
  \vspace*{-2cm}
  \begin{tikzpicture}[remember picture,overlay]
CHAPTER_NUMBER_STYLE_PLACEHOLDER
    % Chapter number (positioned in the visible area) - use the coordinate defined above
    % Color depends on style: white for square (blue background), black for circle (white background)
    \node[CHAPTER_NUMBER_COLOR_PLACEHOLDER,font=\fontsize{60}{72}\selectfont\bfseries,anchor=center] 
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

% Custom cover page with background image
% Usage: \coverpagewithbackground{path/to/background.pdf}
% Layout inspired by: https://sein.wu-99.com/_images/sen.png
\newcommand{\coverpagewithbackground}[1]{%
  \newpage
  \thispagestyle{empty}
  \AddToShipoutPictureBG*{%
    \AtPageLowerLeft{%
      \begin{tikzpicture}[remember picture,overlay]
        \node[anchor=south west,inner sep=0,opacity=0.5] at (current page.south west) {%
          \includegraphics[width=\paperwidth,height=\paperheight,keepaspectratio=false]{#1}%
        };%
      \end{tikzpicture}%
    }%
  }%
  % Professional cover layout with balanced spacing and visual hierarchy
  % All content must fit on one page - use compact spacing
  \begin{center}
    % Top section - Title area
    \vspace*{0.16\paperheight}%
    {\fontsize{68}{82}\selectfont\bfseries\textcolor{black}{Math for AI/ML}}\\[1.0cm]
    
    % Subtitle - elegant spacing below title
    {\fontsize{26}{32}\selectfont\textcolor{black}{A comprehensive mathematics textbook for AI/ML}}\\[2.5cm]
    
    % Middle section - Author (center area)
    {\fontsize{24}{30}\selectfont\textcolor{black}{MockSphere.com}}\\[1.2cm]
    
    % Bottom section - Edition and Year
    % Use fixed spacing to ensure everything fits on one page
    \vspace*{1.0cm}%
    {\fontsize{20}{26}\selectfont\textcolor{black}{Second Edition}}\\[0.3cm]
    {\fontsize{22}{28}\selectfont\bfseries\textcolor{black}{2026}}
    \vspace*{0.10\paperheight}%
  \end{center}
}
