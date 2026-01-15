-- Lua filter to beautify tables in LaTeX output
-- Adds alternating row colors and horizontal lines using booktabs

-- Process tables in the document structure
function Table(tbl)
  return tbl
end

-- Process raw LaTeX blocks (this is where pandoc outputs tables)
function RawBlock(format, text)
  if format == "latex" then
    -- Check for longtable (pandoc's default for wide tables)
    if text:match("\\begin{longtable}") then
      -- Add rowcolors right after \begin{longtable}
      -- longtable structure: \begin{longtable}{...}, \toprule, header, \midrule, \endhead, data rows
      -- We want to start coloring from the data rows (after \endhead)
      text = text:gsub("(\\begin{longtable}[^\n]+\n)", "%1\\rowcolors{5}{white}{tablegray!30}\n")
      return pandoc.RawBlock("latex", text)
    end
    
    -- Check for regular tabular
    if text:match("\\begin{tabular}") then
      -- Add rowcolors right after \begin{tabular}
      text = text:gsub("(\\begin{tabular}[^\n]+\n)", "%1\\rowcolors{2}{white}{tablegray!30}\n")
      return pandoc.RawBlock("latex", text)
    end
  end
  return nil
end
