-- Pandoc Lua filter to add circled line numbers to code blocks
-- This filter converts code blocks to use listings package
-- Line numbers are added ONLY for Python code blocks
-- Other code blocks are rendered without line numbers

function CodeBlock(block)
  local code = block.text
  local lang = block.classes[1] or "text"
  
  -- Map common languages to listings language names, or use empty for unknown
  local listings_lang = ""
  local lang_map = {
    python = "Python",
    bash = "bash",
    sh = "bash",
    javascript = "JavaScript",
    java = "Java",
    cpp = "C++",
    c = "C",
    go = "Go",
    rust = "Rust",
    sql = "SQL",
    html = "HTML",
    css = "CSS",
    json = "JSON",
    yaml = "YAML",
    xml = "XML"
  }
  if lang_map[lang:lower()] then
    listings_lang = "language=" .. lang_map[lang:lower()] .. ","
  end
  
  -- Only add line numbers for Python code blocks
  local is_python = (lang:lower() == "python")
  
  if is_python then
    -- Split code into lines and add circled number at the beginning of each line
    local lines = {}
    local line_num = 0
    for line in code:gmatch("[^\r\n]+") do
      line_num = line_num + 1
      -- Check if line has #HL marker for highlighting
      local should_highlight = false
      local clean_line = line
      -- Check for #HL at the end of line (with optional whitespace)
      if line:match("#HL") then
        should_highlight = true
        -- Remove #HL marker from the line (handle with or without space before)
        clean_line = line:gsub("%s*#HL%s*", "")
      end
      
      -- Add circled number mark at the beginning using escapeinside
      -- Use "solid" style if line should be highlighted (solid fill color)
      -- Add negative horizontal space to move circle closer to left border
      local mark_style = "normal"
      if should_highlight then
        mark_style = "solid"
      end
      local line_with_number = "(*@\\codelinemark{" .. mark_style .. "}{" .. tostring(line_num) .. "}\\hspace{-3pt}@*) " .. clean_line
      
      lines[line_num] = line_with_number
    end
    
    -- If no lines, return original block
    if line_num == 0 then
      return block
    end
    
    -- Use tcolorbox to wrap listings with border and rounded corners, and escapeinside for line numbers
    local listings_code = "\\begin{tcolorbox}[codeblockstyle]\n"
    listings_code = listings_code .. "\\begin{lstlisting}[" .. listings_lang .. "escapeinside={(*@}{@*)},basicstyle=\\ttfamily\\normalsize,breaklines=true]\n"
    listings_code = listings_code .. table.concat(lines, "\n")
    listings_code = listings_code .. "\n\\end{lstlisting}\n"
    listings_code = listings_code .. "\\end{tcolorbox}"
    
    -- Return as RawBlock (LaTeX)
    return pandoc.RawBlock('latex', listings_code)
  else
    -- For non-Python code blocks, use regular listings without line numbers
    -- Apply bash-specific style if it's a bash/shell code block
    local style_option = ""
    if lang:lower() == "bash" or lang:lower() == "sh" or lang:lower() == "shell" then
      style_option = "style=bashstyle,"
    end
    local listings_code = "\\begin{tcolorbox}[codeblockstyle]\n"
    listings_code = listings_code .. "\\begin{lstlisting}[" .. style_option .. listings_lang .. "basicstyle=\\ttfamily\\normalsize,breaklines=true]\n"
    listings_code = listings_code .. code
    listings_code = listings_code .. "\n\\end{lstlisting}\n"
    listings_code = listings_code .. "\\end{tcolorbox}"
    
    -- Return as RawBlock (LaTeX)
    return pandoc.RawBlock('latex', listings_code)
  end
end
