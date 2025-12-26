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
    -- Split code into lines and add circled number at the end of each line
    local lines = {}
    local line_num = 0
    for line in code:gmatch("[^\r\n]+") do
      line_num = line_num + 1
      -- Add circled number mark at the end using escapeinside
      lines[line_num] = line .. " (*@\\codelinemark{" .. tostring(line_num) .. "}@*)"
    end
    
    -- If no lines, return original block
    if line_num == 0 then
      return block
    end
    
    -- Use mdframed to wrap listings with border, and escapeinside for line numbers
    local listings_code = "\\begin{mdframed}[style=codeblockstyle]\n"
    listings_code = listings_code .. "\\begin{lstlisting}[" .. listings_lang .. "escapeinside={(*@}{@*)},basicstyle=\\ttfamily\\normalsize,breaklines=true]\n"
    listings_code = listings_code .. table.concat(lines, "\n")
    listings_code = listings_code .. "\n\\end{lstlisting}\n"
    listings_code = listings_code .. "\\end{mdframed}"
    
    -- Return as RawBlock (LaTeX)
    return pandoc.RawBlock('latex', listings_code)
  else
    -- For non-Python code blocks, use regular listings without line numbers
    -- Apply bash-specific style if it's a bash/shell code block
    local style_option = ""
    if lang:lower() == "bash" or lang:lower() == "sh" or lang:lower() == "shell" then
      style_option = "style=bashstyle,"
    end
    local listings_code = "\\begin{mdframed}[style=codeblockstyle]\n"
    listings_code = listings_code .. "\\begin{lstlisting}[" .. style_option .. listings_lang .. "basicstyle=\\ttfamily\\normalsize,breaklines=true]\n"
    listings_code = listings_code .. code
    listings_code = listings_code .. "\n\\end{lstlisting}\n"
    listings_code = listings_code .. "\\end{mdframed}"
    
    -- Return as RawBlock (LaTeX)
    return pandoc.RawBlock('latex', listings_code)
  end
end
