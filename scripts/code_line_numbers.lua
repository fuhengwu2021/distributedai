-- Pandoc Lua filter to add circled line numbers to code blocks
-- This filter converts code blocks to use listings package
-- Line numbers are added ONLY for Python code blocks
-- Other code blocks are rendered without line numbers

function CodeBlock(block)
  local code = block.text
  local lang = block.classes[1] or "text"
  
  -- Check for background color specification: #BKG:color or #BKG:color;NOLINENUM
  local background_color = "gray!5"  -- default
  local no_line_numbers = false
  local bg_pattern = "^#BKG:([^\r\n]+)"
  local bg_match = code:match(bg_pattern)
  if bg_match then
    -- Extract the full value (may contain color and NOLINENUM)
    local bg_value = bg_match:gsub("^%s+", ""):gsub("%s+$", "")
    -- Check for NOLINENUM in the value
    if bg_value:match(";NOLINENUM") or bg_value:match("NOLINENUM") then
      no_line_numbers = true
      -- Extract just the color part (before semicolon)
      background_color = bg_value:gsub(";NOLINENUM.*", ""):gsub("NOLINENUM.*", ""):gsub("^%s+", ""):gsub("%s+$", "")
      if background_color == "" then
        background_color = "gray!5"  -- default if no color specified
      end
    else
      background_color = bg_value
    end
    -- Remove the #BKG:color line from code (with optional newline after)
    code = code:gsub(bg_pattern .. "%s*\r?\n?", "", 1)
    code = code:gsub("^%s+", "")  -- Remove leading whitespace if any
  end
  
  -- Also check for standalone NOLINENUM marker
  local nolinenum_pattern = "^NOLINENUM%s*\r?\n?"
  if code:match(nolinenum_pattern) then
    no_line_numbers = true
    code = code:gsub(nolinenum_pattern, "", 1)
    code = code:gsub("^%s+", "")  -- Remove leading whitespace if any
  end
  
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
  
  -- Only add line numbers for Python code blocks (unless NOLINENUM is specified)
  local is_python = (lang:lower() == "python")
  local should_add_line_numbers = is_python and not no_line_numbers
  
  if is_python then
    -- Split code into lines
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
      
      if should_add_line_numbers then
        -- Add circled number mark at the beginning using escapeinside
        -- Use "solid" style if line should be highlighted (solid fill color)
        -- Add negative horizontal space to move circle closer to left border
        local mark_style = "normal"
        if should_highlight then
          mark_style = "solid"
        end
        local line_with_number = "(*@\\codelinemark{" .. mark_style .. "}{" .. tostring(line_num) .. "}\\hspace{-3pt}@*) " .. clean_line
        lines[line_num] = line_with_number
      else
        -- No line numbers, just use the clean line
        lines[line_num] = clean_line
      end
    end
    
    -- If no lines, return original block
    if line_num == 0 then
      return block
    end
    
    -- Use tcolorbox to wrap listings with border and rounded corners
    -- Apply custom background color if specified
    local tcolorbox_options = "codeblockstyle"
    local listings_bg_option = ""
    if background_color ~= "gray!5" then
      tcolorbox_options = "codeblockstyle,colback=" .. background_color
      listings_bg_option = "backgroundcolor=\\color{" .. background_color .. "},"
    end
    local listings_code = "\\begin{tcolorbox}[" .. tcolorbox_options .. "]\n"
    -- Only use escapeinside if we're adding line numbers
    local escapeinside_option = ""
    if should_add_line_numbers then
      escapeinside_option = "escapeinside={(*@}{@*)},"
    end
    listings_code = listings_code .. "\\begin{lstlisting}[" .. listings_lang .. listings_bg_option .. escapeinside_option .. "basicstyle=\\ttfamily\\normalsize,breaklines=true]\n"
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
    -- Apply custom background color if specified
    local tcolorbox_options = "codeblockstyle"
    local listings_bg_option = ""
    if background_color ~= "gray!5" then
      tcolorbox_options = "codeblockstyle,colback=" .. background_color
      listings_bg_option = "backgroundcolor=\\color{" .. background_color .. "},"
    end
    local listings_code = "\\begin{tcolorbox}[" .. tcolorbox_options .. "]\n"
    listings_code = listings_code .. "\\begin{lstlisting}[" .. style_option .. listings_lang .. listings_bg_option .. "basicstyle=\\ttfamily\\normalsize,breaklines=true]\n"
    listings_code = listings_code .. code
    listings_code = listings_code .. "\n\\end{lstlisting}\n"
    listings_code = listings_code .. "\\end{tcolorbox}"
    
    -- Return as RawBlock (LaTeX)
    return pandoc.RawBlock('latex', listings_code)
  end
end
