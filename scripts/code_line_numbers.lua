-- Pandoc Lua filter to add circled line numbers to code blocks
-- This filter converts code blocks to use listings package
-- Line numbers are added ONLY for Python code blocks
-- Other code blocks are rendered without line numbers

function CodeBlock(block)
  local code = block.text
  local lang = block.classes[1] or "text"
  
  -- Check for background color specification: #BKG:color or #BKG:color;#LINENUM or #BKG:color;#LINENUM;#LINEBAR
  local background_color = "gray!5"  -- default
  local show_line_numbers = false  -- Default: no line numbers
  local use_linebar_from_bkg = false  -- Track if #LINEBAR is in #BKG line
  local use_linebar_from_linenum = false  -- Track if #LINEBAR was found after #LINENUM
  local bg_pattern = "^#BKG:([^\r\n]+)"
  local bg_match = code:match(bg_pattern)
  if bg_match then
    -- Extract the full value (may contain color, #LINENUM, and #LINEBAR)
    local bg_value = bg_match:gsub("^%s+", ""):gsub("%s+$", "")
    -- Check for #LINEBAR in the value
    if bg_value:match(";#LINEBAR") or bg_value:match("#LINEBAR") then
      use_linebar_from_bkg = true
      -- Remove #LINEBAR from bg_value
      bg_value = bg_value:gsub(";#LINEBAR.*", ""):gsub("#LINEBAR.*", ""):gsub("^%s+", ""):gsub("%s+$", "")
    end
    -- Check for #LINENUM in the value
    if bg_value:match(";#LINENUM") or bg_value:match("#LINENUM") then
      show_line_numbers = true
      -- Extract just the color part (before semicolon)
      background_color = bg_value:gsub(";#LINENUM.*", ""):gsub("#LINENUM.*", ""):gsub("^%s+", ""):gsub("%s+$", "")
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
  
  -- Check for standalone #LINENUM marker (enables line numbers)
  local linenum_pattern = "^#LINENUM%s*\r?\n?"
  if code:match(linenum_pattern) then
    show_line_numbers = true
    code = code:gsub(linenum_pattern, "", 1)
    code = code:gsub("^%s+", "")  -- Remove leading whitespace if any
    -- After removing #LINENUM, check if the next line is #LINEBAR and remove it too
    -- Handle #LINEBAR with optional #HL after it (added by process_code_explain.py)
    local linebar_after_linenum = "^%s*#LINEBAR%s*"
    if code:match(linebar_after_linenum) then
      use_linebar_from_linenum = true
      -- Remove #LINEBAR line (may have #HL after it)
      code = code:gsub("^%s*#LINEBAR%s*#HL%s*\r?\n?", "", 1)
      code = code:gsub("^%s*#LINEBAR%s*\r?\n?", "", 1)
      code = code:gsub("^%s+", "")  -- Remove leading whitespace if any
    end
  end
  
  -- Check for standalone NOLINENUM marker (disables line numbers, overrides LINENUM)
  local nolinenum_pattern = "^NOLINENUM%s*\r?\n?"
  if code:match(nolinenum_pattern) then
    show_line_numbers = false
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
  
  -- Only add line numbers for Python code blocks if explicitly enabled
  local is_python = (lang:lower() == "python")
  local should_add_line_numbers = is_python and show_line_numbers
  
  -- Check for #LINEBAR in first line comment (only if line numbers are enabled)
  -- This must be checked before processing #HL markers
  -- Also check if #LINEBAR was found in #BKG line or after #LINENUM
  local use_linebar = use_linebar_from_bkg or use_linebar_from_linenum
  if should_add_line_numbers then
    if not use_linebar then
      local first_line = code:match("^([^\r\n]+)")
      if first_line then
        -- Check if first line is just #LINEBAR (standalone line)
        if first_line:match("^%s*#LINEBAR%s*$") then
          use_linebar = true
          -- Remove the entire #LINEBAR line
          code = code:gsub("^%s*#LINEBAR%s*\r?\n?", "", 1)
          code = code:gsub("^%s+", "")  -- Remove leading whitespace if any
        -- Check if #LINEBAR is in a comment
        elseif first_line:match("^%s*#") and first_line:match("#LINEBAR") then
          use_linebar = true
          -- Remove #LINEBAR from first line (handle various positions and #HL that might be added later)
          -- Try different patterns to catch #LINEBAR in various positions
          code = code:gsub("^([^\r\n]*)%s*#LINEBAR%s*([^\r\n]*)", "%1%2", 1)
          code = code:gsub("^([^\r\n]*)#HL%s*#LINEBAR%s*([^\r\n]*)", "%1#HL%2", 1)
          code = code:gsub("^([^\r\n]*)#LINEBAR%s*#HL%s*([^\r\n]*)", "%1#HL%2", 1)
          -- Clean up any double spaces
          code = code:gsub("^([^\r\n]*)  +([^\r\n]*)", "%1 %2", 1)
          -- Remove trailing spaces from first line
          code = code:gsub("^([^\r\n]+)%s+(\r?\n)", "%1%2", 1)
          -- Final cleanup: remove any remaining #LINEBAR (in case previous patterns missed it)
          code = code:gsub("#LINEBAR%s*", "", 1)
        end
      end
    end
  end
  
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
        if use_linebar then
          -- Use bar style: number with vertical bar (e.g., "13|" or " 1|")
          local num_str = tostring(line_num)
          -- Right-align numbers (pad with space for single digits)
          if line_num < 10 then
            num_str = " " .. num_str
          end
          local mark_style = "normal"
          if should_highlight then
            mark_style = "solid"
          end
          local line_with_number = "(*@\\codelinemarkbar{" .. mark_style .. "}{" .. tostring(line_num) .. "}@*) " .. clean_line
          lines[line_num] = line_with_number
        else
          -- Add circled number mark at the beginning using escapeinside
          -- Use "solid" style if line should be highlighted (solid fill color)
          -- Add negative horizontal space to move circle closer to left border
          local mark_style = "normal"
          if should_highlight then
            mark_style = "solid"
          end
          local line_with_number = "(*@\\codelinemark{" .. mark_style .. "}{" .. tostring(line_num) .. "}@*) " .. clean_line
          lines[line_num] = line_with_number
        end
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
    local tcolorbox_options = "codeblockstyle,breakable,enhanced"
    local listings_bg_option = ""
    if background_color ~= "gray!5" then
      tcolorbox_options = "codeblockstyle,breakable,enhanced,colback=" .. background_color
      listings_bg_option = "backgroundcolor=\\color{" .. background_color .. "},"
    end
    
    -- When we have escape sequences (line numbers), we can't use listing only
    -- because escape sequences don't work well with listing only
    -- Instead, use manual lstlisting which supports breakable tcolorbox
    local escapeinside_option = ""
    if should_add_line_numbers then
      escapeinside_option = "escapeinside={(*@}{@*)},"
    end
    
    local listings_code = "\\begin{tcolorbox}[" .. tcolorbox_options .. "]\n"
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
    local tcolorbox_options = "codeblockstyle,breakable,enhanced"
    local listings_bg_option = ""
    if background_color ~= "gray!5" then
      tcolorbox_options = "codeblockstyle,breakable,enhanced,colback=" .. background_color
      listings_bg_option = "backgroundcolor=\\color{" .. background_color .. "},"
    end
    -- Use manual lstlisting inside breakable tcolorbox for better compatibility
    local listings_code = "\\begin{tcolorbox}[" .. tcolorbox_options .. "]\n"
    listings_code = listings_code .. "\\begin{lstlisting}[" .. style_option .. listings_lang .. listings_bg_option .. "basicstyle=\\ttfamily\\normalsize,breaklines=true]\n"
    listings_code = listings_code .. code
    listings_code = listings_code .. "\n\\end{lstlisting}\n"
    listings_code = listings_code .. "\\end{tcolorbox}"
    
    -- Return as RawBlock (LaTeX)
    return pandoc.RawBlock('latex', listings_code)
  end
end
