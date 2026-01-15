-- Pandoc Lua filter to add equation numbering and cross-references
-- This filter:
-- 1. Detects display math blocks ($$...$$) with \label{eq:...} and converts them to LaTeX equation environment
-- 2. Processes equation references in text (e.g., @eq:name or \eqref{eq:name}) and converts them to proper LaTeX references

-- Helper function to check if a Para contains only a single DisplayMath with label
local function is_equation_para(para)
  if para.t ~= "Para" or #para.c ~= 1 then
    return false, nil, nil
  end
  
  local inline = para.c[1]
  if inline.t == "Math" and inline.mathtype == "DisplayMath" then
    local math_text = inline.text
    -- Check if the math contains \label{eq:...}
    local label_pattern = "\\label{(eq:[^}]+)}"
    local label_match = math_text:match(label_pattern)
    
    if label_match then
      -- Extract the label name
      local label_name = label_match
      -- Remove the \label from the math content (handle various positions)
      -- Escape special characters for pattern matching
      local escaped_label = label_name:gsub("%%", "%%%%"):gsub("([%(%)%.%+%-%*%?%[%]%^%$])", "%%%1")
      -- Remove \label{eq:...} with optional whitespace before/after (multiple times to catch all occurrences)
      local math_content = math_text
      -- Try multiple patterns to ensure complete removal
      math_content = math_content:gsub("%s*\\label{" .. escaped_label .. "}%s*", "")
      math_content = math_content:gsub("\\label{" .. escaped_label .. "}%s*", "")
      math_content = math_content:gsub("%s*\\label{" .. escaped_label .. "}", "")
      -- Remove any trailing whitespace and newlines
      math_content = math_content:gsub("^%s+", ""):gsub("%s+$", "")  -- Trim whitespace
      return true, math_content, label_name
    end
  end
  
  return false, nil, nil
end

-- Helper function to process inlines for equation references
local function process_inlines_for_refs(inlines)
  local new_inlines = {}
  local i = 1
  
  while i <= #inlines do
    local inline = inlines[i]
    
    -- Check if this is a Cite element with an equation reference
    if inline.t == "Cite" and inline.citations and #inline.citations > 0 then
      local citation = inline.citations[1]
      local citation_id = citation.id
      
      -- Check if it's an equation reference (starts with "eq:")
      if citation_id:match("^eq:") then
        -- Convert to LaTeX \eqref with italic number (remove "equation" text, use italic number)
        -- Use \textit{\ref{...}} to get italic number without "equation" prefix
        local latex_ref = "\\textit{\\ref{" .. citation_id .. "}}"
        table.insert(new_inlines, pandoc.RawInline('latex', latex_ref))
      else
        -- Not an equation reference, keep original
        table.insert(new_inlines, inline)
      end
    elseif inline.t == "Str" then
      -- Also check for @eq:name pattern in Str elements (fallback for cases where Pandoc doesn't parse as Cite)
      local text = pandoc.utils.stringify(inline)
      
      -- Check for @eq:name pattern and process it
      local processed = false
      local parts = {}
      local last_pos = 1
      
      for match_start, match_end, label in text:gmatch("()@(eq:[%w_%-]+)()") do
        processed = true
        -- Add text before the match
        if match_start > last_pos then
          local before_text = text:sub(last_pos, match_start - 1)
          if before_text ~= "" then
            table.insert(parts, {type = "text", content = before_text})
          end
        end
        
        -- Add the reference
        table.insert(parts, {type = "ref", content = label})
        
        last_pos = match_end
      end
      
      -- Add remaining text after last match
      if processed then
        if last_pos <= #text then
          local after_text = text:sub(last_pos)
          if after_text ~= "" then
            table.insert(parts, {type = "text", content = after_text})
          end
        end
        
        -- Rebuild the inlines with processed parts
        for _, part in ipairs(parts) do
          if part.type == "text" then
            table.insert(new_inlines, pandoc.Str(part.content))
          elseif part.type == "ref" then
            -- Convert @eq:name to italic reference number (remove "equation" text)
            local latex_ref = "\\textit{\\ref{" .. part.content .. "}}"
            table.insert(new_inlines, pandoc.RawInline('latex', latex_ref))
          end
        end
      else
        -- No reference pattern found or not processed, keep original
        table.insert(new_inlines, inline)
      end
    else
      -- Not a Cite or Str element, keep as is
      table.insert(new_inlines, inline)
    end
    
    i = i + 1
  end
  
  return new_inlines
end

-- Process paragraphs: check if it's an equation, otherwise process references
function Para(el)
  -- Check if this is an equation paragraph
  local is_eq, math_content, label_name = is_equation_para(el)
  if is_eq then
    -- Check if math_content contains \begin{aligned} or \begin{align}
    -- If it does, we need to insert \label after \end{aligned} or \end{align} but before closing
    if math_content:match("\\begin{aligned}") or math_content:match("\\begin{align}") then
      -- Insert \label after the closing aligned/align tag but before any closing bracket
      -- Remove any existing \label that might be there
      -- Use a more robust pattern that handles both aligned and align
      local latex_code = math_content
      -- Replace \end{aligned} with \end{aligned}\n\label{...}
      latex_code = latex_code:gsub("(\\end{aligned})", "%1\n\\label{" .. label_name .. "}")
      -- Also handle \end{align} if present
      latex_code = latex_code:gsub("(\\end{align})", "%1\n\\label{" .. label_name .. "}")
      -- Wrap in equation environment
      latex_code = "\\begin{equation}\n" .. latex_code .. "\n\\end{equation}"
      return pandoc.RawBlock('latex', latex_code)
    elseif math_content:match("\\begin{equation}") then
      -- Already in equation environment, just insert \label before \end{equation}
      local latex_code = math_content:gsub("(\\end{equation})", "\\label{" .. label_name .. "}\n%1")
      return pandoc.RawBlock('latex', latex_code)
    else
      -- Simple math content, wrap in equation environment
      local latex_code = "\\begin{equation}\n" .. math_content .. "\n\\label{" .. label_name .. "}\n\\end{equation}"
      return pandoc.RawBlock('latex', latex_code)
    end
  end
  
  -- Otherwise, process the paragraph for equation references
  local new_inlines = process_inlines_for_refs(el.c)
  return pandoc.Para(new_inlines)
end

-- Also process other block types that might contain text with references
function Plain(el)
  -- Process Plain blocks for equation references
  local new_inlines = process_inlines_for_refs(el.c)
  return pandoc.Plain(new_inlines)
end
