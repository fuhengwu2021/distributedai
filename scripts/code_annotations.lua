-- Pandoc Lua filter to process code blocks with line annotations
-- This filter:
-- 1. Detects HTML comments in code blocks (e.g., <!--①--> or <!--1-->)
-- 2. Converts them to LaTeX commands for circled numbers
-- 3. Processes explanation blocks for proper formatting

function CodeBlock(block)
  local code = block.text
  local lang = block.classes[1] or ""
  
  -- Check for HTML comment annotations like <!--①--> or <!--1-->
  local annotation_pattern = "<!--([①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]|%d+)-->"
  
  -- Convert circled numbers to regular numbers
  local circled_to_num = {
    ["①"] = "1", ["②"] = "2", ["③"] = "3", ["④"] = "4", ["⑤"] = "5",
    ["⑥"] = "6", ["⑦"] = "7", ["⑧"] = "8", ["⑨"] = "9", ["⑩"] = "10",
    ["⑪"] = "11", ["⑫"] = "12", ["⑬"] = "13", ["⑭"] = "14", ["⑮"] = "15",
    ["⑯"] = "16", ["⑰"] = "17", ["⑱"] = "18", ["⑲"] = "19", ["⑳"] = "20"
  }
  
  -- Process each line
  local lines = {}
  for line in code:gmatch("[^\r\n]+") do
    local annotation_match = line:match(annotation_pattern)
    if annotation_match then
      -- Remove the annotation from the line
      local clean_line = line:gsub(annotation_pattern, ""):gsub("%s+$", "")
      -- Convert to number
      local num = circled_to_num[annotation_match] or annotation_match
      -- Add LaTeX command at the end
      line = clean_line .. " \\codelinemark{" .. num .. "}"
    end
    table.insert(lines, line)
  end
  
  -- Reconstruct code block
  block.text = table.concat(lines, "\n")
  return block
end

function RawInline(el)
  -- Process raw LaTeX in code explanations to ensure line breaks
  if el.format == "tex" then
    -- Replace \codelineannotation with proper line breaks
    el.text = el.text:gsub("\\codelineannotation", "\n\\codelineannotation")
  end
  return el
end

function RawBlock(el)
  -- Process raw LaTeX blocks (like codeexplanation environment)
  if el.format == "tex" then
    -- Ensure each \codelineannotation is on a new line
    -- Replace spaces before \codelineannotation with newline
    el.text = el.text:gsub("%s+\\codelineannotation", "\n\\codelineannotation")
    -- Also handle cases where \codelineannotation is at the start
    el.text = el.text:gsub("\\codelineannotation", "\n\\codelineannotation")
    -- Clean up multiple newlines
    el.text = el.text:gsub("\n\n+", "\n")
  end
  return el
end

