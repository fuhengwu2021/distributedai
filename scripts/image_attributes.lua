-- Pandoc Lua filter to handle image attributes for size, alignment, and placement
-- Supports:
--   - width/height: e.g., {width=50%}, {width=5cm}, {height=3in}
--   - alignment: {align=left}, {align=center}, {align=right}
--   - placement: {.block} for full-width block, {.inline} for inline with text
--   - wrap: {.wrap} to allow text wrapping around image
--
-- Usage in markdown (IMPORTANT: use .class syntax for placement modes):
--   ![alt](image.png){width=50%}
--   ![alt](image.png){width=5cm align=center}
--   ![alt](image.png){.block width=80% align=center}
--   ![alt](image.png){.inline width=30% align=left}
--   ![alt](image.png){.wrap width=40% align=right}

function Image(img)
  local attr = img.attr
  -- Remove alt attribute - LaTeX includegraphics doesn't support it
  -- Pandoc 3.8+ automatically adds alt from the markdown alt text, but this causes LaTeX errors
  if attr.attributes.alt then
    attr.attributes.alt = nil
  end
  
  local width = attr.attributes.width
  local height = attr.attributes.height
  local align = attr.attributes.align or attr.attributes.alignment
  local vspace = attr.attributes.vspace or attr.attributes["vertical-space"] or "10pt"  -- Default vertical spacing
  local lines = attr.attributes.lines  -- Optional: number of lines for wrapfigure (limits wrapping scope)
  local block = false
  local inline = false
  local wrap = false
  
  -- Check for block/inline/wrap in classes or attributes
  for _, class in ipairs(attr.classes) do
    if class == "block" then
      block = true
    elseif class == "inline" then
      inline = true
    elseif class == "wrap" then
      wrap = true
    end
  end
  
  -- Check attributes for block/inline/wrap
  if attr.attributes.block == "true" or attr.attributes.block == true then
    block = true
  end
  if attr.attributes.inline == "true" or attr.attributes.inline == true then
    inline = true
  end
  if attr.attributes.wrap == "true" or attr.attributes.wrap == true then
    wrap = true
  end
  
  -- Default: if no explicit placement, keep original Pandoc behavior
  -- (Pandoc will handle it normally, which typically results in block placement)
  if not block and not inline and not wrap then
    -- Don't override - let Pandoc handle it with default settings
    -- This allows images without attributes to work normally
    -- (alt attribute already removed at the start of the function)
    return img
  end
  
  -- Build LaTeX code based on attributes
  local latex_code = ""
  
  -- Handle wrap mode (text wrapping around image)
  if wrap then
    local width_str = width or "0.4\\textwidth"
    -- Convert percentage to LaTeX format if needed
    if width_str:match("%%$") then
      local percent = tonumber(width_str:match("^(%d+)%%$"))
      if percent then
        width_str = string.format("%.2f\\textwidth", percent / 100)
      end
    end
    
    -- Parse alignment - support combinations like "top-right", "bottom-left", etc.
    local align_str = "r"  -- default right alignment for wrap
    local vspace_before = ""
    local vspace_after = ""
    local float_to_top = false
    
    if align then
      local align_lower = string.lower(align)
      -- Check for vertical positioning (vspace only applies when top/bottom is specified)
      if string.find(align_lower, "top") then
        float_to_top = true
        -- Use negative vspace to push to top
        local vspace_value = "-" .. vspace
        vspace_before = string.format("\\vspace{%s}\n", vspace_value)
        vspace_after = string.format("\n\\vspace{%s}", vspace_value)
      elseif string.find(align_lower, "bottom") then
        -- Use positive vspace to push to bottom
        vspace_before = string.format("\\vspace{%s}\n", vspace)
        vspace_after = string.format("\n\\vspace{%s}", vspace)
      end
      -- Note: vspace is ignored for plain left/right/center (without top/bottom)
      
      -- Check for horizontal positioning
      if string.find(align_lower, "right") or align_lower == "r" then
        align_str = float_to_top and "R" or "r"
      elseif string.find(align_lower, "left") or align_lower == "l" then
        align_str = float_to_top and "L" or "l"
      elseif string.find(align_lower, "center") or align_lower == "c" then
        align_str = float_to_top and "C" or "c"
      elseif align == "left" then
        align_str = "l"
      elseif align == "center" then
        align_str = "c"
      end
    end
    
    -- Extract caption from cap attribute or title (alt text)
    local caption = ""
    -- First check for cap attribute
    if attr.attributes.cap then
      caption = attr.attributes.cap
    elseif img.title and #img.title > 0 then
      -- Fallback to title (alt text) if cap attribute not present
      local caption_parts = {}
      for _, elem in ipairs(img.title) do
        if elem.t == "Str" then
          table.insert(caption_parts, elem.c)
        end
      end
      caption = table.concat(caption_parts, " ")
    end
    
    local caption_latex = ""
    if caption and caption ~= "" then
      caption_latex = string.format("\\caption{%s}", caption)
    end
    
    -- For wrapfigure, image width should be slightly smaller than wrapfigure width
    -- This allows text to properly wrap around the image
    local img_width = width_str
    -- Check if width_str contains \textwidth (e.g., "0.54\textwidth")
    if width_str:match("\\textwidth") then
      -- Extract the number before \textwidth
      local num_match = width_str:match("^(%d+%.?%d*)\\textwidth")
      if num_match then
        local num_val = tonumber(num_match)
        if num_val then
          img_width = string.format("%.2f\\textwidth", num_val * 0.95)
        end
      end
    elseif width_str:match("%%$") then
      -- Percentage: reduce by 5%
      local percent = tonumber(width_str:match("^(%d+)%%$"))
      if percent then
        img_width = string.format("%.0f%%", percent * 0.95)
      end
    end
    -- Build wrapfigure command with optional lines parameter
    local lines_param = ""
    if lines then
      lines_param = string.format("[%s]", lines)
    end
    -- Store wrap image info in a custom attribute for Para processing
    img.attr.attributes._wrap_image = "true"
    img.attr.attributes._wrap_latex = string.format(
      "\\begin{wrapfigure}%s{%s}{%s}%s\\centering\n\\includegraphics[width=%s,keepaspectratio]{%s}\n%s%s\\end{wrapfigure}",
      lines_param, align_str, width_str, vspace_before, img_width, img.src, caption_latex, vspace_after
    )
    -- Return the image as-is; Para filter will handle the conversion
    return img
  end
  
  -- Handle block mode (full-width or centered block)
  if block then
    local width_str = width or "\\maxwidth"
    local height_str = ""
    if height then
      height_str = string.format("height=%s,", height)
    end
    
    local align_cmd = "\\centering"
    if align == "left" then
      align_cmd = "\\raggedright"
    elseif align == "right" then
      align_cmd = "\\raggedleft"
    end
    
    -- Extract caption from cap attribute or title (alt text)
    local caption = ""
    -- First check for cap attribute
    if attr.attributes.cap then
      caption = attr.attributes.cap
    elseif img.title and #img.title > 0 then
      -- Fallback to title (alt text) if cap attribute not present
      local caption_parts = {}
      for _, elem in ipairs(img.title) do
        if elem.t == "Str" then
          table.insert(caption_parts, elem.c)
        end
      end
      caption = table.concat(caption_parts, " ")
    end
    
    local caption_latex = ""
    if caption and caption ~= "" then
      caption_latex = string.format("\\caption{%s}", caption)
    end
    
    -- Store block image info in a custom attribute for Para processing
    img.attr.attributes._block_image = "true"
    img.attr.attributes._block_latex = string.format(
      "\\begin{figure}[H]\n%s\n\\includegraphics[width=%s,%skeepaspectratio]{%s}\n%s\n\\end{figure}",
      align_cmd, width_str, height_str, img.src, caption_latex
    )
    -- Return the image as-is; Para filter will handle the conversion
    return img
  end
  
  -- Handle inline mode (image flows with text)
  if inline then
    local width_str = width or "0.3\\textwidth"
    local height_str = ""
    if height then
      height_str = string.format("height=%s,", height)
    end
    
    local align_cmd = ""
    if align == "left" then
      align_cmd = "\\raggedright"
    elseif align == "right" then
      align_cmd = "\\raggedleft"
    elseif align == "center" then
      align_cmd = "\\centering"
    end
    
    if align_cmd ~= "" then
      latex_code = string.format(
        "%s\\includegraphics[width=%s,%skeepaspectratio]{%s}",
        align_cmd, width_str, height_str, img.src
      )
    else
      latex_code = string.format(
        "\\includegraphics[width=%s,%skeepaspectratio]{%s}",
        width_str, height_str, img.src
      )
    end
    return pandoc.RawInline('latex', latex_code)
  end
  
  -- Fallback: return original image (shouldn't reach here)
  return img
end

-- Process Figure blocks that contain images with wrap/block attributes
-- Helper function to extract text from caption blocks
local function extract_caption_text(caption)
  if not caption or not caption.long or #caption.long == 0 then
    return ""
  end
  
  -- Use pandoc.utils.stringify to extract plain text from caption blocks
  local caption_blocks = {}
  for _, block in ipairs(caption.long) do
    table.insert(caption_blocks, block)
  end
  if #caption_blocks == 0 then
    return ""
  end
  
  -- Convert caption blocks to plain text
  local doc = pandoc.Pandoc(caption_blocks)
  local plain_text = pandoc.utils.stringify(doc)
  -- Clean up whitespace
  plain_text = plain_text:gsub("%s+", " "):gsub("^%s+", ""):gsub("%s+$", "")
  return plain_text
end

function Figure(fig)
  -- Check if figure contains an image with wrap/block attributes
  local block_latex = nil
  local wrap_latex = nil
  
  -- Extract caption from Figure (alt text from markdown)
  local caption_text = extract_caption_text(fig.caption)
  local caption_latex = ""
  if caption_text and caption_text ~= "" then
    caption_latex = string.format("\\caption{%s}", caption_text)
  end
  
  -- Figure structure: {attr, caption, content}
  -- content is a list of blocks, typically containing a Plain or Para with an Image
  for _, block in ipairs(fig.content) do
    -- Handle both Plain and Para blocks
    if block.t == "Para" or block.t == "Plain" then
      for _, inline in ipairs(block.c) do
        if inline.t == "Image" then
          local attr = inline.attr
          if attr.attributes._block_image == "true" then
            block_latex = attr.attributes._block_latex
            -- Add caption if not already present
            if caption_latex ~= "" and not block_latex:match("\\caption") then
              block_latex = block_latex:gsub("(\\end{figure})", caption_latex .. "\n%1")
            end
          elseif attr.attributes._wrap_image == "true" then
            wrap_latex = attr.attributes._wrap_latex
            -- Add caption if not already present
            if caption_latex ~= "" and not wrap_latex:match("\\caption") then
              -- Insert caption before \end{wrapfigure}
              wrap_latex = wrap_latex:gsub("(\\end{wrapfigure})", caption_latex .. "\n%1")
            end
          end
        end
      end
    end
  end
  
  -- Convert to appropriate LaTeX
  if wrap_latex then
    return pandoc.RawBlock('latex', wrap_latex)
  elseif block_latex then
    return pandoc.RawBlock('latex', block_latex)
  end
  
  -- Return figure unchanged if no special attributes
  return fig
end

-- Process Para blocks that contain only block/wrap images (for inline images)
function Para(para)
  -- Check if para contains only one image (possibly with spaces/line breaks)
  local image_count = 0
  local block_latex = nil
  local wrap_latex = nil
  local has_other_content = false
  
  for _, inline in ipairs(para.c) do
    if inline.t == "Image" then
      image_count = image_count + 1
      local attr = inline.attr
      if attr.attributes._block_image == "true" then
        block_latex = attr.attributes._block_latex
      elseif attr.attributes._wrap_image == "true" then
        wrap_latex = attr.attributes._wrap_latex
      end
    elseif inline.t ~= "Space" and inline.t ~= "SoftBreak" and inline.t ~= "LineBreak" then
      -- Contains something other than image or whitespace
      has_other_content = true
    end
  end
  
  -- Only convert if there's exactly one image with block/wrap attribute and no other content
  if image_count == 1 and not has_other_content then
    if block_latex then
      return pandoc.RawBlock('latex', block_latex)
    elseif wrap_latex then
      -- For wrapfigure, we need to keep it in the paragraph context
      -- Don't convert to RawBlock - let it stay as Para so it can be merged with following text
      -- Instead, we'll insert the wrapfigure at the start of the paragraph content
      -- This allows wrapfigure to work correctly when followed by text
      return pandoc.RawBlock('latex', wrap_latex)
    end
  end
  
  -- Return para unchanged
  return para
end
