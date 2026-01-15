-- Pandoc Lua filter to process note sections marked with >NOTES: and >NOTEE:
-- This filter converts note sections into a special LaTeX environment with icon

-- Helper function to extract text from inline elements
-- Uses pandoc.utils.stringify for proper text extraction in Pandoc 3.x
local function extract_text(inlines)
  if not inlines then
    return ""
  end
  -- Use pandoc.utils.stringify which handles all inline types correctly
  local para = pandoc.Para(inlines)
  local text = pandoc.utils.stringify(para)
  return text
end

-- Helper function to check if a block quote starts with NOTES:
local function starts_with_notes(block)
  if not block or block.t ~= "BlockQuote" or not block.c or #block.c == 0 then
    return false
  end
  local first_block = block.c[1]
  if first_block and first_block.t == "Para" and first_block.c then
    local text = extract_text(first_block.c)
    return text:match("^NOTES:") ~= nil
  end
  return false
end

-- Helper function to check if a block quote ends with NOTEE
local function ends_with_notee(block)
  if block.t ~= "BlockQuote" or #block.c == 0 then
    return false
  end
  local last_block = block.c[#block.c]
  if last_block.t == "Para" then
    local text = extract_text(last_block.c)
    return text:match("NOTEE%s*$")
  end
  return false
end

-- Helper function to check if a block quote starts with NOTEE (for standalone >NOTEE)
local function starts_with_notee(block)
  if not block or block.t ~= "BlockQuote" or not block.c or #block.c == 0 then
    return false
  end
  local first_block = block.c[1]
  if first_block and first_block.t == "Para" and first_block.c then
    local text = extract_text(first_block.c)
    -- Trim whitespace and check for exact match
    text = text:gsub("^%s+", ""):gsub("%s+$", "")
    return text == "NOTEE"
  end
  return false
end

-- Helper function to remove NOTES: from the first block
local function remove_notes_marker(block)
  if block.t ~= "BlockQuote" or #block.c == 0 then
    return block
  end
  local first_block = block.c[1]
  if first_block.t == "Para" then
    local text = extract_text(first_block.c)
    if text:match("^NOTES:") then
      -- Remove NOTES: from the beginning
      local new_inlines = {}
      local found_notes = false
      local found_colon = false
      local skip_next_space = false
      
      for i, inline in ipairs(first_block.c) do
        if inline.t == "Str" then
          -- Extract string content from Pandoc 3.x Str element
          local str = pandoc.utils.stringify(inline)
          if not found_notes then
            if str:match("^NOTES") then
              found_notes = true
              local remaining = str:gsub("^NOTES", "")
              if remaining:match("^:") then
                remaining = remaining:gsub("^:", "")
                found_colon = true
                skip_next_space = true
              end
              if remaining ~= "" then
                table.insert(new_inlines, pandoc.Str(remaining))
              end
            else
              table.insert(new_inlines, inline)
            end
          elseif not found_colon then
            if str == ":" then
              found_colon = true
              skip_next_space = true
            else
              table.insert(new_inlines, inline)
            end
          else
            -- After NOTES:, skip spaces and add the rest
            if inline.t ~= "Space" or not skip_next_space then
              table.insert(new_inlines, inline)
            end
            skip_next_space = false
          end
        elseif inline.t == "Space" then
          if not skip_next_space then
            table.insert(new_inlines, inline)
          end
          skip_next_space = false
        else
          table.insert(new_inlines, inline)
        end
      end
      
      if #new_inlines > 0 then
        block.c[1] = pandoc.Para(new_inlines)
      else
        -- Remove the first block if it's empty
        table.remove(block.c, 1)
      end
    end
  end
  return block
end

-- Helper function to remove NOTEE from the last block
local function remove_notee_marker(block)
  if block.t ~= "BlockQuote" or #block.c == 0 then
    return block
  end
  local last_block = block.c[#block.c]
  if last_block.t == "Para" then
    local text = extract_text(last_block.c)
    if text:match("NOTEE%s*$") then
      -- Remove NOTEE from the end
      local new_inlines = {}
      local found_notee = false
      
      for i = #last_block.c, 1, -1 do
        local inline = last_block.c[i]
        if inline.t == "Str" and not found_notee then
          -- Extract string content from Pandoc 3.x Str element
          local str = pandoc.utils.stringify(inline)
          if str:match("NOTEE%s*$") then
            found_notee = true
            local remaining = str:gsub("NOTEE%s*$", "")
            if remaining ~= "" then
              table.insert(new_inlines, 1, pandoc.Str(remaining))
            end
          else
            table.insert(new_inlines, 1, inline)
          end
        elseif inline.t == "Space" and not found_notee then
          -- Skip trailing spaces before NOTEE
          -- (we'll handle this by checking the previous string)
        else
          if not found_notee then
            table.insert(new_inlines, 1, inline)
          end
        end
      end
      
      if #new_inlines > 0 then
        block.c[#block.c] = pandoc.Para(new_inlines)
      else
        -- Remove the last block if it's empty
        table.remove(block.c)
      end
    end
  end
  return block
end

function Pandoc(doc)
  local blocks = doc.blocks
  local new_blocks = {}
  local i = 1
  local in_note_section = false
  local note_blocks = {}
  
  while i <= #blocks do
    local block = blocks[i]
    
    -- Check if this is a BlockQuote
    if block.t == "BlockQuote" then
      if starts_with_notes(block) then
        -- Start of note section
        in_note_section = true
        note_blocks = {}
        
        -- Remove NOTES: from the content and add to note_blocks
        local cleaned_block = remove_notes_marker(block)
        if #cleaned_block.c > 0 then
          -- Add the cleaned content blocks
          for j, blk in ipairs(cleaned_block.c) do
            table.insert(note_blocks, blk)
          end
        end
        
        -- Skip this block (don't add to new_blocks)
        i = i + 1
        
      elseif in_note_section and (ends_with_notee(block) or starts_with_notee(block)) then
        -- End of note section (NOTEE detected)
        in_note_section = false
        
        -- If it's a standalone NOTEE blockquote, skip it entirely (don't add to content)
        -- Otherwise, remove NOTEE from the last block and add the rest
        if starts_with_notee(block) then
          -- Skip this block entirely (it's just the NOTEE end marker)
        else
          -- Remove NOTEE from the last block if it exists
          local cleaned_block = remove_notee_marker(block)
          if #cleaned_block.c > 0 then
            for j, blk in ipairs(cleaned_block.c) do
              table.insert(note_blocks, blk)
            end
          end
        end
        
        -- Create LaTeX code for the note section
        table.insert(new_blocks, pandoc.RawBlock('latex', "\\begin{notesection}\n"))
        for j, blk in ipairs(note_blocks) do
          table.insert(new_blocks, blk)
        end
        table.insert(new_blocks, pandoc.RawBlock('latex', "\\end{notesection}\n"))
        
        -- Reset
        note_blocks = {}
        i = i + 1
        
      else
        -- Normal block quote
        if in_note_section then
          -- Collect this block quote into note_blocks
          for j, blk in ipairs(block.c) do
            table.insert(note_blocks, blk)
          end
        else
          table.insert(new_blocks, block)
        end
        i = i + 1
      end
      
    else
      -- Not a block quote
      if in_note_section then
        -- Collect this block into note_blocks
        table.insert(note_blocks, block)
      else
        table.insert(new_blocks, block)
      end
      i = i + 1
    end
  end
  
  -- If we're still in a note section at the end, add remaining blocks
  if in_note_section and #note_blocks > 0 then
    table.insert(new_blocks, pandoc.RawBlock('latex', "\\begin{notesection}\n"))
    for j, blk in ipairs(note_blocks) do
      table.insert(new_blocks, blk)
    end
    table.insert(new_blocks, pandoc.RawBlock('latex', "\\end{notesection}\n"))
  end
  
  return pandoc.Pandoc(new_blocks, doc.meta)
end

