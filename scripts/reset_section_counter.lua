-- Lua filter to reset section and subsection counters
-- For book class: reset counters before each first section (## heading) that appears after chaptertitlepage
-- For article class: reset counters before first section (level 2 heading)

function Pandoc(doc)
  local blocks = doc.blocks
  local new_blocks = {}
  local first_section_found = false
  local is_book_class = false
  local chapter_count = 0
  local last_was_chapter_title = false
  
  -- Check if document class is book
  if doc.meta and doc.meta.documentclass then
    local doc_class = pandoc.utils.stringify(doc.meta.documentclass)
    if doc_class == "book" then
      is_book_class = true
    end
  end
  
  for i, block in ipairs(blocks) do
    -- Check if this is a chaptertitlepage (RawBlock with latex containing chaptertitlepage)
    if block.t == "RawBlock" and block.format == "latex" then
      local latex_content = block.text
      if latex_content:match("chaptertitlepage") then
        last_was_chapter_title = true
        chapter_count = chapter_count + 1
        table.insert(new_blocks, block)
      else
        table.insert(new_blocks, block)
      end
    -- For book class: reset counters before first section (##) after each chaptertitlepage
    elseif is_book_class and block.t == "Header" and block.level == 2 then
      if last_was_chapter_title or (not first_section_found) then
        -- This is the first section after a chapter title page, reset counters
        table.insert(new_blocks, pandoc.RawBlock('latex', '\\setcounter{section}{0}\n\\setcounter{subsection}{0}\n'))
        first_section_found = true
        last_was_chapter_title = false
      end
      table.insert(new_blocks, block)
    -- For article class: reset counters before first section (level 2 heading)
    elseif not is_book_class and not first_section_found and block.t == "Header" and block.level == 2 then
      first_section_found = true
      -- Insert LaTeX commands to reset both section and subsection counters
      -- Reset section to 0 and subsection to 0, so first subsection shows as "1"
      table.insert(new_blocks, pandoc.RawBlock('latex', '\\setcounter{section}{0}\n\\setcounter{subsection}{0}\n'))
      table.insert(new_blocks, block)
    else
      table.insert(new_blocks, block)
    end
  end
  
  return pandoc.Pandoc(new_blocks, doc.meta)
end

