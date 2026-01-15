-- Lua filter to reset section and subsection counters before first ## heading
-- This ensures that the first ## heading is numbered as "1" instead of "0.1"
-- when there's an unnumbered # heading before it

function Pandoc(doc)
  local blocks = doc.blocks
  local new_blocks = {}
  local first_section_found = false
  
  for i, block in ipairs(blocks) do
    -- Check if this is the first ## heading (section level 2)
    if not first_section_found and block.t == "Header" and block.level == 2 then
      first_section_found = true
      -- Insert LaTeX commands to reset both section and subsection counters
      -- Reset section to 0 and subsection to 0, so first subsection shows as "1"
      table.insert(new_blocks, pandoc.RawBlock('latex', '\\setcounter{section}{0}\n\\setcounter{subsection}{0}\n'))
    end
    table.insert(new_blocks, block)
  end
  
  return pandoc.Pandoc(new_blocks, doc.meta)
end

