# Chapter Format Standard

This document defines the standard format for all chapters in this book.

## Required Structure

Each chapter markdown file must follow this exact structure at the beginning:

```markdown
# Chapter N: Chapter Title

*Subtitle text in italic*

> Quote text here
- Author Name | Title, Organization, Year

**Code Summary**

- `code_element`: Description of the code element
- `another_code`: Another description
- ... (list 10 most important classes or functions used in the chapter)
```

## Format Rules

1. **Chapter Header** (Line 1):
   - Format: `# Chapter N: Chapter Title`
   - Must start with `#` (H1 heading)
   - Include chapter number and full title

2. **Subtitle** (Line 3):
   - Format: `*Subtitle text in italic*`
   - Single line, wrapped in asterisks for italic
   - Should be a brief, descriptive subtitle

3. **Quote Section** (Lines 5-6):
   - Line 5: Blockquote with `>` followed by quote text
   - Line 6: Author attribution with `-` (dash), format: `- Author Name | Title, Organization, Year`
   - Quote should be inspirational or relevant to the chapter

4. **Code Summary** (Lines 8-19):
   - Header: `**Code Summary**` (bold)
   - List format: `- `code_element`: Description`
   - Use backticks around code elements
   - List approximately 10 most important classes or functions used in the chapter
   - Each item should be concise and descriptive

5. **Main Content** (Line 21+):
   - Regular chapter content starts after the Code Summary section
   - All standard markdown formatting applies

## Example

See `chapter1-introduction-to-modern-distributed-ai/chapter1.md` for a complete example.

## Notes

- The quote and code summary sections are automatically extracted by `scripts/extract_chapter_title_quote.py` for PDF generation
- Do not modify the structure of these sections after they are set up
- The code summary should reflect the actual content of the chapter
