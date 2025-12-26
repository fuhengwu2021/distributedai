#!/usr/bin/env python3
"""
Script to add line number annotations to code blocks in markdown files.

Usage:
    python3 add_code_annotations.py input.md output.md

The script processes code blocks and adds LaTeX annotations for line numbers.
Users can mark lines in code blocks using HTML comments like <!--①--> or <!--1-->,
and add explanations after code blocks using a special format.

Example markdown:
```python
print("Hello")  <!--①-->
print("World")  <!--②-->
```

<!--code-explanations-->
① First line prints "Hello"
② Second line prints "World"
<!--/code-explanations-->
"""

import re
import sys
from pathlib import Path


def process_code_block_with_annotations(code_block, language, explanations):
    """Process a code block and add LaTeX annotations."""
    lines = code_block.split('\n')
    annotated_lines = []
    annotation_counter = 1
    
    for line in lines:
        # Check for HTML comment annotations like <!--①--> or <!--1-->
        annotation_match = re.search(r'<!--([①②③④⑤⑥⑦⑧⑨⑩]|\d+)-->', line)
        if annotation_match:
            # Remove the annotation from the line
            clean_line = re.sub(r'<!--([①②③④⑤⑥⑦⑧⑨⑩]|\d+)-->', '', line).rstrip()
            annotation_symbol = annotation_match.group(1)
            
            # Convert circled numbers to regular numbers for lookup
            if annotation_symbol in '①②③④⑤⑥⑦⑧⑨⑩':
                annotation_num = '①②③④⑤⑥⑦⑧⑨⑩'.index(annotation_symbol) + 1
            else:
                annotation_num = int(annotation_symbol)
            
            # Add LaTeX annotation at the end of the line
            annotated_lines.append(f"{clean_line} \\codelineannotation{{{annotation_num}}}{{{explanations.get(annotation_num, '')}}}")
        else:
            annotated_lines.append(line)
    
    return '\n'.join(annotated_lines)


def extract_explanations(content):
    """Extract code explanations from markdown content."""
    explanations = {}
    
    # Pattern to match explanation block
    pattern = r'<!--code-explanations-->([\s\S]*?)<!--/code-explanations-->'
    match = re.search(pattern, content)
    
    if match:
        explanation_text = match.group(1).strip()
        # Parse explanations: ① text or 1 text
        lines = explanation_text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Match circled numbers or regular numbers
            match_num = re.match(r'^([①②③④⑤⑥⑦⑧⑨⑩]|\d+)\s+(.+)$', line)
            if match_num:
                num_symbol = match_num.group(1)
                explanation = match_num.group(2).strip()
                
                # Convert circled number to regular number
                if num_symbol in '①②③④⑤⑥⑦⑧⑨⑩':
                    num = '①②③④⑤⑥⑦⑧⑨⑩'.index(num_symbol) + 1
                else:
                    num = int(num_symbol)
                
                explanations[num] = explanation
    
    return explanations


def process_markdown_file(input_file, output_file):
    """Process markdown file and add code annotations."""
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        print(f"Error: File not found: {input_file}", file=sys.stderr)
        return False
    
    content = input_path.read_text(encoding='utf-8')
    
    # Extract explanations first
    explanations = extract_explanations(content)
    
    # Remove explanation blocks from content (they'll be added as LaTeX later)
    content = re.sub(r'<!--code-explanations-->[\s\S]*?<!--/code-explanations-->', '', content)
    
    # Process code blocks
    # Pattern to match code blocks: ```language\ncode\n```
    code_block_pattern = r'```(\w+)\n(.*?)\n```'
    
    def replace_code_block(match):
        language = match.group(1)
        code_content = match.group(2)
        
        # Process annotations in this code block
        processed_code = process_code_block_with_annotations(code_content, language, explanations)
        
        # Return as LaTeX lstlisting block
        # Escape special LaTeX characters
        processed_code = processed_code.replace('\\', '\\textbackslash{}')
        processed_code = processed_code.replace('{', '\\{')
        processed_code = processed_code.replace('}', '\\}')
        processed_code = processed_code.replace('$', '\\$')
        processed_code = processed_code.replace('&', '\\&')
        processed_code = processed_code.replace('%', '\\%')
        processed_code = processed_code.replace('#', '\\#')
        processed_code = processed_code.replace('^', '\\textasciicircum{}')
        processed_code = processed_code.replace('_', '\\_')
        
        # Generate LaTeX code block with annotations
        latex_code = f"\\begin{{lstlisting}}[language={language}]\n{processed_code}\n\\end{{lstlisting}}"
        
        # Add explanations if any
        if explanations:
            explanation_text = "\\begin{codeexplanation}\n"
            for num in sorted(explanations.keys()):
                explanation_text += f"\\codelineannotation{{{num}}}{{{explanations[num]}}}\\\\\n"
            explanation_text += "\\end{codeexplanation}"
            latex_code += "\n\n" + explanation_text
            explanations.clear()  # Clear after use
        
        return latex_code
    
    # Replace code blocks
    processed_content = re.sub(code_block_pattern, replace_code_block, content, flags=re.DOTALL)
    
    # Write output
    output_path.write_text(processed_content, encoding='utf-8')
    return True


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 add_code_annotations.py input.md output.md", file=sys.stderr)
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if process_markdown_file(input_file, output_file):
        print(f"✅ Successfully processed: {input_file} -> {output_file}")
    else:
        print(f"❌ Failed to process: {input_file}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

