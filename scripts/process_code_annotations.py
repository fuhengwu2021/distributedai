#!/usr/bin/env python3
"""
Script to process code blocks with line annotations in markdown files.

This script:
1. Detects HTML comments in code blocks (e.g., <!--①--> or <!--1-->)
2. Adds LaTeX commands to display circled numbers at the end of code lines
3. Processes explanation blocks to ensure proper line breaks

Usage:
    python3 process_code_annotations.py input.md output.md
"""

import re
import sys
from pathlib import Path


def get_circled_number(num):
    """Convert number to circled number Unicode character."""
    circled_numbers = {
        1: '①', 2: '②', 3: '③', 4: '④', 5: '⑤',
        6: '⑥', 7: '⑦', 8: '⑧', 9: '⑨', 10: '⑩',
        11: '⑪', 12: '⑫', 13: '⑬', 14: '⑭', 15: '⑮',
        16: '⑯', 17: '⑰', 18: '⑱', 19: '⑲', 20: '⑳'
    }
    return circled_numbers.get(num, str(num))


def process_code_block(match):
    """Process a code block and add LaTeX annotations."""
    language = match.group(1)
    code_content = match.group(2)
    
    lines = code_content.split('\n')
    processed_lines = []
    
    for line in lines:
        # Check for HTML comment annotations like <!--①--> or <!--1-->
        annotation_match = re.search(r'<!--([①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]|\d+)-->', line)
        if annotation_match:
            # Remove the annotation from the line
            clean_line = re.sub(r'<!--([①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]|\d+)-->', '', line).rstrip()
            annotation_symbol = annotation_match.group(1)
            
            # Convert circled numbers to regular numbers
            if annotation_symbol in '①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳':
                annotation_num = '①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳'.index(annotation_symbol) + 1
            else:
                annotation_num = int(annotation_symbol)
            
            # Add LaTeX command at the end of the line (using raw LaTeX)
            # We'll use a special marker that will be converted to LaTeX later
            processed_lines.append(f"{clean_line}  %ANNOTATION:{annotation_num}%")
        else:
            processed_lines.append(line)
    
    processed_code = '\n'.join(processed_lines)
    
    # Return the code block with annotations
    return f"```{language}\n{processed_code}\n```"


def process_explanations(content):
    """Process explanation blocks to ensure proper line breaks."""
    # Pattern to match codeexplanation environment
    pattern = r'\\begin\{codeexplanation\}(.*?)\\end\{codeexplanation\}'
    
    def replace_explanations(match):
        explanation_content = match.group(1)
        # Split by \codelineannotation and ensure each is on a new line
        lines = explanation_content.split('\\codelineannotation')
        processed_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            if i == 0:
                # First line (before first \codelineannotation)
                if line:
                    processed_lines.append(line)
            else:
                # Lines starting with \codelineannotation
                processed_lines.append('\\codelineannotation' + line)
        
        # Join with double backslash for line breaks
        result = '\n'.join(processed_lines)
        return f"\\begin{{codeexplanation}}\n{result}\n\\end{{codeexplanation}}"
    
    return re.sub(pattern, replace_explanations, content, flags=re.DOTALL)


def convert_annotations_to_latex(content):
    """Convert annotation markers in code blocks to LaTeX commands."""
    # Pattern to match code blocks
    code_block_pattern = r'```(\w+)\n(.*?)\n```'
    
    def replace_code_block(match):
        language = match.group(1)
        code_content = match.group(2)
        
        lines = code_content.split('\n')
        processed_lines = []
        
        for line in lines:
            # Check for annotation marker
            if '%ANNOTATION:' in line:
                # Extract annotation number
                annotation_match = re.search(r'%ANNOTATION:(\d+)%', line)
                if annotation_match:
                    annotation_num = int(annotation_match.group(1))
                    # Remove the marker and add LaTeX command
                    clean_line = re.sub(r'\s*%ANNOTATION:\d+%', '', line)
                    # Add LaTeX command for circled number at the end
                    processed_lines.append(f"{clean_line} \\codelinemark{{{annotation_num}}}")
                else:
                    processed_lines.append(line)
            else:
                processed_lines.append(line)
        
        processed_code = '\n'.join(processed_lines)
        return f"```{language}\n{processed_code}\n```"
    
    return re.sub(code_block_pattern, replace_code_block, content, flags=re.DOTALL)


def process_markdown_file(input_file, output_file):
    """Process markdown file and add code annotations."""
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        print(f"Error: File not found: {input_file}", file=sys.stderr)
        return False
    
    content = input_path.read_text(encoding='utf-8')
    
    # First, process code blocks to add annotation markers
    code_block_pattern = r'```(\w+)\n(.*?)\n```'
    content = re.sub(code_block_pattern, process_code_block, content, flags=re.DOTALL)
    
    # Convert annotation markers to LaTeX commands
    content = convert_annotations_to_latex(content)
    
    # Process explanation blocks for proper line breaks
    content = process_explanations(content)
    
    # Write output
    output_path.write_text(content, encoding='utf-8')
    return True


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 process_code_annotations.py input.md output.md", file=sys.stderr)
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

