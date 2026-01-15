#!/usr/bin/env python3
"""
Script to process CODE_EXPLAIN_START/CODE_EXPLAIN_END blocks in markdown files.

This script converts CODE_EXPLAIN_START/CODE_EXPLAIN_END blocks to LaTeX codeexplanation format.
Also adds #LINENUM marker to code blocks and #HL markers to explained lines.

Usage:
    python3 process_code_explain.py --process-file input.md output.md
"""

import re
import sys
import argparse
from pathlib import Path


def process_code_explain_blocks(content):
    """Convert CODE_EXPLAIN_START/CODE_EXPLAIN_END blocks to LaTeX format.
    Also adds #LINENUM marker to the preceding code block and #HL markers to explained lines."""
    
    # Pattern to match code block followed by CODE_EXPLAIN_START ... CODE_EXPLAIN_END
    # This captures the entire block including the explanation
    code_block_pattern = r'(```(?:python|py|javascript|js|java|cpp|c|go|rust|sql|html|css|json|yaml|xml|bash|sh|shell|text)?\n(?:[^`]|`(?!``))*?```)\s*\n\s*CODE_EXPLAIN_START:\s*\n(.*?)\n\s*CODE_EXPLAIN_END'
    
    def process_code_block_with_explanation(match):
        code_block = match.group(1)
        explanation_content = match.group(2).strip()
        
        # Check if #LINEBAR is used (standalone or in comment)
        use_linebar = False
        if '#LINEBAR' in code_block:
            use_linebar = True
        
        # Extract line numbers from explanation
        explained_lines = set()
        for line in explanation_content.split('\n'):
            line = line.strip()
            if not line:
                continue
            # Match pattern: - 1: description
            m = re.match(r'-\s*(\d+):\s*(.+)', line)
            if m:
                line_num = int(m.group(1))
                explained_lines.add(line_num)
        
        # Check if #LINENUM is already present
        has_linenum = '#LINENUM' in code_block
        
        # Add #LINENUM if not present
        if not has_linenum:
            code_block = re.sub(
                r'(```(?:python|py|javascript|js|java|cpp|c|go|rust|sql|html|css|json|yaml|xml|bash|sh|shell|text)?)\n',
                r'\1\n#LINENUM\n',
                code_block,
                count=1
            )
        
        # Add #HL markers to explained lines
        if explained_lines:
            lines = code_block.split('\n')
            modified_lines = []
            line_num = 0
            in_code = False
            
            for line in lines:
                # Check if we're entering the code block
                if re.match(r'^```(?:python|py|javascript|js|java|cpp|c|go|rust|sql|html|css|json|yaml|xml|bash|sh|shell|text)?$', line):
                    modified_lines.append(line)
                    in_code = True
                    continue
                
                # Check if we're leaving the code block
                if line.strip() == '```':
                    modified_lines.append(line)
                    in_code = False
                    continue
                
                # Skip #LINENUM and #LINEBAR lines (they will be removed later)
                if line.strip() == '#LINENUM':
                    modified_lines.append(line)
                    continue
                if line.strip() == '#LINEBAR' or (line.strip().startswith('#LINEBAR') and '#HL' in line):
                    modified_lines.append(line)
                    continue
                
                # Process actual code lines
                if in_code:
                    line_num += 1
                    if line_num in explained_lines:
                        # Add #HL marker at the end of the line
                        # Remove any existing #HL first
                        line = re.sub(r'\s*#HL\s*$', '', line)
                        line = line.rstrip() + ' #HL'
                    modified_lines.append(line)
                else:
                    modified_lines.append(line)
            
            code_block = '\n'.join(modified_lines)
        
        # Process explanation to LaTeX
        latex_lines = []
        for line in explanation_content.split('\n'):
            line = line.strip()
            if not line:
                continue
            m = re.match(r'-\s*(\d+):\s*(.+)', line)
            if m:
                num = m.group(1)
                desc = m.group(2).strip()
                # Escape LaTeX special characters
                desc = desc.replace('\\', '\\textbackslash{}')
                desc = desc.replace('{', '\\{')
                desc = desc.replace('}', '\\}')
                desc = desc.replace('$', '\\$')
                desc = desc.replace('&', '\\&')
                desc = desc.replace('%', '\\%')
                desc = desc.replace('#', '\\#')
                desc = desc.replace('^', '\\textasciicircum{}')
                desc = desc.replace('_', '\\_')
                # Use box style annotation if #LINEBAR is used
                if use_linebar:
                    latex_lines.append(f"\\codelineannotationbar{{{num}}}{{{desc}}}")
                else:
                    latex_lines.append(f"\\codelineannotation{{{num}}}{{{desc}}}")
        
        latex_part = ""
        if latex_lines:
            latex_content = "\\begin{codeexplanation}\n" + "\n".join(latex_lines) + "\n\\end{codeexplanation}"
            latex_part = "\n\n```{=latex}\n" + latex_content + "\n```"
        
        return code_block + latex_part
    
    # Process code blocks with explanations
    processed_content = re.sub(code_block_pattern, process_code_block_with_explanation, content, flags=re.MULTILINE | re.DOTALL)
    return processed_content


def process_markdown_file(input_file, output_file):
    """Process markdown file and convert CODE_EXPLAIN blocks."""
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        print(f"Error: File not found: {input_file}", file=sys.stderr)
        return False
    
    content = input_path.read_text(encoding='utf-8')
    
    # Process CODE_EXPLAIN blocks
    processed_content = process_code_explain_blocks(content)
    
    # Write output
    output_path.write_text(processed_content, encoding='utf-8')
    return True


def main():
    parser = argparse.ArgumentParser(description='Process CODE_EXPLAIN_START/CODE_EXPLAIN_END blocks in markdown files')
    parser.add_argument('--process-file', nargs=2, metavar=('INPUT', 'OUTPUT'),
                       help='Process input markdown file and write to output file')
    
    args = parser.parse_args()
    
    if args.process_file:
        input_file, output_file = args.process_file
        if process_markdown_file(input_file, output_file):
            return 0
        else:
            return 1
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
