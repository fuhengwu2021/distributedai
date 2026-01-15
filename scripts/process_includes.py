#!/usr/bin/env python3
"""
Script to process conditional includes in Markdown files.

This script:
1. Reads a markdown file
2. Finds all <!-- include: file.md if condition --> directives
3. Loads condition values from peanut.config (or custom config file)
4. Evaluates conditions and replaces includes with file contents
5. Handles recursive includes (max depth: 10)
6. Outputs processed markdown

Usage:
    python3 scripts/process_includes.py --process-file input.md output.md [--config config_file]
"""

import re
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Set, Optional


def load_config(config_path: Optional[Path]) -> Dict[str, bool]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to config file, or None to use default
        
    Returns:
        Dictionary of condition names to boolean values
    """
    if config_path is None:
        # Default: look for peanut.config in project root
        script_dir = Path(__file__).parent.parent
        config_path = script_dir / "peanut.config"
    
    if not config_path.exists():
        # Config file not found - return empty dict (all conditions will be False)
        return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Convert all values to boolean
        result = {}
        for key, value in config.items():
            if isinstance(value, bool):
                result[key] = value
            elif isinstance(value, str):
                # String values: non-empty strings are True
                result[key] = bool(value and value.strip())
            elif isinstance(value, (int, float)):
                # Numbers: non-zero are True
                result[key] = bool(value)
            else:
                result[key] = bool(value)
        
        return result
    except json.JSONDecodeError as e:
        print(f"Warning: Invalid JSON in config file {config_path}: {e}", file=sys.stderr)
        return {}
    except Exception as e:
        print(f"Warning: Error reading config file {config_path}: {e}", file=sys.stderr)
        return {}


def evaluate_condition(condition_expr: str, config: Dict[str, bool]) -> bool:
    """
    Evaluate a condition expression.
    
    Supports:
    - Simple condition: "condition_name"
    - Negation: "not condition_name"
    - AND: "condition1 and condition2"
    - OR: "condition1 or condition2"
    - Parentheses: "(condition1 or condition2) and condition3"
    
    Args:
        condition_expr: Condition expression string
        config: Configuration dictionary
        
    Returns:
        True if condition is met, False otherwise
    """
    condition_expr = condition_expr.strip()
    
    if not condition_expr:
        return False
    
    # Handle parentheses first
    while '(' in condition_expr:
        # Find innermost parentheses
        start = condition_expr.rfind('(')
        end = condition_expr.find(')', start)
        if end == -1:
            # Unmatched parenthesis - treat as invalid
            return False
        
        inner = condition_expr[start+1:end]
        inner_result = evaluate_condition(inner, config)
        condition_expr = condition_expr[:start] + str(inner_result).lower() + condition_expr[end+1:]
    
    # Handle NOT operators
    condition_expr = re.sub(r'\bnot\s+(\w+)\b', lambda m: str(not config.get(m.group(1), False)).lower(), condition_expr, flags=re.IGNORECASE)
    
    # Handle AND operators (higher precedence than OR)
    while ' and ' in condition_expr.lower():
        match = re.search(r'\b(\w+)\s+and\s+(\w+)\b', condition_expr, re.IGNORECASE)
        if match:
            left = match.group(1).lower()
            right = match.group(2).lower()
            
            # Check if these are boolean strings or condition names
            if left in ('true', 'false'):
                left_val = left == 'true'
            else:
                left_val = config.get(match.group(1), False)
            
            if right in ('true', 'false'):
                right_val = right == 'true'
            else:
                right_val = config.get(match.group(2), False)
            
            result = left_val and right_val
            condition_expr = condition_expr[:match.start()] + str(result).lower() + condition_expr[match.end():]
        else:
            break
    
    # Handle OR operators
    while ' or ' in condition_expr.lower():
        match = re.search(r'\b(\w+)\s+or\s+(\w+)\b', condition_expr, re.IGNORECASE)
        if match:
            left = match.group(1).lower()
            right = match.group(2).lower()
            
            # Check if these are boolean strings or condition names
            if left in ('true', 'false'):
                left_val = left == 'true'
            else:
                left_val = config.get(match.group(1), False)
            
            if right in ('true', 'false'):
                right_val = right == 'true'
            else:
                right_val = config.get(match.group(2), False)
            
            result = left_val or right_val
            condition_expr = condition_expr[:match.start()] + str(result).lower() + condition_expr[match.end():]
        else:
            break
    
    # Final value should be a single boolean string or condition name
    condition_expr = condition_expr.strip().lower()
    if condition_expr in ('true', 'false'):
        return condition_expr == 'true'
    else:
        # Treat as condition name
        return config.get(condition_expr, False)


def process_includes(
    content: str,
    base_dir: Path,
    config: Dict[str, bool],
    visited_files: Optional[Set[Path]] = None,
    depth: int = 0
) -> str:
    """
    Process include directives in markdown content.
    
    Args:
        content: Markdown content to process
        base_dir: Base directory for resolving relative paths
        config: Configuration dictionary
        visited_files: Set of already visited files (to prevent circular includes)
        depth: Current recursion depth
        
    Returns:
        Processed markdown content
    """
    if visited_files is None:
        visited_files = set()
    
    if depth > 10:
        print(f"Warning: Maximum include depth (10) exceeded. Stopping recursion.", file=sys.stderr)
        return content
    
    # Pattern: <!-- include: file_path if condition -->
    include_pattern = r'<!--\s*include:\s*([^\s]+)\s+if\s+([^>]+)\s*-->'
    
    def replace_include(match):
        file_path_str = match.group(1).strip()
        condition_expr = match.group(2).strip()
        
        # Evaluate condition
        if not evaluate_condition(condition_expr, config):
            # Condition not met - remove the include directive
            return ''
        
        # Resolve file path (relative to base_dir)
        if Path(file_path_str).is_absolute():
            include_path = Path(file_path_str)
        else:
            include_path = (base_dir / file_path_str).resolve()
        
        # Check for circular includes
        if include_path in visited_files:
            print(f"Warning: Circular include detected: {include_path}", file=sys.stderr)
            return ''
        
        # Check if file exists
        if not include_path.exists():
            print(f"Warning: Include file not found: {include_path} (relative to {base_dir})", file=sys.stderr)
            return ''
        
        # Read included file
        try:
            included_content = include_path.read_text(encoding='utf-8')
        except Exception as e:
            print(f"Warning: Error reading include file {include_path}: {e}", file=sys.stderr)
            return ''
        
        # Recursively process includes in the included file
        included_dir = include_path.parent
        visited_files.add(include_path)
        processed_content = process_includes(
            included_content,
            included_dir,
            config,
            visited_files,
            depth + 1
        )
        visited_files.remove(include_path)
        
        # Return the processed content (with a newline before and after for proper spacing)
        return f'\n{processed_content}\n'
    
    # Replace all include directives
    result = re.sub(include_pattern, replace_include, content)
    
    return result


def process_markdown_file(
    input_file: Path,
    output_file: Path,
    config_path: Optional[Path] = None
) -> bool:
    """
    Process a markdown file and handle conditional includes.
    
    Args:
        input_file: Input markdown file path
        output_file: Output markdown file path
        config_path: Optional path to config file
        
    Returns:
        True if successful, False otherwise
    """
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}", file=sys.stderr)
        return False
    
    # Load config
    config = load_config(config_path)
    
    # Read input file
    try:
        content = input_file.read_text(encoding='utf-8')
    except Exception as e:
        print(f"Error: Failed to read input file {input_file}: {e}", file=sys.stderr)
        return False
    
    # Process includes
    base_dir = input_file.parent
    processed_content = process_includes(content, base_dir, config)
    
    # Write output file
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(processed_content, encoding='utf-8')
        return True
    except Exception as e:
        print(f"Error: Failed to write output file {output_file}: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Process conditional includes in Markdown files'
    )
    parser.add_argument(
        '--process-file',
        nargs=2,
        metavar=('INPUT', 'OUTPUT'),
        help='Process input file and write to output file'
    )
    parser.add_argument(
        '--config',
        '-c',
        metavar='CONFIG_FILE',
        help='Path to config file (default: peanut.config in project root)'
    )
    
    args = parser.parse_args()
    
    if args.process_file:
        input_path = Path(args.process_file[0])
        output_path = Path(args.process_file[1])
        config_path = Path(args.config) if args.config else None
        
        if process_markdown_file(input_path, output_path, config_path):
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
