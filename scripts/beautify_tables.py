#!/usr/bin/env python3
"""
Post-process LaTeX file to add table beautification (alternating row colors)
"""
import re
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

def beautify_tables(latex_content):
    """Add rowcolors to all longtable and tabular environments"""
    logger.info("Starting table beautification")
    
    # Fix xcolor loading: ensure [table] option is set
    # Pandoc 3.8 loads xcolor without [table] option, which prevents \rowcolors from working
    # Strategy: Use \PassOptionsToPackage{table}{xcolor} before any xcolor load
    # Then ensure \usepackage{xcolor} has [table] option
    # If \PassOptionsToPackage{table}{xcolor} is not present, add it before the first xcolor load
    
    # Find all xcolor package declarations
    xcolor_matches = list(re.finditer(r'\\usepackage(?:\[[^\]]*\])?\{xcolor\}', latex_content))
    logger.debug(f"Found {len(xcolor_matches)} xcolor package declarations")
    
    # Replace all \usepackage{xcolor} (without [table]) with \usepackage[table]{xcolor}
    # But keep \usepackage[table]{xcolor} as is
    def fix_xcolor(match):
        full_match = match.group(0)
        if '[table]' in full_match:
            return full_match  # Already has [table], keep as is
        else:
            # Replace with [table] option
            return r'\usepackage[table]{xcolor}'
    
    latex_content = re.sub(r'\\usepackage(?:\[[^\]]*\])?\{xcolor\}', fix_xcolor, latex_content)
    
    if xcolor_matches:
        logger.info("Fixed xcolor package declarations to include [table] option")
    
    # Remove ALL duplicate xcolor loads (keep only the first one with [table])
    # This includes both \usepackage[table]{xcolor} and any other xcolor variants
    xcolor_table_pattern = r'\\usepackage\[table\]\{xcolor\}'
    xcolor_any_pattern = r'\\usepackage(?:\[[^\]]*\])?\{xcolor\}'
    
    table_matches = list(re.finditer(xcolor_table_pattern, latex_content))
    all_matches = list(re.finditer(xcolor_any_pattern, latex_content))
    logger.debug(f"Found {len(table_matches)} xcolor[table] declarations, {len(all_matches)} total xcolor declarations")
    
    if len(all_matches) > 1:
        logger.info(f"Removing {len(all_matches) - 1} duplicate xcolor declarations (keeping first [table] version)")
        # Keep the first [table] version, remove all others
        first_kept = False
        for match in reversed(all_matches):
            if '[table]' in match.group(0) and not first_kept:
                # This is the first [table] version, keep it
                first_kept = True
                continue
            else:
                # Remove this one (either duplicate [table] or non-[table] version)
                # Find start of line
                line_start = latex_content.rfind('\n', 0, match.start()) + 1
                # Find end of line
                line_end = latex_content.find('\n', match.end())
                if line_end == -1:
                    line_end = len(latex_content)
                else:
                    line_end += 1  # Include the newline
                # Remove the entire line
                latex_content = latex_content[:line_start] + latex_content[line_end:]
    
    # Ensure tableyellow and tablered colors are defined
    # Insert after \PassOptionsToPackage{table}{xcolor} or \usepackage[table]{xcolor} or before \begin{document}
    colors_to_define = []
    if r'\definecolor{tableyellow}' not in latex_content:
        colors_to_define.append(r'\definecolor{tableyellow}{RGB}{255,255,200}')
    if r'\definecolor{tablered}' not in latex_content:
        colors_to_define.append(r'\definecolor{tablered}{RGB}{255,230,230}')
    
    if colors_to_define:
        color_defs = '\n'.join(colors_to_define)
        logger.info(f"Colors not found, inserting definitions: {', '.join([c.split('{')[1] for c in colors_to_define])}")
        # Try to find insertion point after xcolor-related commands
        insert_patterns = [
            (r'(\\PassOptionsToPackage\{table\}\{xcolor\})', rf'\1\n{color_defs}'),
            (r'(\\usepackage\[table\]\{xcolor\})', rf'\1\n{color_defs}'),
        ]
        
        inserted = False
        for pattern, replacement in insert_patterns:
            if re.search(pattern, latex_content):
                logger.debug(f"Inserting colors after pattern: {pattern}")
                latex_content = re.sub(pattern, replacement, latex_content, count=1)
                inserted = True
                break
        
        # If no xcolor command found, insert before \begin{document}
        if not inserted:
            doc_pattern = r'(\\begin\{document\})'
            if re.search(doc_pattern, latex_content):
                logger.debug("Inserting colors before \\begin{document}")
                latex_content = re.sub(doc_pattern, rf'{color_defs}\n\1', latex_content, count=1)
    else:
        logger.debug("All colors already defined")
    
    # Pattern to match \begin{longtable}...\end{longtable}
    # We need to add \rowcolors after \begin{longtable} but before content
    def add_rowcolors_to_longtable(match):
        full_table = match.group(0)
        logger.debug("Processing longtable")
        
        # Remove existing rowcolors (including those from \AtBeginEnvironment in the template)
        # Match \rowcolors{...}{...}{...} with optional % comment and newline
        old_rowcolors = re.findall(r'\\rowcolors\{[^}]+\}\{[^}]+\}\{[^}]+\}', full_table)
        if old_rowcolors:
            logger.info(f"Removing {len(old_rowcolors)} existing rowcolors command(s): {old_rowcolors}")
        else:
            logger.debug("No existing rowcolors found to remove")
        # Remove \rowcolors commands (with or without % comment, with or without newline)
        full_table = re.sub(r'\\rowcolors\{[^}]+\}\{[^}]+\}\{[^}]+\}%?\s*\n?', '', full_table)
        full_table = re.sub(r'\\noalign\{\\global\\rownum=\d+\}%?\s*\n?', '', full_table)
        
        # Add white header
        if r'\toprule' in full_table:
            logger.debug("Found \\toprule, adding \\rowcolor{white} for header")
            full_table = re.sub(r'(\\toprule\s*\n)', r'\1\\rowcolor{white}\n', full_table, count=1)
        
        # Insert rowcolors in correct location
        # After \endhead, we reset row counter to 1, but header is row 1, so start from row 2 (first data row)
        # The header is already set to white with \rowcolor{white}
        # Use tablered for odd rows (1,3,5...) and tableyellow for even rows (2,4,6...)
        # After reset to 1, row 1 = header (white), row 2 = first data (red), row 3 = second data (yellow)
        if r'\endlastfoot' in full_table:
            logger.info("Found \\endlastfoot, inserting rowcolors{2}{tablered}{tableyellow} after it (row counter reset, skip header)")
            pattern = r'(\\endlastfoot\s*\n)'
            replacement = r'\1\\noalign{\\global\\rownum=1}\n\\rowcolors{2}{tablered}{tableyellow}\n'
            result = re.sub(pattern, replacement, full_table, count=1)
            # Verify insertion
            if r'\rowcolors{2}{tablered}{tableyellow}' in result:
                logger.debug("✓ Successfully inserted rowcolors{2} after \\endlastfoot")
            else:
                logger.warning("✗ Failed to insert rowcolors after \\endlastfoot")
        elif r'\endhead' in full_table:
            logger.info("Found \\endhead (no \\endlastfoot), inserting rowcolors{2}{tablered}{tableyellow} after it (row counter reset, skip header)")
            pattern = r'(\\endhead\s*\n)'
            replacement = r'\1\\noalign{\\global\\rownum=1}\n\\rowcolors{2}{tablered}{tableyellow}\n'
            result = re.sub(pattern, replacement, full_table, count=1)
            # Verify insertion
            if r'\rowcolors{2}{tablered}{tableyellow}' in result:
                logger.debug("✓ Successfully inserted rowcolors{2} after \\endhead")
            else:
                logger.warning("✗ Failed to insert rowcolors after \\endhead")
        else:
            logger.info("No \\endlastfoot or \\endhead, inserting rowcolors{2}{tablered}{tableyellow} after \\begin{longtable} (skip header)")
            pattern = r'(\\begin{longtable}[^\n]+\n)'
            replacement = r'\1\\rowcolors{2}{tablered}{tableyellow}\n'
            result = re.sub(pattern, replacement, full_table, count=1)
            # Verify insertion
            if r'\rowcolors{2}{tableyellow}{white}' in result:
                logger.debug("✓ Successfully inserted rowcolors{2} after \\begin{longtable}")
            else:
                logger.warning("✗ Failed to insert rowcolors after \\begin{longtable}")
        lines = result.split('\n')
        new_lines = []
        past_endhead = False
        for i, line in enumerate(lines):
            # Check if we're past the header section
            if '\\endhead' in line:
                past_endhead = True
                new_lines.append(line)
                continue
            
            # Add \hline BEFORE \tabularnewline for data rows (not in header, not before \bottomrule)
            if '\\tabularnewline' in line and past_endhead:
                # Check if next non-empty line is not \bottomrule or \end{longtable}
                # Find next non-empty line
                next_non_empty = None
                for j in range(i + 1, len(lines)):
                    if lines[j].strip():
                        next_non_empty = lines[j].strip()
                        break
            new_lines.append(line)
        
        result = '\n'.join(new_lines)
        return result
    
    # Pattern to match \begin{tabular}...\end{tabular}
    def add_rowcolors_to_tabular(match):
        full_table = match.group(0)
        logger.debug("Processing tabular")
        # Check if rowcolors already exists
        if '\\rowcolors' in full_table:
            logger.debug("rowcolors already exists in tabular, skipping")
            # Still need to add hdashline even if rowcolors exists
            pass
        else:
            logger.debug("Adding rowcolors to tabular")
            # Add \rowcolors{2}{white}{tableyellow} after \begin{tabular}{...}
            pattern = r'(\\begin{tabular}[^\n]+\n)'
            # replacement = r'\1\\rowcolors{2}{white}{tableyellow!20}\n\\arrayrulecolor{black}\n'
            replacement = r'\1\\arrayrulecolor{black}\n'
            full_table = re.sub(pattern, replacement, full_table, count=1)
        
        # Add solid lines between data rows using \midrule (booktabs compatible)
        lines = full_table.split('\n')
        new_lines = []
        for i, line in enumerate(lines):
            new_lines.append(line)
            # Add \midrule after \tabularnewline for data rows (not before \end{tabular})
            if '\\tabularnewline' in line:
                # Find next non-empty line
                next_non_empty = None
                for j in range(i + 1, len(lines)):
                    if lines[j].strip():
                        next_non_empty = lines[j].strip()
                        break
        result = '\n'.join(new_lines)
        return result
    
    # Process longtable environments (non-greedy match to get each table separately)
    longtable_count = len(re.findall(r'\\begin{longtable}', latex_content))
    logger.info(f"Found {longtable_count} longtable environment(s)")
    latex_content = re.sub(
        r'\\begin{longtable}.*?\\end{longtable}',
        add_rowcolors_to_longtable,
        latex_content,
        flags=re.DOTALL
    )
    
    # Process tabular environments (non-greedy match)
    tabular_count = len(re.findall(r'\\begin{tabular}', latex_content))
    logger.info(f"Found {tabular_count} tabular environment(s)")
    latex_content = re.sub(
        r'\\begin{tabular}.*?\\end{tabular}',
        add_rowcolors_to_tabular,
        latex_content,
        flags=re.DOTALL
    )
    
    logger.info("Table beautification completed")
    return latex_content

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: beautify_tables.py <latex_file>")
        sys.exit(1)
    
    latex_file = sys.argv[1]
    logger.info(f"Processing LaTeX file: {latex_file}")
    
    with open(latex_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    logger.debug(f"File size: {len(content)} characters")
    beautified = beautify_tables(content)
    
    with open(latex_file, 'w', encoding='utf-8') as f:
        f.write(beautified)
    
    logger.info(f"Successfully processed and saved: {latex_file}")
