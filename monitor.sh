#!/bin/bash

# Monitor script to watch chapterX.md files and auto-convert to PDF
# Usage:
#   ./monitor.sh                    # Monitor all chapters
#   ./monitor.sh 1                   # Monitor only chapter 1
#   ./monitor.sh 1 2 3               # Monitor specific chapters

# Get the script directory (root of the project)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONVERT_SCRIPT="$SCRIPT_DIR/convert_to_pdf.sh"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to extract chapter number from path
get_chapter_number() {
    local path="$1"
    # Extract chapter number from path like chapter1-... or chapter10-...
    if [[ "$path" =~ chapter([0-9]+) ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo ""
    fi
}

# Function to convert a specific chapter
convert_chapter() {
    local chapter_num="$1"
    local chapter_name="$2"
    
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} Detected change in ${YELLOW}$chapter_name${NC}"
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} Converting chapter $chapter_num to PDF..."
    
    if "$CONVERT_SCRIPT" "$chapter_num" > /tmp/convert_${chapter_num}.log 2>&1; then
        echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} ✓ Successfully converted chapter $chapter_num"
    else
        echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} ✗ Conversion failed for chapter $chapter_num"
        echo -e "${YELLOW}Check log: /tmp/convert_${chapter_num}.log${NC}"
    fi
    echo ""
}

# Function to convert appendix
convert_appendix() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} Detected change in ${YELLOW}chapterx/chapterx.md${NC}"
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} Converting appendix to PDF..."
    
    if [ -f "chapterx/chapterx.md" ]; then
        if "$CONVERT_SCRIPT" chapterx/chapterx.md > /tmp/convert_appendix.log 2>&1; then
            echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} ✓ Successfully converted appendix"
        else
            echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} ✗ Conversion failed for appendix"
            echo -e "${YELLOW}Check log: /tmp/convert_appendix.log${NC}"
        fi
    else
        echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} ✗ Appendix file not found: chapterx/chapterx.md"
    fi
    echo ""
}

# Function to monitor files
monitor_files() {
    local watch_dirs=()
    
    # Always include chapterx directory for appendix
    if [ -d "chapterx" ]; then
        watch_dirs+=("chapterx")
        echo -e "${GREEN}Monitoring:${NC} chapterx/chapterx.md (Appendix)"
    fi
    
    # If specific chapters are provided, monitor only those
    if [ $# -gt 0 ]; then
        for chapter_num in "$@"; do
            # Find chapter directory
            chapter_dir=$(find . -maxdepth 1 -type d -name "chapter${chapter_num}-*" | head -1)
            if [ -n "$chapter_dir" ] && [ -f "$chapter_dir/chapter${chapter_num}.md" ]; then
                watch_dirs+=("$chapter_dir")
                echo -e "${GREEN}Monitoring:${NC} $chapter_dir/chapter${chapter_num}.md"
            else
                echo -e "${YELLOW}Warning:${NC} Chapter $chapter_num not found or chapter${chapter_num}.md missing"
            fi
        done
    else
        # Monitor all chapters
        echo -e "${GREEN}Monitoring all chapters...${NC}"
        while IFS= read -r -d '' dir; do
            chapter_num=$(get_chapter_number "$dir")
            if [ -n "$chapter_num" ] && [ -f "$dir/chapter${chapter_num}.md" ]; then
                watch_dirs+=("$dir")
                echo -e "${GREEN}Monitoring:${NC} $dir/chapter${chapter_num}.md (Chapter $chapter_num)"
            fi
        done < <(find . -maxdepth 1 -type d -name "chapter*" -print0 | sort -z)
    fi
    
    if [ ${#watch_dirs[@]} -eq 0 ]; then
        echo -e "${YELLOW}No chapters to monitor. Exiting.${NC}"
        exit 1
    fi
    
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}File monitor started${NC}"
    echo -e "${BLUE}Press Ctrl+C to stop${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    
    # Use inotifywait to monitor file changes
    inotifywait -m -r -e close_write,moved_to --format '%w%f' "${watch_dirs[@]}" 2>/dev/null | while read -r file; do
        # Process appendix file
        if [[ "$file" =~ /chapterx/chapterx\.md$ ]]; then
            # Small delay to ensure file is fully written
            sleep 0.5
            convert_appendix
        # Process chapterX.md files
        elif [[ "$file" =~ /chapter([0-9]+)\.md$ ]]; then
            chapter_num="${BASH_REMATCH[1]}"
            chapter_dir=$(dirname "$file")
            chapter_name=$(basename "$chapter_dir")
            
            if [ -n "$chapter_num" ]; then
                # Small delay to ensure file is fully written
                sleep 0.5
                convert_chapter "$chapter_num" "$chapter_name"
            fi
        fi
    done
}

# Check if convert_to_pdf.sh exists
if [ ! -f "$CONVERT_SCRIPT" ]; then
    echo -e "${YELLOW}Error:${NC} convert_to_pdf.sh not found at $CONVERT_SCRIPT"
    exit 1
fi

# Check if inotifywait is available
if ! command -v inotifywait &> /dev/null; then
    echo -e "${YELLOW}Error:${NC} inotifywait not found. Please install inotify-tools:"
    echo "  sudo apt-get install inotify-tools  # Ubuntu/Debian"
    echo "  sudo yum install inotify-tools     # CentOS/RHEL"
    exit 1
fi

# Start monitoring
monitor_files "$@"
