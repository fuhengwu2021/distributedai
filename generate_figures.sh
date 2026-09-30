#!/bin/bash
# Regenerate chapter figures: run every chapterN-*/img/<name>.py that has no
# matching <name>.png yet (same rule the book build uses).
# Adopted from ~/mmb standard.
#
# Examples:
#   ./generate_figures.sh              # only missing figures, all chapters
#   ./generate_figures.sh -f           # force: redraw everything
#   ./generate_figures.sh -c 3 -f      # force chapter 3 only
#   ./generate_figures.sh -c 1-3       # chapters 1 through 3
set -euo pipefail

PYTHON="${PYTHON:-python}"
FORCE=0
CHAPTERS=""

usage() {
    sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -f|--force) FORCE=1; shift ;;
        -c|--chapters) CHAPTERS="$2"; shift 2 ;;
        -h|--help) usage 0 ;;
        *) echo "unknown option: $1" >&2; usage 1 ;;
    esac
done

# Expand "1-3" / "1,3" / "3" into a match list; empty means every chapter.
selected() {
    local dir="$1" num
    num="$(basename "$dir" | sed -n 's/^chapter\([0-9]\+\)-.*/\1/p')"
    [[ -z "$CHAPTERS" ]] && return 0
    [[ -z "$num" ]] && return 1
    local spec
    for spec in ${CHAPTERS//,/ }; do
        if [[ "$spec" == *-* ]]; then
            (( num >= ${spec%-*} && num <= ${spec#*-} )) && return 0
        elif [[ "$num" == "$spec" ]]; then
            return 0
        fi
    done
    return 1
}

cd "$(dirname "$0")"
made=0
skipped=0
for img_dir in chapter*/img; do
    selected "$(dirname "$img_dir")" || continue
    for script in "$img_dir"/*.py; do
        [[ -e "$script" ]] || continue
        name="$(basename "$script" .py)"
        [[ "$name" == _* ]] && continue          # shared helper, not a figure
        [[ "$name" == "mindmap" ]] && continue   # mindmap is generated differently
        if [[ $FORCE -eq 0 && -f "$img_dir/$name.png" ]]; then
            if ! grep -q "localized_figure" "$script" || [[ -f "$img_dir/${name}_zh.png" ]]; then
                skipped=$((skipped + 1))
                continue
            fi
        fi
        echo "==> $script"
        (cd "$img_dir" && "$PYTHON" "$name.py")
        made=$((made + 1))
    done
done
echo "generated $made figure(s), skipped $skipped up-to-date figure(s)"

