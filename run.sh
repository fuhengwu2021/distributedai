#!/bin/bash
# Build full books and per-chapter PDFs; collect outputs under books/.
#
# Usage:
#   ./run.sh              # full book, lang from peanut.config (default: en)
#   ./run.sh --chapters   # also build per-chapter PDFs (ch. 1–12)
#   ./run.sh tc --template lulu_6x9 --cover 6x9
#   ./run.sh all          # cn, tc, and en (only when translations exist)
#   ./run.sh cn tc --chapters
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOKS="$ROOT/books"

cd "$ROOT"
mkdir -p "$BOOKS"

PDF_OPT=(--optimize-pdf --optimize-pdf-quality ebook)

# Detect langs from chapter filenames (bubble conventions: chapterN.md, chapterN_zh.md, …).
bubble_langs() {
  python3 - "$ROOT" "$@" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
cmd = sys.argv[2]
arg = sys.argv[3] if len(sys.argv) > 3 else ""

# bubble --lang code -> markdown filename suffix
LOCALES = {
    "en": "",
    "cn": "_zh",
    "tc": "_tc",
    "jp": "_jp",
    "sp": "_sp",
}
ORDER = ("en", "cn", "tc", "jp", "sp")

CHAPTER_NUM = re.compile(r"^chapter(\d+)\.md$")
CHAPTER_NUM_L10N = {
    lang: re.compile(rf"^chapter(\d+){re.escape(suffix)}\.md$")
    for lang, suffix in LOCALES.items()
    if suffix
}


def scan_langs() -> set[str]:
    found: set[str] = set()
    for chapter_dir in sorted(root.glob("chapter*/")):
        if not chapter_dir.is_dir():
            continue
        for md in chapter_dir.glob("*.md"):
            name = md.name
            if CHAPTER_NUM.match(name):
                found.add("en")
                continue
            for lang, pat in CHAPTER_NUM_L10N.items():
                if pat.match(name):
                    found.add(lang)
                    break
            if name == "chapterx.md":
                found.add("en")
            else:
                for lang, suffix in LOCALES.items():
                    if suffix and name in (f"chapterx{suffix}.md", f"preface{suffix}.md"):
                        found.add(lang)
    return found


def peanut_default_lang() -> str:
    cfg = root / "peanut.config"
    if cfg.exists():
        try:
            v = json.loads(cfg.read_text(encoding="utf-8")).get("lang", "en")
            if isinstance(v, str) and v.strip().lower() in LOCALES:
                return v.strip().lower()
        except Exception:
            pass
    return "en"


available = scan_langs()

if cmd == "list":
    print(" ".join(lang for lang in ORDER if lang in available))
elif cmd == "has":
    sys.exit(0 if arg in available else 1)
elif cmd == "default":
    preferred = peanut_default_lang()
    if preferred in available:
        print(preferred)
    elif "en" in available:
        print("en")
    elif available:
        print(next(lang for lang in ORDER if lang in available))
    else:
        print("en")
else:
    sys.exit(2)
PY
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS] [LANG ...]

Build full books (default). LANG may be:
  cn   simplified Chinese (chapterN_zh.md)
  tc   traditional Chinese (chapterN_tc.md)
  jp   Japanese (chapterN_jp.md)
  sp   Spanish (chapterN_sp.md)
  en   English (chapterN.md)
  all  every language with matching chapter files in this repo

Options:
  --chapters              Also build per-chapter PDFs (chapters 1–12) for each LANG
  --template NAME         Passed to bubble-build (e.g. lulu_6x9 or lulu_6x9.tpl)
  --cover SIZE            Passed to bubble-build for the cover build only (e.g. 6x9, 7x10)

Examples:
  $(basename "$0")                                    # lang from peanut.config
  $(basename "$0") --chapters                         # full book + chapters
  $(basename "$0") tc --template lulu_6x9 --cover 6x9
  $(basename "$0") all
  $(basename "$0") en
EOF
}

collect_to_books() {
  shopt -s nullglob
  for f in "$ROOT"/book*.pdf "$ROOT"/book*.epub "$ROOT"/book*.docx; do
    [ -f "$f" ] && mv -f "$f" "$BOOKS/"
  done
  for dir in "$ROOT"/chapter*-*/; do
    [ -d "$dir" ] || continue
    for f in "$dir"/chapter*.pdf; do
      [ -f "$f" ] && mv -f "$f" "$BOOKS/"
    done
  done
}

build_full_for_lang() {
  local lang="$1"
  local cover_args=()
  if [ ${#BUBBLE_COVER[@]} -gt 0 ]; then
    cover_args=("${BUBBLE_COVER[@]}")
  fi
  bubble-build --lang "$lang" --style square "${PDF_OPT[@]}" "${BUBBLE_COMMON[@]}" "${cover_args[@]}"
  bubble-build --lang "$lang" --style square --no-cover "${PDF_OPT[@]}" "${BUBBLE_COMMON[@]}"
  bubble-build --lang "$lang" --style none --no-cover "${PDF_OPT[@]}" "${BUBBLE_COMMON[@]}"
}

build_chapters_for_lang() {
  local lang="$1"
  local i
  for i in $(seq 1 12); do
    # bubble-convert "$i" --lang "$lang" --protect --ads "Above The Clouds for Prof Strang - Xuan Xin" --watermark "Confidential" --watermark-opacity 0.05 --style square
    bubble-convert "$i" --lang "$lang" --protect --ads "Above The Clouds - Xuan Xin" --style square "${PDF_OPT[@]}"
  done
}

parse_langs() {
  LANGS=()
  if [ $# -eq 0 ]; then
    LANGS=("$(bubble_langs default)")
    return
  fi

  local arg
  for arg in "$@"; do
    case "$arg" in
      -h|--help)
        usage
        exit 0
        ;;
      all)
        read -r -a LANGS <<< "$(bubble_langs list)"
        if [ ${#LANGS[@]} -eq 0 ]; then
          LANGS=("$(bubble_langs default)")
        fi
        return
        ;;
      en|cn|tc|jp|sp)
        if ! bubble_langs has "$arg"; then
          echo "No chapter markdown for lang '$arg' (scan chapter*/chapter*.md)." >&2
          continue
        fi
        local seen=0
        local existing
        for existing in "${LANGS[@]+"${LANGS[@]}"}"; do
          if [ "$existing" = "$arg" ]; then
            seen=1
            break
          fi
        done
        if [ "$seen" -eq 0 ]; then
          LANGS+=("$arg")
        fi
        ;;
      *)
        echo "Unknown lang: $arg" >&2
        usage >&2
        exit 1
        ;;
    esac
  done

  if [ ${#LANGS[@]} -eq 0 ]; then
    echo "No languages selected (no matching chapter*.md files)." >&2
    usage >&2
    exit 1
  fi
}

require_arg() {
  local opt="$1"
  local val="${2-}"
  if [ -z "$val" ] || [[ "$val" == --* ]]; then
    echo "Option $opt requires a value." >&2
    usage >&2
    exit 1
  fi
}

BUILD_CHAPTERS=0
BUBBLE_COMMON=()
BUBBLE_COVER=()
LANG_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --chapters)
      BUILD_CHAPTERS=1
      shift
      ;;
    --template)
      require_arg --template "${2-}"
      BUBBLE_COMMON+=(--template "$2")
      shift 2
      ;;
    --cover)
      require_arg --cover "${2-}"
      BUBBLE_COVER=(--cover "$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      LANG_ARGS+=("$1")
      shift
      ;;
  esac
done

parse_langs "${LANG_ARGS[@]}"

echo "Building languages: ${LANGS[*]}"
if [ "$BUILD_CHAPTERS" -eq 1 ]; then
  echo "Per-chapter PDFs: enabled (--chapters)"
else
  echo "Per-chapter PDFs: skipped (use --chapters to build)"
fi
if [ ${#BUBBLE_COMMON[@]} -gt 0 ]; then
  echo "bubble-build options: ${BUBBLE_COMMON[*]}"
fi
if [ ${#BUBBLE_COVER[@]} -gt 0 ]; then
  echo "Cover build: ${BUBBLE_COVER[*]} (interior builds use --no-cover)"
fi

for lang in "${LANGS[@]}"; do
  echo "==> Full book: $lang"
  build_full_for_lang "$lang"
done

if [ "$BUILD_CHAPTERS" -eq 1 ]; then
  for lang in "${LANGS[@]}"; do
    echo "==> Chapters 1-12: $lang"
    build_chapters_for_lang "$lang"
  done
fi

collect_to_books
echo "Outputs in: $BOOKS"
