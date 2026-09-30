---
name: python-env
description: Guidelines and specifications for Python environment usage. Always use conda env usao (via `conda run -n usao ...` or initializing `conda activate usao`) to execute Python scripts, tests, commands, and book builds (bubble-batch) in this repository. Includes bilingual figure localization standards from ~/mmb.
---

# Python Environment & Book Building Standard: conda env usao

## Mandatory Environment: `usao`

Always use the `usao` conda environment for all Python, PyTorch, vLLM, SGLang, Django, pytest, pip, diagram generation, and book compilation commands (`bubble-batch`) in this repository.

- **Do NOT create new virtual environments**: Never create `venv`, `virtualenv`, or other isolated environments in this repository.
- **Do NOT hardcode environment absolute paths**: Avoid hardcoding `/home/.../miniconda3/envs/usao/bin/python` in scripts unless strictly required. Always execute dynamically via `conda run -n usao` or after activating `usao`.

## Shell & Subshell Initialization

In non-interactive bash subshells where `conda` is not in `$PATH` by default, initialize conda via the official setup hook before activating or running commands:

```bash
# Recommended subshell invocation pattern
source /home/wukong/miniconda3/etc/profile.d/conda.sh && conda activate usao
```

Or invoke conda directly via its absolute executable path:

```bash
/home/wukong/miniconda3/bin/conda run -n usao <command>
```

## Standard Invocation Patterns

### 1. Python Scripts & Testing
```bash
# Execute Python scripts
conda run -n usao python <script_or_command>

# Testing & Pip package management
conda run -n usao pytest <args>
conda run -n usao pip <args>
```

### 2. Book Building (`bubble-batch`)
`bubble-batch` is the core book and per-chapter build tool installed in the `usao` environment (from `peanutbook` / `bubble`):

```bash
# Build Chinese full book with standard square chapter headers
bubble-batch --lang cn --style square --chapter-opener-size 1.5 --j 4
# Or via conda run:
conda run -n usao bubble-batch --lang cn --style square --chapter-opener-size 1.5 --j 4

# Build English full book
bubble-batch --lang en --style square --chapter-opener-size 1.5 --j 4
```

## Bilingual Figure Localization Standard (`~/mmb` Pattern)

Following the gold standard from `~/mmb` (`mortgagekit/figstyle.py`), figures must support bilingual localization (English & Chinese) without code duplication.

### 1. English Figure Fidelity & Visual Diff Verification (英文原图保真与比对验证)
**CRITICAL**: Do NOT assume newly generated English figures are automatically identical. Matplotlib rendering is sensitive to global `rcParams`, font stacks, and `bbox_inches='tight'` calculations.
- **Visual Regression Verification**: When refactoring an existing figure script, always backup the original PNG first and run a pixel/dimension comparison (`ImageChops.difference`) against the regenerated English PNG.
- **Exact text match**: `LABELS["en"]` must copy the exact strings, line breaks, punctuation, and capitalization from the original English code.
- **Environment isolation**: Always execute each language pass inside `with plt.rc_context():` so that CJK font injection or unicode configuration for Chinese never leaks into or alters the English rendering environment.
- **Do not overwrite chart defaults**: Never globally disable grids, change spines, or alter ticks in shared style functions, as this would distort existing charts (e.g. loss curves, throughput benchmarks).

### 2. Single Implementation, Multiple Outputs
**Never** create separate scripts like `foo.py` and `foo_zh.py`. Duplicate scripts cause layout, coordinate, and style drift. Instead, implement a single script per figure using `localized_figure`:

```python
from figstyle import localized_figure

LABELS = {
    "en": {
        "title": "Production Ingress Traffic",
        "desc": "HTTP / gRPC Requests (100% Live Workload)",
        ...
    },
    "zh": {
        "title": "生产环境入站流量",
        "desc": "HTTP / gRPC 外部请求 (100% 全量生产负载)",
        ...
    }
}

def draw(text: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(...))
    # Draw using text["title"], text["desc"], etc.
    ...
    fig.tight_layout()
    return fig

if __name__ == "__main__":
    localized_figure(draw, "my_figure_stem", LABELS, __file__)
```

### 3. Output Convention
Running the script once generates both language assets:
- English: `<stem>.png` and `<stem>.pdf`
- Chinese: `<stem>_zh.png` and `<stem>_zh.pdf`

### 4. Markdown Reference Convention
- In English chapter files (`chapterN.md`):
  ```markdown
  ![Caption](img/my_figure_stem.png){#fig:my-figure ...}
  ```
- In Chinese chapter files (`chapterN_zh.md`):
  ```markdown
  ![中文图题](img/my_figure_stem_zh.png){#fig:my-figure ...}
  ```

### 5. Font & Typography Settings
- For Chinese figures (`lang="zh"`), load CJK fonts in priority order:
  `["Noto Sans CJK SC", "Noto Serif CJK SC", "WenQuanYi Micro Hei", "WenQuanYi Zen Hei"]`
- Set `axes.unicode_minus = False` to prevent minus signs rendering as broken squares.
- Avoid non-ASCII emoji glyphs in matplotlib text (e.g. 🚨) to avoid LaTeX STIXGeneral font compilation errors.

## Conda Run Caveats

- **`conda run` does NOT forward stdin**: Commands using stdin piping or heredocs like `conda run -n usao python - <<EOF` will silently execute nothing without returning an error.
  - **Always write scripts to a file first**: Save Python code into a script file (e.g. `script.py` or a temporary `.py` file) and then execute it:
    ```bash
    conda run -n usao python path/to/script.py
    ```

## Dynamic Path Resolution

When a tool or configuration strictly demands an interpreter path rather than a shell wrapper, resolve it dynamically:
```bash
"$(conda info --base)/envs/usao/bin/python" <script_or_command>
```

## Search Restrictions

**NEVER** search in user home `~` or system root `/` (e.g., `find ~`, `find /`, `grep -r ... /`, `grep -r ... ~`). Searches must always be scoped strictly to the specific project directory or workspace.

## Django & Test Execution

For Django management and testing in this repository:

```bash
# Run all tests
conda run -n usao python web/manage.py test

# Run tests for specific apps
conda run -n usao python web/manage.py test chapters accounts
```
