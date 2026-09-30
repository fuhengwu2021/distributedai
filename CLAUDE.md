use conda env usao

# Distributed AI Systems — Instructions for Claude & AI Agents

## Python Environment: Mandatory `usao`

Always use the `usao` conda environment for all Python, PyTorch, vLLM, SGLang, Django, pytest, pip, diagram generation, and book compilation commands (`bubble-batch`) in this repository.

- **Do NOT create new virtual environments**: Never create `venv`, `virtualenv`, or other isolated environments in this repo.
- **Do NOT hardcode environment absolute paths**: Avoid hardcoding paths like `/home/.../miniconda3/envs/usao/bin/python`.
- **Subshell invocation pattern**:
  ```bash
  source /home/wukong/miniconda3/etc/profile.d/conda.sh && conda activate usao
  ```
  Or invoke dynamically via:
  ```bash
  conda run -n usao <command>
  ```
- **Conda run stdin caveat**: `conda run` does NOT forward stdin. Always write python code to a script file first, then run `conda run -n usao python path/to/script.py`.

## Book Building (`bubble-batch`)

`bubble-batch` is the core book and per-chapter build tool installed in `usao` (from `peanutbook` / `bubble`):

```bash
# Build Chinese full book with standard square chapter headers
bubble-batch --lang cn --style square --chapter-opener-size 1.5 --j 4
# Or via conda run:
conda run -n usao bubble-batch --lang cn --style square --chapter-opener-size 1.5 --j 4

# Build English full book
bubble-batch --lang en --style square --chapter-opener-size 1.5 --j 4
```

## Bilingual Figure Localization Standard (`~/mmb` Pattern)

Following the standard established in `~/mmb` (`mortgagekit/figstyle.py`), all figures support bilingual localization (English & Chinese) without code duplication.

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

### 3. Output & Markdown Convention
- Running the script once generates both:
  - English: `<stem>.png` and `<stem>.pdf`
  - Chinese: `<stem>_zh.png` and `<stem>_zh.pdf`
- English chapter files (`chapterN.md`): `![Caption](img/my_figure_stem.png){...}`
- Chinese chapter files (`chapterN_zh.md`): `![中文图题](img/my_figure_stem_zh.png){...}`

## Cross-Platform Agent Skills

This repository maintains cross-platform skill specifications synchronized across:
- Claude Code workspace skills: `.claude/skills/python-env/SKILL.md`
- Claude global skills: `~/.claude/skills/python-env/SKILL.md`
- Universal agent workspace skills: `.agents/skills/python-env/SKILL.md`
- Antigravity global skills: `~/.gemini/config/skills/python-env/SKILL.md`
