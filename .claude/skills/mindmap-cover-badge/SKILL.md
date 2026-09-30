---
name: mindmap-cover-badge
description: Adds the book cover thumbnail badge to a chapter's mindmap.png in the top-left corner (light-blue border, 16% width, auto-synced thumbnail), matching the style already applied to chapters 1, 3, 4, and 5. Use when asked to add/apply the book cover to a chapter's mindmap image, or a follow-up like "do the same for chapter N".
---

# Mindmap Cover Badge

Pastes a small bordered book-cover thumbnail into the top-left corner of a chapter's `mindmap.png`, using the `bubble-p2p` peanutbook command (built specifically for this). Already applied to chapters 1, 3, 4, 5 as of this writing — check the target chapter's `mindmap.png` before assuming it's missing.

## Steps

1. **Resolve the path.** Find the chapter directory: `chapter<N>-*/img/mindmap.png` (glob, since the directory name includes a slug after the number, e.g. `chapter3-distributed-training-with-pytorch-ddp`).

2. **Check the top-left corner is clear** before editing, so the badge won't land on tree/branch content:
   ```python
   from PIL import Image
   img = Image.open("<chapter-dir>/img/mindmap.png")
   img.crop((0, 0, 900, 700)).save("<scratchpad>/topleft_check.png")
   ```
   Read the crop. Every chapter so far has had clear top-left space (mindmaps are laid out left-to-right starting from a root node lower down), but always verify — layouts vary per chapter and this isn't guaranteed.

3. **Run the merge command**, from the `usao` conda env where `bubble-p2p` is installed:
   ```bash
   source /home/wukong/miniconda3/etc/profile.d/conda.sh && conda activate usao
   bubble-p2p <chapter-dir>/img/mindmap.png /home/wukong/distributedai/cover/7x10/amazon_cover.jpg
   ```
   Use the defaults (matching every prior chapter) unless there's a specific reason not to: top-left corner, badge width = 16% of the poster's width, 40px margin, light-blue border (`#c9d7ea` — deliberately not black, see below). The command also auto-detects and re-syncs the sibling `mindmap_thumb.jpg` in the same directory — no separate step needed.

4. **Verify.** Re-crop the top-left corner to confirm the badge sits cleanly with no overlap into the tree, and downscale-view the full image to confirm nothing else shifted or clipped.

5. **Report** which chapter was updated and confirm the thumbnail was synced (the command prints `synced thumbnail: ...` when it does).

## Notes

- `bubble-p2p` lives at `~/peanutbook/scripts/picture_to_picture.py`, part of the editable `peanutbook` package. If the command isn't found: `cd ~/peanutbook && pip install -e . --no-deps -q` inside the `usao` env re-registers it.
- Always pass the full-resolution `cover/7x10/amazon_cover.jpg` — `bubble-p2p` handles resizing itself; don't pre-resize.
- The border is intentionally light (`#c9d7ea`), not black — a black border was tried first and read as too heavy against the mindmap's white background.
- If a chapter's top-left corner isn't clear, don't force it — either pass `--corner top-right` (or another corner `bubble-p2p --help` lists) or check with the user before deviating from the established top-left convention used by every other chapter.
- Full command reference: `bubble-p2p --help`.
