# Vertical-line page splitter

Cuts two-column scanned pages along their printed vertical divider, whitens the line, pads the curved-cut wedges with `#E01AD1`, and stacks the right column under the left.

## Files

| File | What it is |
|---|---|
| `line_detection.py` | Shared algorithm (seed search, trace, JSON I/O). Imported by both the notebook and the GUI. |
| `split_on_vertical_line.ipynb` | Jupyter notebook for batch processing. |
| `correction_tool.py` | Tkinter GUI for manually correcting individual pages. |

All three must live in the same folder (the GUI and notebook both `import line_detection`).

## Quick start

```
pip install numpy pillow opencv-python matplotlib jupyter
# python3-tk is needed for the GUI on Linux: apt install python3-tk
```

### 1. Batch-process a folder (notebook)

Open `split_on_vertical_line.ipynb`, set `SOURCE_DIR` and `TARGET_DIR` in the last cell, run all cells. On a cold run it auto-detects the divider for every image and writes:

- `<SOURCE_DIR>/starts.json` — seed x for each image, all `autogen: true`
- `<SOURCE_DIR>/overlays.json` — full traced curve for each image, all `autogen: true`
- `<TARGET_DIR>/<name>_split.png` — the stacked output
- `<TARGET_DIR>/_debug/<name>_overlay.png` — the source image with the red traced line burned in (for spot-checking)

### 2. Fix the ones that went wrong (GUI)

```
python correction_tool.py /path/to/source_folder
```

(or run with no argument and a folder picker pops up)

In the GUI:

- **← / →** or buttons — navigate images
- **Status bar** — `page (x, y)` and `Δ from line: ±N px` follow the mouse, GIMP-style
- **Click anywhere on the image** — sets that x as the new seed and re-traces immediately
- **Shift ± px box** — type `+12` or `-8` and press Enter to nudge the seed
- **Seed x box** — type an absolute pixel column and press Enter
- **Re-detect (auto)** — throw away your edits for this page, run auto again
- **Save all** — writes the modified `starts.json` and `overlays.json`

A page is "viewed" the moment you navigate to it; unviewed pages don't get entries in the saved JSONs (per spec rule 5).

When you adjust a page, both files mark that entry `autogen: false`. The trace is re-run with the same algorithm the notebook uses, so what you see in the GUI is what you'll get.

### 3. Re-run the notebook

The notebook now picks up the manual corrections automatically. The precedence table:

| starts entry            | overlays entry          | Notebook does                              |
|-------------------------|-------------------------|--------------------------------------------|
| —                       | —                       | Auto-detect from scratch                   |
| autogen / manual        | —                       | Re-trace from the given seed_x             |
| —                       | autogen / manual        | Use the precomputed overlay as-is          |
| autogen                 | autogen                 | Use the precomputed overlay (consistent)   |
| **manual**              | autogen                 | **Manual wins — re-trace from seed_x**     |
| any                     | manual                  | Use the precomputed (manual) overlay       |

After the notebook runs again, both JSONs are rewritten reflecting what was used. `autogen: false` flags persist across runs.

## Tuning per edition

In the notebook's `CONFIG` cell (and the GUI implicitly via `DEFAULT_CONFIG`):

- **`dark_threshold`** (default 140) — raise for faded lines, lower if too much paper counts as dark
- **`gray_mode`** — `'gray'` (default), or `'blue'` for brown-paper originals (use with `dark_threshold` ~50)
- **`search_band`** — `(0.30, 0.70)` central band; tighten if the seed locks onto a non-divider column
- **`max_drift_from_seed`** — 15 px; lower if the trace wanders into nearby text
- **`max_gap_rows`** — 200 rows; raise for older paper with longer faded sections
- **`flank_width`** — 30 px; the "thin spike" half-width that distinguishes a divider from text columns

## Common issues

- **Seed locked onto a text column** — narrow `search_band`, or use the GUI to click on the actual divider.
- **Trace stops short top/bottom** — bump `max_gap_rows` (200 → 400).
- **Wandering line** — reduce `max_drift_from_seed` (15 → 8) or `trace_max_jump` (4 → 3).
- **Output cut goes through letters** — the trace was off; correct in the GUI. The whitening step skips its widening near letters but the cut x itself comes from the trace.
