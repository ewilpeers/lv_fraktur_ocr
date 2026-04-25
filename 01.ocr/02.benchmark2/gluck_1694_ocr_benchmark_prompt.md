# OCR Model Benchmark — Glück 1694 Latvian Bible (Printed Fraktur)

## Task
Find the best off-the-shelf OCR model for the 1694 Glück Latvian Bible. The
winner will later be fine-tuned against the existing gold-truth dataset. This
task is **model selection only** — no fine-tuning in this run.

## Data
- Gold set: 9 pages at `https://t.noit.pro/gluck_1694/gold/` (`00013.jpg` …
  `00021.jpg`, each with matching `<name>.jpg.txt`).
- Alphabet reference: 
  (~80 distinct codepoints, full list:
  `aeituhsnrẜmkwdoplſjbcꞥzgUKDSWTNIAGPłBâMEŗꞨàL‐RZëꞢꞣŁîꞡêOJßèäû?ùòꞤž/CHón̄`)
- Download all pages + txt to a local `gold/` directory at the start.
- Use **all 9 pages as eval**. The set is small — report per-page CER/WER
  and the standard deviation, not just the mean. A single bad page can dominate
  the average.

## Scope and non-goals
- **In scope:** 17C printed Fraktur/blackletter recognition of Latvian
  orthography with long-s (`ſ`, `ẜ`), Baltic diacritics
  (`ꞥ ꞧ ꞩ ꞣ ꞡ ꞤꞢ Ŗ ŗ ł Ł`), Latin diacritics (`â ê î ô û à è ë ò ù ó ä`),
  German `ß`, combining macron `n̄`, Unicode hyphen `‐` (U+2010).
- **Out of scope:** modern Latvian OCR models (`lav.traineddata`, modern
  Latvian Kraken models) — only include as a clearly-labeled out-of-domain
  reference if at all. Handwriting models — all Glück editions are printed.
  Layout reconstruction beyond Kraken's default two-column `blla` baseline
  segmentation.

## Primary engine
**Kraken** is the preferred target (the winner will be fine-tuned with
`ketos`). Also run a small non-Kraken baseline bracket for sanity:
- Tesseract with `frk` (German Fraktur) and the `Fraktur` LSTM model — these
  are already familiar from prior work; use them as a floor.
- Do not add other engines unless a Kraken model can't be sourced.

## Candidate models to evaluate

Pull candidates from:
1. `kraken list` (the Kraken model repository / Zenodo catalog)
2. The OCR-D model zoo (UB Mannheim, DFG-funded Fraktur models)
3. Transkribus public models exported to Kraken format, if available
Filter candidates by typeface family (Fraktur / blackletter), not by century 
or by label containing the word 'old' or 'historical'. 
A Roman-type model from 1500 is further from Glück 1694 than a Fraktur model from 1850.

Minimum set to benchmark (adjust based on what's actually available at run
time; do not invent model names — verify each exists before downloading):
- A general German Fraktur Kraken model (e.g. `german_print_best` or
  current equivalent)
- A 17–18C historical German print Kraken model if one exists
- An `austriannewspapers`-style 19C Fraktur model as a secondary reference
- Any OCR-D Fraktur GT4HistOCR-trained model available for Kraken
- Tesseract `frk` (baseline)
- Tesseract `Fraktur` (baseline)

If fewer than 3 Kraken candidates are findable, say so explicitly and stop to
confirm before proceeding.

## Environment
- Single shared conda environment, already activated. **Do not run
  `conda activate` or `conda deactivate`** — they fail in this setup. Install
  packages into the current env with `pip install` (or `mamba install -n
  <current-env-name>`). Check active env with `conda info --envs | grep '*'`
  if needed.
- Hardware: NVIDIA A100 (Ampere, tensor cores available). Kraken defaults
  to CPU — pass `--device cuda:0` explicitly for all runs.
- At the top of any PyTorch/Kraken benchmark script, set:
  ```python
  import torch
  torch.set_float32_matmul_precision('high')
  ```
  This addresses the Ampere matmul-precision warning seen in earlier runs.

## Segmentation
- Use Kraken default baseline segmenter (`blla`) on all pages. Do not pre-split
  columns manually.
- If `blla` produces visibly bad line cuts on any page (report these),
  additionally try the legacy `box` segmenter on those pages and note the
  difference.
- Do not change segmentation per-model — segmentation must be held constant
  across the comparison. Recognition is the variable being tested.

## Evaluation metrics

**Primary:** Character Error Rate (CER), computed with:
- **No Unicode normalization.** Do not apply NFC/NFKC/NFD. `ẜ` must stay
  `ẜ`, `ſ` must stay `ſ`, and must never be folded to `s`.
- **No case folding.**
- **No whitespace collapsing beyond single-space.** Preserve newlines as
  newlines in line-by-line evaluation.
- Match OCR output to gold line-by-line after Kraken's own line segmentation.
  If the number of lines differs, report it — do not silently globally align.

**Secondary:**
- WER (word-level).
- Per-character confusion matrix — top 20 substitutions, insertions, and
  deletions. Pay explicit attention to:
  - `ſ` ↔ `f` (long-s / f confusion — the known hard case)
  - `ſ` ↔ `ẜ` (plain long-s vs descender long-s)
  - `ꞥ` / `ꞧ` / `ꞩ` / `Ꞩ` / `Ŗ` / `ŗ` (Baltic-specific letters — likely
    missing from most models' codecs and mapped to nearest Latin char)
  - `ß` ↔ `ſs`
  - `/` → `,` or `.` (slash is used as a sentence separator in Glück)
  - `‐` (U+2010) vs `-` (U+002D)
- Throughput: pages/minute on the A100 (tie-breaker only).

**Codec coverage check:** before benchmarking, for each candidate model, dump
the model's output alphabet (`kraken show <model>`) and compute set
difference against `alphabet.txt`. Report which gold characters the model
*cannot produce* at all — these are guaranteed errors and matter for the
fine-tuning plan (they tell us whether we'll need `ketos train --resize add`
or `--resize both`).

## Deliverables

1. **Results table** (markdown), one row per model:

   | Model | Source | Mean CER | CER stdev | WER | Pages/min | Missing chars |

2. **Top 5 confusions per model** — a small table per model showing the most
   common substitution/insertion/deletion pairs.

3. **Recommendation:** one winner + one runner-up, with a 2–3 sentence
   rationale. If two models are within 0.5 CER points, prefer the one with
   better codec coverage of Baltic characters.

4. **Reproduction block:** exact shell commands used (download, segment,
   recognize, score) so the run can be re-executed.

5. **Fine-tuning preamble for the winner:**
   - Model's VGSL architecture spec
   - Current codec size vs. target alphabet size
   - Recommended `ketos train --resize` mode (`add` if target is a
     superset, `both` if there's partial overlap with characters to drop)

6. **Saved predictions:** per-page OCR output in a `predictions/<model>/`
   directory, so the gold-vs-pred diff can be inspected later.

## Things to flag back before or during the run
- If fewer than 3 Kraken candidates can be sourced.
- If any gold `.txt` file fails to download or appears malformed.
- If segmentation fails on a page (blla crashes or returns 0 lines).
- Any warning beyond the matmul-precision one — surface it, don't swallow.

## Non-negotiables
- No Unicode normalization anywhere in the pipeline.
- No modern Latvian models treated as in-domain baselines.
- Segmentation held constant across all models being compared.
- Don't invent model names — verify each exists in the Kraken repo or a
  reachable URL before downloading.
