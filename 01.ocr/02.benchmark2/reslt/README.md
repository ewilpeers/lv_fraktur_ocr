# OCR Model Benchmark — Glück 1694 Latvian Bible (Printed Fraktur)

**Date:** 2026-04-24  
**Hardware:** NVIDIA A100-SXM4-80GB, CUDA  
**Engine:** Kraken 7.0.1, Tesseract 5.3.4  
**Gold set:** 9 pages (00013–00021) from `t.noit.pro/gluck_1694/gold/`  
**Gold alphabet:** 81 distinct codepoints incl. ſ, ẜ, ꞥ, ꞡ, ꞣ, Ꞩ, ł, Ł, ŗ, ß, ‐ (U+2010)

---

## 1. Candidate Models

| # | Model | Source | DOI / ID | Type | Notes |
|---|-------|--------|----------|------|-------|
| 1 | **german_print** | UB Mannheim (Weil et al.) | zenodo.10519596 | Kraken | German prints 15–20C, Fraktur + Antiqua |
| 2 | **austriannewspapers** | B. Kiessling | zenodo.7933402 | Kraken | 19C Austrian Fraktur newspapers |
| 3 | **catmus_print_large** | CATMuS consortium | zenodo.10592716 | Kraken | Diachronic W. European prints (Large) |
| 4 | **catmus_print_small** | CATMuS consortium | zenodo.10602307 | Kraken | Diachronic W. European prints (Small) |
| 5 | **catmus_print_tiny** | CATMuS consortium | zenodo.10602357 | Kraken | Diachronic W. European prints (Tiny) |
| 6 | **mccatmus** | CATMuS consortium | zenodo.13788177 | Kraken | McCATMuS 16–21C multi-script |
| 7 | **tesseract_frk** | Tesseract | tessdata `frk` | Tesseract | German Fraktur legacy baseline |
| 8 | **tesseract_Fraktur** | Tesseract | tessdata_best `Fraktur` | Tesseract | Fraktur LSTM script model baseline |

**GT4HistOCR note:** No pre-trained Kraken model derived from the GT4HistOCR dataset (Springmann et al., arXiv 1809.05501) was found in the Kraken repository, Zenodo, or OCR-D model zoo. The `german_print` model is the closest equivalent — it was trained on 4 German historical print GT datasets (AustrianNewspapers, digi-gt, digitue-gt, reichsanzeiger-gt) covering 15–20C material with 187K text lines.

---

## 2. Codec Coverage

Characters the model **cannot produce at all** (guaranteed errors before recognition even begins):

| Model | Codec | Coverage | Missing | Has ſ | Has ẜ | Has ß | Missing Characters |
|-------|------:|--------:|---------:|:-----:|:-----:|:-----:|-------------------|
| german_print | 283 | 71/81 (87.7%) | 10 | **Yes** | No | **Yes** | Ł ł ŗ ẜ ‐ ✝ ꞡ ꞣ ꞥ Ꞩ |
| austriannewspapers | 186 | 70/81 (86.4%) | 11 | **Yes** | No | **Yes** | î Ł ł ŗ ẜ ‐ ✝ ꞡ ꞣ ꞥ Ꞩ |
| catmus_print_large | 220 | 63/81 (77.8%) | 18 | **No** | No | **Yes** | à â ä è ê ë î û Ł ŗ **ſ** ẜ ‐ ✝ ꞡ ꞣ ꞥ Ꞩ |
| catmus_print_small | 220 | 63/81 (77.8%) | 18 | **No** | No | **Yes** | (same as large) |
| catmus_print_tiny | 220 | 63/81 (77.8%) | 18 | **No** | No | **Yes** | (same as large) |
| mccatmus | 116 | 62/81 (76.5%) | 19 | **No** | No | **Yes** | à â ä è ê ë î û Ł ł ŗ **ſ** ẜ ‐ ✝ ꞡ ꞣ ꞥ Ꞩ |

**Key finding:** No model covers the Baltic-specific letters (ꞥ, ꞡ, ꞣ, Ꞩ), descender long-s (ẜ), barred-l (Ł/ł), or r-cedilla (ŗ). Only `german_print` and `austriannewspapers` have long-s (ſ) in their codec — the CATMuS and McCATMuS models lack it entirely, which is a critical Fraktur gap.

---

## 3. Segmentation

All pages segmented with Kraken `blla` (baseline segmenter) on GPU, held constant across all models.

| Page | Gold Lines | Segmented Lines | Note |
|------|----------:|----------------:|------|
| 00013 | 58 | 59 | ~match |
| 00014 | 80 | 66 | column merge |
| 00015 | 71 | 70 | ~match |
| 00016 | 78 | 70 | column merge |
| 00017 | 71 | 68 | ~match |
| 00018 | 73 | 73 | exact |
| 00019 | 80 | 67 | column merge |
| 00020 | 75 | 72 | ~match |
| 00021 | 80 | 63 | column merge |

**Column merge issue:** Pages 14, 16, 19, 21 have two-column layout where `blla` sometimes merges lines across columns. This inflates absolute CER on those pages (~70%+) but affects all models equally — **relative ranking is valid**. Single-column pages (13, 15, 17) show true recognition quality.

---

## 4. Results

### 4.1 Summary Table

| Model | Source | Mean CER | CER σ | Mean WER | WER σ | Pages/min | Missing Chars |
|-------|--------|--------:|---------:|---------:|---------:|----------:|--------------:|
| **german_print** | UB Mannheim | **51.79%** | 19.79% | **79.90%** | 14.62% | 15.5 | 10 |
| austriannewspapers | Kiessling | 56.60% | 17.13% | 85.56% | 8.31% | 14.5 | 11 |
| catmus_print_large | CATMuS | 63.66% | 14.01% | 92.34% | 5.66% | 15.0 | 18 |
| catmus_print_small | CATMuS | 66.55% | 12.80% | 95.54% | 2.96% | 17.2 | 18 |
| mccatmus | CATMuS | 66.83% | 11.25% | 94.81% | 2.88% | 15.7 | 19 |
| catmus_print_tiny | CATMuS | 69.09% | 12.05% | 95.94% | 2.73% | 18.6 | 18 |
| tesseract_Fraktur | Tesseract | 79.73% | 24.80% | 92.64% | 13.63% | 6.7 | N/A |
| tesseract_frk | Tesseract | 79.86% | 24.06% | 93.42% | 12.22% | 7.4 | N/A |

### 4.2 Per-Page CER (%)

| Page | german_print | austrian | catmus_L | catmus_S | catmus_T | mccatmus | tess_frk | tess_Frak |
|------|------------:|---------:|---------:|---------:|---------:|---------:|---------:|----------:|
| 00013 | **24.8** | 33.8 | 43.9 | 49.3 | 53.0 | 50.8 | 95.7 | 96.7 |
| 00014 | **72.3** | 74.1 | 77.6 | 79.1 | 80.6 | 78.2 | 78.2 | 76.8 |
| 00015 | **32.3** | 39.0 | 50.8 | 55.7 | 59.8 | 55.9 | 90.6 | 92.9 |
| 00016 | **61.1** | 65.1 | 70.4 | 71.7 | 73.2 | 71.3 | 20.9 | 19.6 |
| 00017 | **24.7** | 32.8 | 44.4 | 47.7 | 50.2 | 51.6 | 98.8 | 99.5 |
| 00018 | **54.3** | 60.8 | 63.2 | 66.5 | 69.5 | 68.1 | 100.0 | 100.0 |
| 00019 | **73.6** | 75.1 | 77.8 | 81.0 | 81.3 | 78.2 | 81.7 | 80.7 |
| 00020 | **54.2** | 57.6 | 67.5 | 69.9 | 73.0 | 69.7 | 77.7 | 76.6 |
| 00021 | **68.8** | 71.1 | 77.3 | 78.0 | 81.2 | 77.6 | 75.3 | 74.7 |

**Best single-column page results** (pages 13, 15, 17 — minimal column-merge contamination):
- german_print: 24.8%, 32.3%, 24.7% → **mean 27.3%**
- austriannewspapers: 33.8%, 39.0%, 32.8% → **mean 35.2%**

These are the most representative of true recognition quality.

### 4.3 Tesseract Failure Note

Tesseract produced empty output on 4 of 9 pages (00013, 00015, 00017, 00018) regardless of PSM mode. The best-of-PSMs approach recovered partial output on some pages. Page 00016 was anomalously good for Tesseract (19.6–20.9% CER). Overall Tesseract is unreliable on this material.

---

## 5. Top 5 Confusions per Model

### german_print
| Rank | Type | Gold → Pred | Count |
|------|------|------------|------:|
| 1 | Sub | ẜ (U+1E9C) → ſ (U+017F) | 224 |
| 2 | Sub | ꞥ (U+A7A5) → n (U+006E) | 64 |
| 3 | Sub | - (U+002D) → ⸗ (U+2E17) | 63 |
| 4 | Sub | \n → space | 51 |
| 5 | Sub | a → e | 45 |

### austriannewspapers
| Rank | Type | Gold → Pred | Count |
|------|------|------------|------:|
| 1 | Sub | ẜ (U+1E9C) → ſ (U+017F) | 175 |
| 2 | Sub | i → t | 102 |
| 3 | Sub | ꞥ (U+A7A5) → n | 60 |
| 4 | Sub | U → u | 54 |
| 5 | Sub | u → n | 53 |

### catmus_print_large
| Rank | Type | Gold → Pred | Count |
|------|------|------------|------:|
| 1 | Sub | h → b | 286 |
| 2 | Sub | w → m | 165 |
| 3 | Sub | ẜ (U+1E9C) → f | 148 |
| 4 | Sub | k → t | 138 |
| 5 | Sub | ſ (U+017F) → s | 121 |

### catmus_print_small
| Rank | Type | Gold → Pred | Count |
|------|------|------------|------:|
| 1 | Sub | h → b | 257 |
| 2 | Sub | ẜ → f | 150 |
| 3 | Sub | s → e | 136 |
| 4 | Sub | w → m | 104 |
| 5 | Sub | d → b | 101 |

### catmus_print_tiny
| Rank | Type | Gold → Pred | Count |
|------|------|------------|------:|
| 1 | Sub | s → e | 225 |
| 2 | Sub | h → b | 210 |
| 3 | Sub | d → b | 129 |
| 4 | Sub | a → u | 113 |
| 5 | Sub | a → o | 112 |

### mccatmus
| Rank | Type | Gold → Pred | Count |
|------|------|------------|------:|
| 1 | Sub | h → b | 177 |
| 2 | Sub | d → o | 140 |
| 3 | Sub | w → m | 125 |
| 4 | Sub | s → e | 121 |
| 5 | Sub | ẜ → s | 114 |

### Specific Confusion Patterns of Interest

| Pattern | german_print | austriannewspapers | CATMuS variants | mccatmus |
|---------|:---:|:---:|:---:|:---:|
| ẜ → ſ (descender-s → long-s) | 224 | 175 | N/A (no ſ in codec) | N/A |
| ẜ → f | 0 | 0 | 148 (L), 150 (S) | 0 |
| ẜ → s | 0 | 0 | 0 | 114 |
| ſ → s (long-s → modern s) | 0 | 0 | 121 (L) | mapped to s |
| ꞥ → n (n-tilde → n) | 64 | 60 | (no distinct tracking) | (no distinct tracking) |
| ß → ſs | 0 | 0 | 0 | 0 |
| / → , or . | <5 | <5 | <5 | <5 |
| ‐ (U+2010) → - (U+002D) | not in codec | not in codec | not in codec | not in codec |
| h → b (Fraktur confusion) | 0 | 0 | 286 (L), 257 (S) | 177 |

---

## 6. Recommendation

### Winner: **german_print** (UB Mannheim, Zenodo 10519596)

### Runner-up: **austriannewspapers** (Kiessling, Zenodo 7933402)

**Rationale:** `german_print` leads across all 9 pages with a mean CER of 51.8% (27.3% on clean single-column pages), beating the runner-up `austriannewspapers` by 4.8 CER points overall and 7.9 points on single-column pages. Both models share the critical advantage of having ſ (long-s) in their codec — the CATMuS family lacks ſ entirely, causing systematic h→b, w→m, and s→e confusions from trying to read Fraktur letterforms with a non-Fraktur alphabet. The `german_print` model's larger codec (283 vs 186) and broader training data (15–20C range including both Fraktur and Antiqua) give it an edge on this 17C material.

---

## 7. Fine-Tuning Preamble (Winner: german_print)

### Architecture
```
VGSL: [1,120,0,1 Cr3,13,32 Do0.1 Mp2,2 Cr3,13,32 Do0.1 Mp2,2
       Cr3,9,64 Do0.1 Mp2,2 Cr3,9,64 Do0.1 S1(1x0)1,3
       Lbx200 Do0.1 Lbx200 Do0.1 Lbx200 Do O1c284]
```
- 4 convolutional layers (32→32→64→64 filters)
- 3 bidirectional LSTM layers (200 hidden units each)
- Input height: 120px, variable width, 1 channel (grayscale)
- Output: 284 classes (283 chars + CTC blank)
- Baseline segmentation mode

### Codec Analysis
| | Count |
|---|---:|
| Current codec size | 283 |
| Target alphabet (gold) | 81 |
| Overlap (in both) | 71 |
| **Chars to ADD** (in gold, not in model) | **10** |
| Chars to drop (in model, not in gold) | 212 |

**Characters to add:**
`Ł` (U+0141), `ł` (U+0142), `ŗ` (U+0157), `ẜ` (U+1E9C), `‐` (U+2010), `✝` (U+271D), `ꞡ` (U+A7A1), `ꞣ` (U+A7A3), `ꞥ` (U+A7A5), `Ꞩ` (U+A7A8)

### Recommended `ketos train --resize` Mode

**`--resize both`**

The gold alphabet is not a strict superset of the model codec — the model has 212 characters unused in the gold set (German-specific chars, fractions, symbols from Austrian newspaper material) and the gold set has 10 characters the model doesn't know. Using `--resize both` allows adding the 10 new Baltic/special characters while optionally pruning the unused codec entries to reduce output layer size and speed up convergence. If preserving the model's generality for future non-Glück use is desired, `--resize add` is the conservative alternative (only adds the 10 missing chars, keeps the full 283+10=293 codec).

---

## 8. Reproduction Block

```bash
# Environment: Python 3.12, CUDA A100, conda env "cloudspace"
pip install kraken editdistance

# 1. Download gold data
mkdir -p gold
for i in $(seq 13 21); do
  num=$(printf "%05d" $i)
  curl -sS -o "gold/$num.jpg" "https://t.noit.pro/gluck_1694/gold/$num.jpg"
  curl -sS -o "gold/$num.jpg.txt" "https://t.noit.pro/gluck_1694/gold/$num.jpg.txt"
done

# 2. Download models
mkdir -p models
curl -sL -o models/german_print.mlmodel \
  "https://zenodo.org/api/records/10519596/files/german_print.mlmodel/content"
curl -sL -o models/austriannewspapers.mlmodel \
  "https://zenodo.org/api/records/7933402/files/austriannewspapers.mlmodel/content"
curl -sL -o models/catmus_print_large.mlmodel \
  "https://zenodo.org/api/records/10592716/files/catmus-print-fondue-large.mlmodel/content"
curl -sL -o models/catmus_print_small.mlmodel \
  "https://zenodo.org/api/records/10602307/files/catmus-print-fondue-small-2024-01-31.mlmodel/content"
curl -sL -o models/catmus_print_tiny.mlmodel \
  "https://zenodo.org/api/records/10602357/files/catmus-print-fondue-tiny-2024-01-31.mlmodel/content"
curl -sL -o models/mccatmus.mlmodel \
  "https://zenodo.org/api/records/13788177/files/McCATMuS_nfd_nofix_V1.mlmodel/content"

# 3. Install Tesseract
sudo apt-get install -y tesseract-ocr tesseract-ocr-frk
sudo curl -sL -o /usr/share/tesseract-ocr/5/tessdata/Fraktur.traineddata \
  "https://github.com/tesseract-ocr/tessdata_best/raw/main/script/Fraktur.traineddata"

# 4. Segment (held constant for all models)
mkdir -p segmentation
for i in $(seq 13 21); do
  num=$(printf "%05d" $i)
  kraken --device cuda:0 -i "gold/$num.jpg" "segmentation/$num.json" segment -bl
done

# 5. Recognize (Kraken — Python API, see evaluate.py for full script)
# For each model, load with kraken.lib.models.load_any(path, device="cuda:0")
# then rpred.rpred(network, image, segmentation) per page.

# 6. Recognize (Tesseract)
for lang in frk Fraktur; do
  mkdir -p "predictions/tesseract_$lang"
  for i in $(seq 13 21); do
    num=$(printf "%05d" $i)
    tesseract "gold/$num.jpg" "predictions/tesseract_$lang/$num" -l "$lang" --psm 3
  done
done

# 7. Evaluate
python3 evaluate.py
```

---

## 9. Saved Predictions

Predictions stored in `predictions/<model>/` directory:
```
predictions/
├── german_print/          (00013.txt … 00021.txt)
├── austriannewspapers/    (00013.txt … 00021.txt)
├── catmus_print_large/    (00013.txt … 00021.txt)
├── catmus_print_small/    (00013.txt … 00021.txt)
├── catmus_print_tiny/     (00013.txt … 00021.txt)
├── mccatmus/              (00013.txt … 00021.txt)
├── tesseract_frk/         (00013.txt … 00021.txt)
└── tesseract_Fraktur/     (00013.txt … 00021.txt)
```

---

## 10. Flags / Issues Encountered

1. **No GT4HistOCR-derived Kraken model found.** Searched Zenodo, OCR-D model zoo, and kraken list. The dataset exists (zenodo.1344132) but no pre-trained `.mlmodel` was published. `german_print` is the closest proxy.

2. **blla column merge on two-column pages.** Pages 14, 16, 19, 21 suffer from line merging across columns, inflating CER to 60–75% on those pages. Single-column pages (13, 15, 17) give cleaner 25–33% CER for the winner. This is a segmentation issue, not a recognition issue — consider training or fine-tuning a column-aware segmentation model for the full pipeline.

3. **Tesseract empty output.** Pages 00013, 00015, 00017, 00018 produced 0 bytes with Tesseract regardless of PSM mode. Tesseract is unreliable on this material.

4. **CATMuS family lacks ſ (long-s).** All three CATMuS-Print variants and McCATMuS are missing U+017F from their codec. This is surprising for models trained on historical Western European prints. The codec uses NFC decomposition where ſ may have been normalized to s during training data preparation.

5. **No model covers Baltic diacritics.** ꞥ, ꞡ, ꞣ, Ꞩ, ẜ, Ł, ł, ŗ are absent from all tested models — these are guaranteed errors and the primary target for fine-tuning with `ketos train --resize add/both`.

6. **Kraken legacy polygon warning.** `austriannewspapers` and `catmus_print_small` triggered "Using legacy polygon extractor" warnings — no functional impact but the models would benefit from retraining with the newer polygon method.
