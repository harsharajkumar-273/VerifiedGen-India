# Glyph pairs: pilot results (OCR and controls only)

**Status: pilot, not confirmatory.**
- One rendering per word (575 forward and 450 mirrored pairs, 1,150 images).
- Synthetic backgrounds.
- The lexicon and lookalike tables are not yet checked by native readers.
- Only the evaluators that run on CPU without a model download were scored. The CLIP-family evaluators, VQAScore and VLM judges are in `notebooks/glyph_pairs_gpu.ipynb`.

Reproduce with the commands in `docs/GLYPH_PAIRS.md`, at seed 0, `--variants 1`, Tesseract 5.3.4 with `tessdata_best`.

## Reading accuracy (shared error types; forward and mirrored pooled; 0.5 = no reading)

| Evaluator | Latin | Devanagari | Kannada | Tamil | Gap, Latin − pooled Indic [95% CI] |
|---|---|---|---|---|---|
| Tesseract, sign crop | 0.97 [0.95, 0.99] | 0.92 [0.88, 0.96] | 0.90 [0.87, 0.93] | 0.88 [0.82, 0.93] | +0.067 [+0.035, +0.098] |
| Tesseract, full image | 0.85 [0.81, 0.89] | 0.62 [0.56, 0.70] | 0.54 [0.51, 0.58] | 0.57 [0.53, 0.62] | +0.273 [+0.219, +0.328] |
| Image statistics (prompt-blind) | 0.50 | 0.50 | 0.50 | 0.50 | 0.000 |
| Random | 0.47 [0.42, 0.52] | 0.54 [0.49, 0.60] | 0.50 [0.44, 0.56] | 0.49 [0.41, 0.56] | −0.038 [−0.113, +0.031] |

Both OCR gaps survive Holm correction across the four evaluators (adjusted p < 0.001). The bootstrap resamples whole concepts.

## By error type (Tesseract reading accuracy)

| Error type | Crop: Latin | Crop: Deva | Crop: Kannada | Crop: Tamil | Full: Latin | Full: Deva | Full: Kannada | Full: Tamil |
|---|---|---|---|---|---|---|---|---|
| drop_vowel_sign | 0.99 | 0.95 | 0.85 | 0.92 | 0.86 | 0.66 | 0.50 | 0.53 |
| swap_vowel_sign | 1.00 | 0.89 | 0.86 | 0.70 | 0.96 | 0.58 | 0.54 | 0.53 |
| lookalike | 0.88 | 0.80 | 0.90 | 0.93 | 0.81 | 0.56 | 0.54 | 0.57 |
| drop_akshara | 0.99 | 0.96 | 0.90 | 0.88 | 0.76 | 0.60 | 0.50 | 0.53 |
| pseudo_word | 1.00 | 1.00 | 1.00 | 0.97 | 0.88 | 0.72 | 0.64 | 0.68 |
| script_swap | – | 0.94 | 0.89 | 0.99 | – | 0.61 | 0.51 | 0.71 |
| transliterated | – | 1.00 | 1.00 | 0.99 | – | 0.72 | 0.78 | 0.76 |
| unshaped (forward only) | – | 0.65 | 0.86 | 0.64 | – | 0.62 | 0.58 | 0.53 |

Reference for unshaped: the prompt-blind `image_stats` baseline scores 0.38 (Devanagari), 0.00 (Kannada) and 0.50 (Tamil) on the same pairs.

## What the pilot shows

1. **The method works as designed.** Prompt-blind and random evaluators sit at 0.5 reading accuracy on every script. Before mirroring was added, the prompt-blind baseline "detected" 100% of deletion errors (shorter text means less ink), which would have inflated every evaluator.
2. **The OCR gap depends mostly on finding the text.**
   - When OCR is given the sign region (an oracle crop), Indic reading is close to Latin: a gap of about 7 points.
   - On the full image the gap is 27 points. Kannada and Tamil fall to near chance while Latin stays at 0.85.
   - So for OCR-based checking, the Indic-specific failure is mainly *finding and segmenting* the text, not reading it.
3. **Unshaped rendering is the hardest error.** Even with the crop, OCR misses about a third of unshaped Devanagari and Tamil signs: broken conjuncts and misplaced vowel signs are read as the correct word. This is a failure that text-to-image models and rendering software actually produce.
4. **Tamil vowel-sign swaps are missed more often (0.70 with the crop)**, perhaps because the vowel signs ெ, ே and ை differ only slightly.
5. **Lesson for the paper: OCR settings matter.** Default Tesseract settings returned *nothing* on about 75% of clean Kannada and Tamil signs, which made reading accuracy look like chance. Taking the best of six readings (raw or Otsu-binarised, page-segmentation modes 7, 6 and 13) fixed that. Any OCR baseline in this area should report its settings and its read rate on clean images.

## Caveats

- One rendering per word, and 18 words. The intervals are wide per cell, and per-type cells are descriptive only.
- Synthetic boards are easier than real or generated scenes.
- The OCR evaluator chooses the reading closest to the requested text. That helps both images of a pair equally, but it is optimistic.
- Error types the native readers reject will be dropped before the confirmatory run (`docs/GLYPH_PROTOCOL.md`).
