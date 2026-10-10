# Pre-registration: do image evaluators read Indic scripts?

Status: **draft**. Freeze this file (commit hash and date) after native-reader validation and **before** any real evaluator (beyond the pilot OCR and controls) is scored on the final set. Anything decided after that point is reported as exploratory.

Method details are in `docs/GLYPH_PAIRS.md`. Positioning against prior work is in `docs/LITERATURE_SURVEY.md`.

## 1. Questions

- **Q1.** Do text-to-image evaluators detect errors in Indic-script sign text as reliably as matched errors in Latin-script text?
- **Q2.** Which errors do they miss? The candidates are: vowel signs, lookalike letters, missing syllables, pseudo-words, unshaped rendering, neighbouring scripts, and transliteration.
- **Q3.** How much of an evaluator's apparent success comes from preferring correct-looking images regardless of the prompt (the *prior*), rather than from reading?
- **Q4.** Do cheap fixes close the gap?
  - An OCR-grounded judge.
  - A probe trained on synthetic pairs, including its transfer to a script it never saw.

## 2. Stimuli

- **Words.** `data/glyph_pairs/lexicon_v1.json`: 18 sign concepts in Hindi (Devanagari), Kannada, Tamil and English (Latin control).
- **Error types.**
  - Shared with the Latin control: `drop_vowel_sign`, `swap_vowel_sign`, `lookalike`, `drop_akshara`, `pseudo_word`.
  - Indic-only: `unshaped`, `script_swap`, `transliterated`.
  - Control: `null_font`.
- **Rendering.** 3 renderings (`--variants 3`) per (word, script, error type). Each pair keeps the same layout, palette, mild perspective, blur and noise on both sides; only the text differs.
- **Mirroring.** Every mirrorable pair is also scored with the prompt asking for the corrupted string.
- **Validity filters, applied before scoring:**
  - automatically: missing glyphs; pixel-identical pairs;
  - by native readers (`glyph_validation.py`, rules in its docstring): a clean image that readers judge wrong, an illegible image, or an error type that readers detect less than 80% of the time for that script.

## 3. Evaluators

| Role | Evaluator |
|---|---|
| Primary | CLIPScore (`openai/clip-vit-large-patch14`) |
| Primary | Multilingual CLIP (OpenCLIP `xlm-roberta-large-ViT-H-14`) |
| Primary | VQAScore (`clip-flant5-xl`, or `-xxl` if hardware allows) |
| Primary | Open VLM judge (Qwen2.5-VL, the largest that fits), with the standard judge prompt |
| Primary | Tesseract OCR, cropped to the sign region (1 − CER); an upper bound for OCR checking |
| Secondary | Hosted VLM judge(s); TextPecker; Tesseract on the full image |
| Fix | VLM judge given the OCR reading (`ocr_grounded`); learned probe on CLIP features, cross-fitted by concept, plus a held-out-script variant |
| Control | `random`; `image_stats` (prompt-blind) |

## 4. Outcomes

- **Reading accuracy** (primary): P(the image matching the prompt is scored higher), pooled over forward and mirrored pairs, ties counted as 0.5. Any prompt-independent preference scores exactly 0.5.
- **Gap**: reading accuracy (Latin) − reading accuracy (Indic). Computed on shared error types, per script and pooled over the three Indic scripts.
- **Prior**: forward − mirrored accuracy.
- **Forward-only results**: `unshaped` and `null_font` are interpreted only relative to the `image_stats` and `random` controls on the same pairs.

**Uncertainty.** 2,000 bootstrap resamples of whole concepts, so the Latin and Indic pairs of a concept move together, with 95% percentile intervals.

## 5. Hypotheses and decision rules

| ID | Hypothesis | Test | Status |
|---|---|---|---|
| H1 | Evaluators read Indic text less reliably than Latin text | Pooled gap > 0, per primary evaluator; Holm across primary evaluators, α = 0.05; the 95% CI must exclude 0 | **Confirmatory** |
| H2 | Some evaluators do not read Indic text at all | Reading accuracy per Indic script; 95% CI including 0.5 = "no evidence of reading" | Secondary |
| H3 | VLM judges show a larger prior on Indic than on Latin text (autocorrection) | Prior (Indic) − prior (Latin), bootstrap CI | Secondary |
| H4 | Errors in short words are missed more often | Reading accuracy by length bin | Exploratory |
| H5 | Unshaped Indic text goes unnoticed | Forward accuracy on `unshaped` vs `image_stats` on the same pairs | Secondary |
| H6 | Fixes reduce the gap | Gap (fix) vs gap (base judge / base CLIP), paired bootstrap over concepts; held-out-script probe reported separately | Secondary |

Every result is informative. A gap shows that current automatic evaluation is unreliable for Indian text. No gap, with tight intervals, is the first validation of these evaluators for Indic signage.

## 6. Power

`scripts/glyph_power.py` simulates concept-level random effects (logit SD 0.5) with 18 concepts, 3 Indic scripts and Latin accuracy 0.85:

| Variants | Indic accuracy | Power (pooled gap, 95% CI excludes 0) |
|---|---|---|
| 1 | 0.80 | 0.33 |
| 1 | 0.75 | 0.84 |
| 3 | 0.80 | 0.76 |
| 3 | 0.75 | 1.00 |

These figures are optimistic: renderings of the same word are treated as independent once the concept effect is accounted for. **Decision:** use 3 variants. Add concepts (target 30) if the pilot's concept-level variance is above 0.5 or the expected gap is below 10 points.

## 7. Threats to validity

- **Synthetic images.** Clean fonts on boards are easier than generated images. Mitigations: perspective, blur and noise; licensed real-photo backgrounds as a robustness check; and the real-generator track in `docs/RESEARCH_PROTOCOL.md` for ecological validity.
- **Lookalike tables and spellings** were chosen by the authors. Native-reader validation filters them, and the filtered error types are reported.
- **OCR evaluator with an oracle crop.** The crop is an upper bound, so it is also reported without the crop.
- **The OCR-grounded judge** chooses, among Tesseract's readings, the one closest to the requested text. This is symmetric across clean and corrupt images, but it is optimistic.
- **Prompt wording.** Prompts name no language, so forward and mirrored prompts are built identically. A sensitivity check with the language named is exploratory.
- **Multiple comparisons.** Only H1 is confirmatory. All per-type, per-script and per-length cells are descriptive.
