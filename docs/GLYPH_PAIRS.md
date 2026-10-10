# Glyph minimal pairs

The question: **do automatic image evaluators actually read Indic-script text, or do they only check that something text-shaped is there?** For background and positioning, see `docs/LITERATURE_SURVEY.md`.

## Design

Each pair is two images that are identical in every respect except the sign text:
- the same background, board, palette and font size;
- the same blur and noise seed.

The *clean* image shows the requested word. The *corrupt* image shows one controlled error. Ground truth comes from how the pair is built.

An evaluator passes a pair when it scores the image that matches the prompt higher.

### Prompt mirroring (why there are two directions)

Many errors change the image in ways an evaluator can exploit without reading:
- dropping a letter leaves less ink;
- a pseudo-word looks less familiar;
- unshaped text looks messier.

The pilot showed this directly. A prompt-blind pixel-statistics baseline (`image_stats`) "detected" 100% of deletion errors.

So each pair is also scored **mirrored**: the same two images, with a prompt asking for the corrupted string. The image that matches the prompt is now the other one.

- **Reading accuracy** = forward and mirrored pairs pooled. **This is the primary metric.** Any evaluator that ignores the prompt scores exactly 0.5 on it; the pilot confirms this for `image_stats`.
- **Prior** = forward − mirrored accuracy. This is the prompt-independent preference for the correct-looking image. For VLM judges this is where autocorrection towards real words should appear (see FaithC4 in the survey).

`unshaped` and `null_font` cannot be mirrored, because the requested text is the same on both sides. They are reported forward only, and must always be read against the prompt-blind baselines. In the pilot, `image_stats` prefers the *unshaped* Kannada rendering in every pair.

### Error types

| Type | Indic | Latin control | Notes |
|---|---|---|---|
| `drop_vowel_sign` | remove one matra | remove one vowel letter (never the first letter) | shared |
| `swap_vowel_sign` | replace one matra with another | replace a vowel letter | shared; rare Devanagari ॆ/ॊ excluded |
| `lookalike` | swap a confusable letter (table in `glyph_pairs.SCRIPTS`) | E/F, C/G, O/Q, … | shared; **table not yet validated** |
| `drop_akshara` | remove one syllable cluster (never the first) | remove one letter | shared |
| `pseudo_word` | random well-formed syllables, same length | random letters, same length | shared |
| `unshaped` | correct text rendered without HarfBuzz shaping | – | forward only; dropped if it renders identically |
| `script_swap` | code-point transliteration into the neighbouring script (Deva→Gujarati, Kannada→Telugu, Tamil→Malayalam) | – | dropped if a glyph is missing |
| `transliterated` | the word in Latin letters | – | |
| `null_font` | *not an error*: same text, other font | same | expected accuracy 0.5; gives the noise floor |

Word length is measured in syllables, so the bins (1–2, 3–4, 5+) are roughly comparable across scripts:
- **Indic:** aksharas. Tamil does not form conjuncts, so `C+pulli` counts as its own syllable.
- **Latin:** vowel groups.

### Lexicon

`data/glyph_pairs/lexicon_v1.json` contains 18 shop-sign concepts aligned across Hindi, Kannada, Tamil and English, with informal romanisations. **Every Indic spelling and romanisation still needs a native reader's check.**

## Running it

```bash
pip install -r requirements.txt            # Pillow must include libraqm (the PyPI wheels do)
python scripts/build_glyph_pairs.py fetch-fonts            # Noto fonts (SIL OFL) into data/fonts/
python scripts/build_glyph_pairs.py build --out-dir outputs/glyph_pairs/pilot
#   optional: --variants 3 (more renderings per word), --background-dir <licensed photos>
python scripts/score_images.py --manifest outputs/glyph_pairs/pilot/candidates.jsonl \
    --evaluators random image_stats clipscore --output-dir outputs/glyph_pairs/pilot/scores
python scripts/analyze_glyph_pairs.py --pairs outputs/glyph_pairs/pilot/pairs.jsonl \
    --scores outputs/glyph_pairs/pilot/scores/*.jsonl --output-dir outputs/glyph_pairs/pilot/results
```

The build writes these files:

| File | Contents |
|---|---|
| `pairs.jsonl` | One row per pair, forward and mirrored. |
| `candidates.jsonl` | One row per (image, prompt), in the `ImageCandidate` format that every scorer reads. |
| `dropped.jsonl` | Pairs rejected by the validity checks: a missing glyph, or a pixel-identical pair. |
| `contact_sheet.png` | Sample pairs, for checking by eye. |

The pilot (one rendering per word) has 575 forward pairs, 450 mirrored pairs and 1,150 images. It builds in about 2 minutes on CPU.

External evaluators write score files keyed by `candidate_id`, the same format the rest of the repo uses:
- VQAScore via `t2v_metrics`;
- VLM judges;
- the TextPecker model;
- an Indic OCR plus CER.

## Evaluators, fixes and validation

| Tool | Command | Notes |
|---|---|---|
| Tesseract OCR | `score_images.py --evaluators ocr_tesseract` | Uses the language of the requested script; reads the sign crop by default (`--option ocr_tesseract.crop=false` for the full image); downloads `tessdata_best` on first use; caches readings |
| CLIPScore | `--evaluators clipscore --clip-model …` | Any `transformers` CLIP checkpoint |
| Multilingual CLIP | `--evaluators openclip` | OpenCLIP `xlm-roberta-large-ViT-H-14` by default |
| VQAScore | `--evaluators vqascore` | Needs `pip install t2v_metrics` |
| VLM judge | `--evaluators vlm_judge --option vlm_judge.model_name=Qwen/Qwen2.5-VL-7B-Instruct` | Score = P("yes"). `backend=openai` for any OpenAI-compatible endpoint with logprobs |
| OCR-grounded judge (fix) | `… --option vlm_judge.ocr_grounded=true` | The judge is told what the OCR read |
| Learned probe (fix) | `glyph_probe.py extract` then `score` | Low-rank bilinear head on frozen CLIP features, trained on pairs in both directions and cross-fitted by concept; `--holdout-script` tests transfer to an unseen script |
| Native-reader validation | `glyph_validation.py export` / `aggregate` | Blinded per-language sheets; produces `exclusions.json` for `analyze_glyph_pairs.py --exclusions` |
| Power | `scripts/glyph_power.py` | Simulated power of the pooled gap |

`notebooks/glyph_pairs_gpu.ipynb` runs everything end to end on Colab, Kaggle or a lab GPU. The pre-registration is `docs/GLYPH_PROTOCOL.md`.

## Status

Done:
- [x] Generator with perspective, sign-region boxes, validity checks and prompt mirroring.
- [x] Analysis: reading, forward and mirrored accuracy, prior, gap with concept-clustered bootstrap and Holm correction, exclusions.
- [x] Evaluator adapters (OCR, CLIP, OpenCLIP, VQAScore, VLM judge on local models or an API, OCR-grounded judge) and the learned probe.
- [x] Native-reader validation kit, power simulation, GPU notebook, pre-registration draft.
- [x] Pilot OCR results: `docs/GLYPH_PILOT_RESULTS.md`.
- [x] Tests, including the analysis checked on synthetic "reader" and "ink" evaluators and the probe on synthetic features.

Still to do (needs people or a GPU):
- [ ] Native readers fill in the sheets from `glyph_validation.py export`: Hindi, Kannada, Tamil, plus English.
- [ ] Freeze `docs/GLYPH_PROTOCOL.md`, then run the notebook with `VARIANTS = 3`.
- [ ] Optional: licensed real-photo backgrounds; more scripts (Bengali, Telugu, Malayalam, Gujarati); hosted judges; TextPecker.
