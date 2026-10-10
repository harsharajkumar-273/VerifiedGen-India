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

## Status

Done:
- [x] Generator, validity checks, prompt mirroring and analysis, with concept-clustered bootstrap and Holm correction across evaluators.
- [x] Tests: segmentation, the error types, mirroring, rendering, and the analysis checked on synthetic "reader" and "ink" evaluators.
- [x] Pilot run with `random` and `image_stats`. Both sit at 0.5 reading accuracy, as they should.

Still to do:
- [ ] Native-reader check: the lexicon, the lookalike tables, and about 50 rendered pairs per script, confirming each error type is visible and really an error.
- [ ] Real evaluators on a GPU: CLIPScore (`openai/clip-vit-large-patch14` and a multilingual CLIP), VQAScore, 2–3 VLM judges, TextPecker-8B, and an Indic OCR.
- [ ] Licensed real-photo backgrounds (`--background-dir`) as a robustness check against the plain synthetic backgrounds.
- [ ] More scripts: Bengali, Telugu, Malayalam, Gujarati, Gurmukhi, Odia. The ISCII-aligned code handles them once fonts, lexicon entries and lookalike tables are added.
- [ ] Pre-register the analysis, with reading accuracy and the pooled gap as primary, before scoring the real evaluators.
