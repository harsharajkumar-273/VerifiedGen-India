# VerifiedGen-India

Do automatic image evaluators work for India? A controlled study of evaluator reliability on India-centric text-to-image generation, plus a multi-axis verifier.

## Research question

**Are automatic image evaluators less reliable on India-centric content than on generic content, and does that let bad images through?**

Text-to-image papers report quality with CLIPScore, question-answering scores and vision-language judges. These evaluators are mostly validated on Western, English prompts. This project tests them on matched pairs of prompts: each India-centric prompt has a control with the same structure and difficulty, where only the cultural or script content changes. Human panels, with native readers for each script, provide ground truth.

| Hypothesis | Test |
|---|---|
| H1 Reliability drops on India-centric prompts | AUROC(control) − AUROC(India), bootstrap over matched pairs, Holm-corrected |
| H2 More bad India-centric images slip through | False-accept rate at a threshold calibrated on control images |
| H3 Script, not locale, drives the text-prompt drop | Three-way design: Indic text in India / English text in India / English text in a Western city |
| H4 Best-of-N selection helps less on India-centric prompts | Share of the achievable human-score gain that the evaluator's pick captures |

The full pre-registration draft is in [`docs/RESEARCH_PROTOCOL.md`](docs/RESEARCH_PROTOCOL.md) and the annotator instructions are in [`docs/ANNOTATION_GUIDE.md`](docs/ANNOTATION_GUIDE.md). The VerifiedGen multi-axis verifier is one of the evaluators under test, alongside CLIPScore, VQAScore, vision-language judges, and random and image-statistics controls.

## Study workflow

```bash
# 1. Matched prompt bank (185 prompts: 45 script triplets, 20 cultural pairs, 10 code-mixed)
python3 scripts/build_matched_prompts.py

# 2. Generate 4 images per prompt per generator into <image-dir>/<prompt_id>/, then build a manifest
python3 scripts/import_image_candidates.py --prompt-bank data/image_verification/matched_prompt_bank_v1.jsonl \
  --image-dir data/image_verification/generated/sdxl --generator-name sdxl \
  --output-manifest data/image_verification/study_manifest.jsonl

# 3. Score with built-in evaluators (external ones write the same {"candidate_id","evaluator","score"} JSONL)
python3 scripts/score_images.py --manifest data/image_verification/study_manifest.jsonl \
  --evaluators random image_stats verifiedgen clipscore --output-dir outputs/study/scores

# 4. Export blinded annotation tasks (images copied under opaque ids; keep the key file private)
python3 scripts/annotate.py export --manifest data/image_verification/study_manifest.jsonl \
  --output-jsonl data/image_verification/annotation/tasks.jsonl --key-json data/image_verification/annotation/study.key.json \
  --image-dir data/image_verification/annotation/images --salt "<secret>"

# 5. Run the pre-specified analysis on the returned labels
python3 scripts/run_study.py --manifest data/image_verification/study_manifest.jsonl \
  --labels-jsonl data/image_verification/annotation/labels.jsonl --key-json data/image_verification/annotation/study.key.json \
  --scores outputs/study/scores/*.jsonl --output-dir outputs/study
```

`outputs/study/study_report.md` contains the headline contrasts with confidence intervals, a per-condition table with the human ceiling, the per-script breakdown and best-of-N selection results. `study_results.json` holds everything in machine-readable form.

The first grounded external-validation track uses AIKosh agriculture resources to define and validate India-specific agriculture prompts. AIKosh data is not automatically downloaded; access and licensing must be checked before using or redistributing any source data.

## Glyph minimal pairs: do evaluators read Indic scripts?

The main experiment now in progress. Correct sign text is rendered with fonts next to an otherwise identical image containing one controlled error, such as a dropped matra, a lookalike letter, unshaped conjuncts, or a neighbouring script. Ground truth therefore comes from how each pair is built. Every pair is also scored with a *mirrored* prompt, so evaluators that ignore the prompt score exactly 0.5. The design is described in [`docs/GLYPH_PAIRS.md`](docs/GLYPH_PAIRS.md) and positioned against prior work in [`docs/LITERATURE_SURVEY.md`](docs/LITERATURE_SURVEY.md).

```bash
python3 scripts/build_glyph_pairs.py fetch-fonts
python3 scripts/build_glyph_pairs.py build --out-dir outputs/glyph_pairs/pilot
python3 scripts/score_images.py --manifest outputs/glyph_pairs/pilot/candidates.jsonl --evaluators random image_stats --output-dir outputs/glyph_pairs/pilot/scores
python3 scripts/analyze_glyph_pairs.py --pairs outputs/glyph_pairs/pilot/pairs.jsonl --scores outputs/glyph_pairs/pilot/scores/*.jsonl --output-dir outputs/glyph_pairs/pilot/results
```

## The VerifiedGen verifier

```text
India-centric prompts -> generated candidates -> manifest -> verifier -> annotation -> experiments
```

The verifier checks prompt alignment, object evidence, colors, visual quality, artifact risk, safety labels, multilingual text, and cultural-context markers. It returns `pass`, `review`, or `fail`. Object detections, OCR text, safety labels and observed cultural markers are read from the manifest. In the study they must come from models, never from the human labels, or the evaluation becomes circular.

Safety is a hard constraint:

```text
unsafe -> reject_safety
safe but weak -> regenerate
safe and verified -> accept
```

The reward interface in `src/image_verification/reward.py` is ready for later preference learning or constrained RL. It should be calibrated with human labels before training a generation policy.

## Legacy workflows

The commands below predate the matched study design. They still work for the unpaired prompt banks.

### Legacy prompt bank

```bash
python3 scripts/build_genfailbench_prompts.py --preset sarvam --limit 85 --output data/image_verification/sarvam_prompt_bank_v1.jsonl
```

The Sarvam track contains nine Indic scripts, code-mixed signage, regional agriculture and governance, and cultural-context prompts. Its specification is in `data/image_verification/sarvam_benchmark_spec.json`.

Agriculture prompts include provenance metadata linking them to the AIKosh ICAR rice/maize reference source. The registry is stored in `data/image_verification/aikosh_sources.json`.

### Import generated images

Put images under `data/image_verification/generated/real_sdxl_cpu/<prompt_id>/`, then run:

```bash
python3 scripts/import_image_candidates.py \
  --prompt-bank data/image_verification/india_prompt_bank_v1.jsonl \
  --image-dir data/image_verification/generated/real_sdxl_cpu \
  --generator-name sdxl_cpu \
  --output-manifest data/image_verification/india_real_sdxl_cpu_manifest.jsonl
```

### Validate and annotate

```bash
python3 scripts/validate_image_manifest.py \
  --manifest data/image_verification/india_real_sdxl_cpu_manifest.jsonl \
  --output-json outputs/image_verification/india_manifest_validation.json

# Not blinded: use scripts/annotate.py export for study labels
python3 scripts/export_image_annotations.py export \
  --manifest data/image_verification/india_real_sdxl_cpu_manifest.jsonl \
  --output-jsonl data/image_verification/india_annotation_tasks.jsonl
```

### Run experiments

```bash
python3 scripts/run_image_experiments.py \
  --manifest data/image_verification/india_real_sdxl_cpu_manifest.jsonl \
  --output-dir outputs/image_verification/india_experiments \
  --config-file configs/verifiedgen_india.experiment.json
```

Evaluate OCR by language and script after adding `ocr_text` to the manifest:

```bash
python3 scripts/evaluate_ocr.py \
  --manifest data/image_verification/sarvam_manifest.jsonl \
  --output-json outputs/image_verification/sarvam_ocr_metrics.json
```

Score candidates with the safety gate and reward interface:

```bash
python3 scripts/score_candidates.py \
  --manifest data/image_verification/india_real_sdxl_cpu_manifest.jsonl \
  --output-jsonl outputs/image_verification/india_candidate_scores.jsonl
```

Build human-preference pairs after adding `human_rank` values to the scored records:

```bash
python3 scripts/train_preference_model.py pairs \
  --scored-jsonl outputs/image_verification/india_candidate_scores.jsonl \
  --output-jsonl data/image_verification/preference_pairs.jsonl
```

Train the pairwise reward model:

```bash
python3 scripts/train_preference_model.py train \
  --pairs-jsonl data/image_verification/preference_pairs.jsonl \
  --output-dir outputs/image_verification/reward_model \
  --epochs 20
```

## Run tests

```bash
pip install numpy Pillow fonttools
python3 scripts/build_glyph_pairs.py fetch-fonts   # optional: enables the rendering tests
python3 -m unittest discover -s tests
```

## Code map

Study:

- `src/image_verification/matched_pairs.py`: matched India/control prompt bank
- `src/image_verification/evaluators.py`: evaluator registry (random, image stats, verifier, OCR, CLIPScore, precomputed scores)
- `src/image_verification/human_labels.py`: blinded task export, label aggregation, inter-annotator agreement and human ceiling
- `src/image_verification/study.py`: pre-specified analysis and Markdown report
- `src/image_verification/selection.py`: best-of-N selection analysis
- `src/image_verification/glyph_pairs.py`: glyph minimal-pair generator (error types, shaping-aware rendering, prompt mirroring)
- `src/image_verification/glyph_eval.py`: forward / mirrored / reading accuracy and the Latin-vs-Indic gap
- `src/image_verification/stats.py`: AUROC, Spearman, calibrated false-accept rate, Krippendorff's alpha, cluster bootstrap, Holm correction

Verifier and tooling:

- `src/image_verification/verifier.py`: central scoring and decision logic
- `src/image_verification/schema.py`: candidate/report data structures
- `src/image_verification/prompt_bank.py`: curated India prompt bank
- `src/image_verification/manifest_tools.py`: image-folder to JSONL conversion
- `src/image_verification/annotation.py`: human-label task export
- `src/image_verification/experiments.py`: experiment runner
- `src/image_verification/metrics.py`: evaluation metrics
- `src/image_verification/safety.py`: hard safety gate
- `src/image_verification/reward.py`: reward and constrained decisions
- `src/image_verification/preference.py`: preference-pair builder and reward-model training
- `src/image_verification/aikosh.py`: AIKosh provenance and grounded-domain metadata
- `src/utils/checkpoint.py`: resumable-run support

Generated images, annotations, and model caches are intentionally not included in the rebuilt source tree. They must be regenerated locally or on a GPU machine.
