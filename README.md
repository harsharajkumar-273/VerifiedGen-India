# VerifiedGen-India

Verification-first evaluation for India-centric text-to-image generation.

## Research question

Can a multi-axis verifier detect weak India-centric generated images before users see them?

The project builds a small benchmark, generates candidate images, scores them, and compares the verifier against human judgments and simpler baselines.

The first grounded external-validation track uses AIKosh agriculture resources to define and validate India-specific agriculture prompts. AIKosh data is not automatically downloaded; access and licensing must be checked before using or redistributing any source data.

## Pipeline

```text
India-centric prompts -> generated candidates -> manifest -> verifier -> annotation -> experiments
```

The verifier checks prompt alignment, object evidence, colors, visual quality, artifact risk, safety labels, multilingual text, and cultural-context markers. It returns `pass`, `review`, or `fail`.

Safety is a hard constraint:

```text
unsafe -> reject_safety
safe but weak -> regenerate
safe and verified -> accept
```

The reward interface in `src/image_verification/reward.py` is ready for later preference learning or constrained RL. It should be calibrated with human labels before training a generation policy.

## Rebuild the prompt bank

```bash
python3 scripts/build_genfailbench_prompts.py --preset sarvam --limit 85 --output data/image_verification/sarvam_prompt_bank_v1.jsonl
```

The Sarvam track contains nine Indic scripts, code-mixed signage, regional agriculture and governance, and cultural-context prompts. Its specification is in `data/image_verification/sarvam_benchmark_spec.json`.

Agriculture prompts include provenance metadata linking them to the AIKosh ICAR rice/maize reference source. The registry is stored in `data/image_verification/aikosh_sources.json`.

## Run tests

```bash
python3 -m unittest discover -s tests
```

## Import generated images

Put images under `data/image_verification/generated/real_sdxl_cpu/<prompt_id>/`, then run:

```bash
python3 scripts/import_image_candidates.py \
  --prompt-bank data/image_verification/india_prompt_bank_v1.jsonl \
  --image-dir data/image_verification/generated/real_sdxl_cpu \
  --generator-name sdxl_cpu \
  --output-manifest data/image_verification/india_real_sdxl_cpu_manifest.jsonl
```

## Validate and annotate

```bash
python3 scripts/validate_image_manifest.py \
  --manifest data/image_verification/india_real_sdxl_cpu_manifest.jsonl \
  --output-json outputs/image_verification/india_manifest_validation.json

python3 scripts/export_image_annotations.py export \
  --manifest data/image_verification/india_real_sdxl_cpu_manifest.jsonl \
  --output-jsonl data/image_verification/india_annotation_tasks.jsonl
```

## Run experiments

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

## Code map

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
