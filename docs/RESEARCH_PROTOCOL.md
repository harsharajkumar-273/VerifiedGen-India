# Research protocol: evaluator reliability on India-centric images

Status: **draft pre-registration**. Freeze this file (commit hash + date) before any human labels are collected. Anything decided after seeing labels must be reported as exploratory.

## 1. Question

Text-to-image papers report quality with automatic evaluators: CLIPScore, question-answering scores such as VQAScore or TIFA, and vision-language models used as judges. These evaluators are mostly validated on Western, English prompts.

**RQ1 (primary).** Are automatic image evaluators less reliable on India-centric prompts than on matched generic prompts, and does that let bad images through?

**RQ2.** For prompts that ask for rendered text, is the reliability drop driven by the *script* (Indic vs Latin) or by the *locale* (Indian vs Western setting)?

**RQ3 (exploratory).** Does reliability vary across Indic scripts, and does it track how heavily a script relies on combining marks?

**RQ4 (secondary).** When an evaluator is used to pick the best of N generated images, does its choice help human-judged quality as much on India-centric prompts as on controls?

## 2. Hypotheses

Reliability = AUROC of the evaluator's score against the human panel's accept/reject verdict.

| ID | Hypothesis | Test statistic | Direction |
|---|---|---|---|
| H1 | Evaluators are less reliable on India-centric prompts | AUROC(control) − AUROC(India-centric), pooled over the script and cultural tracks | > 0 |
| H2 | More human-rejected India-centric images are accepted | FAR(India) − FAR(control) with the threshold calibrated on controls to keep 80% of human-accepted control images | > 0 |
| H3a | Within the script track, the drop comes from script | AUROC(india_latin) − AUROC(india_indic) | > 0 |
| H3b | …not from locale | AUROC(global_latin) − AUROC(india_latin) | ≈ 0 |
| H4 | Best-of-N selection helps less on India-centric prompts | capture(control) − capture(India), see §6 | > 0 |

**Confirmatory test.** H1 only, one test per primary evaluator, Holm-corrected across evaluators at α = 0.05. Everything else is secondary or exploratory and is reported with intervals but no significance claims.

**All outcomes are informative.** A gap shows current evaluation practice is unreliable for Indian content. No gap, with tight intervals, is the first careful validation of these evaluators for Indian use.

## 3. Stimuli: matched prompt bank

`data/image_verification/matched_prompt_bank_v1.jsonl`, built by `scripts/build_matched_prompts.py`.

| Track | Conditions | Pairs | Prompts |
|---|---|---|---|
| Script | `india_indic` / `india_latin` / `global_latin` | 45 (9 languages × 5 concepts) | 135 |
| Cultural | `india` / `global` | 20 (agriculture, public services, festivals, everyday scenes) | 40 |
| Code-mixed (unpaired, descriptive) | `india_code_mixed` | – | 10 |

Matching rules:
- the same sentence template;
- the same number of required objects;
- the same category;
- for the script track, the same concept and the same English control text on both Latin-script sides.

The script track is a three-way design: the same Indian city with Indic text, then English text, then a Western city with English text. That lets script effects be separated from locale effects.

Planned scale: 185 prompts × 2 generators × 4 images = **1,480 images**.

## 4. Evaluators

| Role | Evaluator | Source |
|---|---|---|
| Primary | CLIPScore (`openai/clip-vit-large-patch14`) | `scripts/score_images.py --evaluators clipscore` |
| Primary | VQAScore | Official implementation; import as a score file |
| Primary | Vision-language judge (one open-weight, one hosted) | Prompt below; import as a score file |
| Primary | VerifiedGen multi-axis verifier | `scripts/score_images.py --evaluators verifiedgen` |
| Control | Random | Chance floor |
| Control | Image statistics only | Prompt-blind floor |

Score files are JSONL rows `{"candidate_id", "evaluator", "score"}`, where a higher score means more acceptable.

**Standard judge prompt.** Use this verbatim so the judge answers the same question as the human panel. The score is P("yes"), or the 1–10 rating divided by 10 if the model does not expose probabilities.

> This image was generated for the prompt: "{prompt}". Would you show this image to the person who typed this prompt? Check that any requested text is spelled correctly in the requested script, that the setting and cultural details match the prompt, and that requested objects are present. Answer yes or no.

If the verifier uses detector, OCR or cultural-marker inputs, those inputs must come from models, not from the human labels. Otherwise the evaluation is circular.

## 5. Human labels

- **Panel.** 3 annotators per image; an odd count avoids ties, and tied images are excluded from binary analyses. Every image that contains requested text is labelled by annotators who read that script natively. Record each annotator's region and languages.
- **Blinding.** Tasks are exported with `scripts/annotate.py export --image-dir …`, which:
  - replaces ids with salted hashes and copies images under those opaque names;
  - drops generator, condition and scores from the task;
  - shuffles task order.
  
  The key file stays with the study lead.
- **Same pool for both conditions.** India-centric and control images are labelled by the same annotators, in the same session, interleaved.
- **Pilot.** Run 50 images first to refine `docs/ANNOTATION_GUIDE.md`, then discard those labels.
- **Ceiling.** Report Krippendorff's α per condition, plus a held-out-annotator AUROC: each annotator's verdict scored against the majority of the others. A drop in evaluator AUROC only counts as an evaluator failure to the extent the human ceiling does not drop too.

## 6. Analysis plan

Run with `scripts/run_study.py`; settings are in `configs/study.json`.

- **AUROC** of the score against the majority verdict, per condition.
- **Spearman** correlation of the score with mean human alignment (1–5), per condition.
- **FAR at a control-calibrated threshold.** Choose the threshold that keeps 80% of human-accepted control images in the same track, then count human-rejected images scoring at or above it. This mirrors a practitioner who tunes an evaluator on generic data.
- **Uncertainty.** 2,000 bootstrap resamples of whole matched pairs, since images of a pair share a template, with 95% percentile intervals. Two-sided bootstrap p-values are for difference statistics only.
- **Best-of-N.** Per prompt, capture = (human score of the evaluator's top pick − mean human score) / (best human score − mean), pooled over prompts. The report also plots expected pick quality over subset sizes n = 1…N.
- **Script complexity.** The share of non-space code points in Unicode categories Mn/Mc (combining marks). It is a crude, reproducible proxy, correlated with per-script AUROC across nine scripts; exploratory only.
- **Robustness.** Report the H1 gaps per generator. A gap that appears for only one generator is reported as such.

**Decision rule for H1.** Supported for an evaluator if the Holm-adjusted p < 0.05 and the 95% interval excludes zero. Interpret the size relative to the human-ceiling gap in the same conditions.

## 7. Power (to be updated after the pilot)

A synthetic simulation (`tests/test_study.py::synthetic_study`) had human agreement of α ≈ 0.77 and 65 pairs with 4 images per prompt and one generator. In it, the pooled H1 interval for an unbiased evaluator was about ±0.035 AUROC, and an injected gap of 0.4 was detected with p < 0.001. Real label noise will be higher. After the pilot, re-run that simulation with the observed agreement and good-image rates to estimate the smallest detectable gap, and add prompts if it exceeds 0.05.

## 8. Threats to validity

- **Imperfect controls.** Diwali and Christmas scenes differ in visual complexity, not only in culture. Mitigations: the three-way script design, per-category reporting, and reporting the human good rate per condition. AUROC does not depend on base rates, but it does depend on how hard the good-versus-bad cases are to tell apart.
- **Annotator background.** Indian annotators may judge Western controls less strictly, or the reverse. Record backgrounds; if possible, add a second panel from outside India for the controls.
- **OCR confound.** When an OCR-based evaluator fails on Indic text, the OCR engine may be the problem rather than the image. Human transcriptions let the two be told apart.
- **Tokenizer truncation.** English CLIP tokenizers split Indic text into byte fragments. That is part of the phenomenon being measured, not something to correct for.
- **Generator drift.** Record exact model versions, seeds and sampler settings in the manifest.

## 9. Related work to check before writing

- Cultural text-to-image benchmarks (e.g. CUBE) and studies of cultural representation in image generators.
- Validation of automatic text-to-image evaluators against humans (TIFA, VQAScore, GenEval, and critiques of CLIPScore).
- Multilingual and multi-script text rendering in diffusion models.

The claimed contribution is the controlled, matched design with a script/locale decomposition and native-reader labels. Confirm that no prior work already reports this for Indic scripts.
