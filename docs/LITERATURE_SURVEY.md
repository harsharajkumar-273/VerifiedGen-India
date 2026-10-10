# Literature survey: do image evaluators read Indic scripts?

Status: first pass, 2026-10-10. It is built from search-result summaries, abstracts and project pages. arxiv.org could not be fetched from the working environment, so **every entry marked (abstract-only) still needs a full-text read** before it is cited in a paper.

## The question being positioned

Automatic evaluators are used to score and select text-to-image outputs. The evaluators in scope are CLIPScore, VQAScore, VLM judges, OCR-based text metrics and text-rendering reward models. When an image is supposed to contain Indic-script text, do these evaluators tell correctly rendered text apart from broken text? Is that ability worse than for Latin-script text?

Planned method: synthetic **glyph minimal pairs**. Correct text is rendered with fonts onto real scenes, next to copies with one controlled, script-specific error. The design adds a Latin-script control, a best-of-N harm analysis and a fix.

## 1. Meta-evaluation of text-to-image evaluators

| Work | What it does | Relevance |
|---|---|---|
| **T2IScoreScore (TS2)**, Saxon et al., NeurIPS 2024 ([proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/9b9cfd5428153ccfbd4ba34b7e007305-Abstract.html)) | Uses "semantic error graphs": a prompt plus images with an increasing, known number of errors. Metrics are scored by ordering (Spearman) and separation (Kolmogorov–Smirnov). Finds that VLM-based metrics (TIFA, DSG, VIEScore) do not clearly beat CLIPScore on hard, natural errors. | **The closest methodological template.** Glyph minimal pairs are TS2 for scripts, with controlled errors instead of model errors. Reuse its meta-metrics. Its error types are objects and attributes; it has none for text or scripts. |
| **VQAScore / GenAI-Bench**, Lin et al., ECCV 2024 ([arXiv 2404.01291](https://arxiv.org/abs/2404.01291)) | Scores an image as P("Yes") to "Does this figure show {prompt}?" with CLIP-FlanT5. Comes with 1,600 compositional prompts and over 15k human ratings. Code is in `t2v_metrics`. | A primary evaluator under test. Its validation data is English and compositional, with no Indic text. |
| **Gecko**, Wiles et al., ICLR 2025 ([arXiv 2404.16820](https://arxiv.org/pdf/2404.16820)) | Over 100k human ratings across four rating templates. Shows that a metric's ranking depends on the setting: model ordering, pairwise, or pointwise. | Argues for reporting pairwise *and* pointwise results. English only. |
| **MJ-Bench** ([arXiv 2407.04842](https://arxiv.org/pdf/2407.04842)) (abstract-only) | Benchmarks multimodal reward models as judges for text-to-image feedback. | A source of VLM-judge and reward-model baselines. Not multilingual. |
| **Lost in Translation: Do LVLM Judges Generalize Across Languages?**, ACL 2026 ([note](https://en.papernotes.org/ACL2026/multimodal_vlm/lost_in_translation_do_lvlm_judges_generalize_across_languages/)) (abstract-only) | Keeps the image fixed and varies only the judging language. Open-source judges degrade most. Translating a reward benchmark into 24 languages and fine-tuning on it recovers much of the loss. | **Takes most of the novelty from the "prompt-language invariance" idea (Idea 2).** Language of the *prompt*, not text *in the image*. Check whether Indic languages are included. |
| **LingT2I**, arXiv 2608.11002 ([commentary](https://pith.science/paper/2608.11002)) (abstract-only) | A 33k-prompt, 10-language T2I benchmark run on 17 models. Non-Latin prompts drop most. Uses MetaCLIP-2 CLIPScore and a Qwen2.5-VL judge. | Prompt-side multilingual generation. A commentary flags that judge circularity is not addressed, which is exactly our question. |

## 2. Evaluating rendered text in generated images

| Work | What it does | Relevance |
|---|---|---|
| **TextPecker**, Zhu et al., CVPR 2026 ([arXiv 2602.20903](https://arxiv.org/html/2602.20903)) | Shows that leading MLLMs and specialist OCR models nearly all fail to perceive *structural anomalies* (malformed strokes or characters) in rendered text; OCR models score about 0 F1. Builds a stroke-editing synthesis engine and an anomaly-aware reward for RL. Chinese and English. | **The closest prior work. It must be cited and differentiated.** It establishes the phenomenon for Chinese and English glyph *structure*. Our differences: (a) Indic scripts; (b) linguistically defined, script-specific errors (vowel signs, conjuncts, reordering, script swaps) rather than stroke damage; (c) the evaluators used in T2I papers (CLIPScore, VQAScore, judges), not only OCR; (d) a Latin control and best-of-N harm. Its released 8B model should be one of our evaluators. |
| **VISTAR** ([arXiv 2508.06152](https://arxiv.org/pdf/2508.06152)) (abstract-only) | For text rendering, a deterministic OCR script matched human pairwise judgments more often than a VLM judge (82.5% vs 76.1%, McNemar p < 0.01). | English evidence that VLM judges are weaker than OCR for text. Our study tests whether that holds, or reverses, for Indic. OCR may be even worse there. |
| **STRICT** ([arXiv 2505.18985](https://arxiv.org/html/2505.18985v1)) | A long-text rendering stress test in English, Chinese and French. Scored with OCR NED, CER and WER. | No Indic. Its OCR-metric pipeline is reusable. |
| **UltraText Bench** ([arXiv 2610.09823](https://arxiv.org/html/2610.09823)) | English and Chinese text rendering scored with a VLM judge (Q-Judger). Its own repo says its small human study does not validate the judge. | An example of an unvalidated judge. No Indic. |
| **OneIG-Bench** ([arXiv 2506.07977](https://arxiv.org/pdf/2506.07977)); LongText-Bench; CVTG-2K | Text rendering scored with OCR edit distance and completion ratio. | Criticised by TextPecker for structure-blind OCR. No Indic. |
| **Character-aware models improve visual text rendering**, Liu et al., ACL 2023 ([arXiv 2212.10562](https://arxiv.org/pdf/2212.10562)) | Character-blind text encoders (T5 and similar) cannot spell. Byte-level ByT5 fixes much of it. Very large models learn spelling but "did not generalize well beyond English". | Explains *why* generators fail on Indic text, so generated Indic text will often be pseudo-text. That makes reliable evaluators matter more. |
| **Glyph-ByT5-v2** ([arXiv 2406.10208](https://arxiv.org/pdf/2406.10208)); **EasyText** ([arXiv 2505.24417](https://arxiv.org/pdf/2505.24417)) | Multilingual rendering models. Glyph-ByT5-v2 covers 10 languages: English, Chinese, Japanese, Korean, French, German, Spanish, Italian, Portuguese, Russian. | **No Indic scripts.** Even "multilingual" rendering work stops before Indic, and so does its evaluation. |
| **MultiTextEdit** ([arXiv 2605.08163](https://arxiv.org/html/2605.08163)) | Text-in-image *editing* across 12 languages, Bengali included. Degradation is concentrated in text accuracy and script fidelity. The VLM judge was validated on 240 native-speaker samples (QWK about 0.76). | The only rendering benchmark found that includes an Indic script. Judge validation is pooled, and it does not ask whether reliability differs by script. |

## 3. Can VLMs and OCR read Indic scripts?

| Work | Finding | Relevance |
|---|---|---|
| **Do VLMs Read or Rewrite?** (FaithC4), Amazon ([arXiv 2607.21617](https://arxiv.org/pdf/2607.21617)) | Text with character-level perturbations (scrambles, substitutions, look-alikes) is rendered as documents. General VLMs *silently correct* it to the plausible word; traditional OCR does not. Correction is strongest for short words. English, Chinese, Korean. | **The key mechanism prediction.** A VLM judge shown a misspelled Indic sign may "read" it as the correct word and approve it. Stratify by word length, and include OCR-specialised models as a contrast. |
| **Can OCR-VLMs read Devanagari?** ([arXiv 2606.29213](https://arxiv.org/pdf/2606.29213)) (abstract-only) | Ten systems on Hindi. Most degrade sharply on real scans. English OCR performance does not predict Indic performance. | Evaluator weakness on Devanagari is likely. Choose OCR baselines from its leaderboard. |
| **IndicVisionBench**, ICLR 2026 ([arXiv 2511.04727](https://arxiv.org/pdf/2511.04727)) | English plus 10 Indic languages, including an OCR track of 876 documents. Drops for low-resource languages; proprietary models beat open ones. | Per-language evidence for choosing which scripts to cover. Document OCR only, not scene text. |
| **GlotOCR Bench**, ICML 2026 ([arXiv 2604.12978](https://arxiv.org/html/2604.12978v1)) | Covers 100+ Unicode scripts. Most OCR models work well on fewer than ten scripts. Ships a rendering pipeline. | Its rendering code may be reusable for the pair generator. |
| **BanglaWild** ([arXiv 2608.03884](https://arxiv.org/html/2608.03884)) | 2,535 in-the-wild Bengali scene-text photos, tested on OCR and VLMs. | Real Bengali sign scenes, possibly usable as backgrounds or a realism check. Check the licence. |
| **IndicSTR12**, ICDAR 2023 ([project](https://cvit.iiit.ac.in/research/projects/cvit-projects/indicstr)) | Real scene-text words in 12 Indic languages, plus 3M+ synthetic word images rendered with Unicode fonts. | Precedent for font-rendered Indic text. Possible source of vocabularies and fonts (CC BY-NC-ND, so check reuse). |
| **sarvamai/indic-ocr-bench** ([HF](https://huggingface.co/datasets/sarvamai/indic-ocr-bench)) | An Indic OCR dataset on the Hugging Face Hub. | Not yet inspected. |

## 4. Why CLIP-family metrics should struggle

- **Disentangling visual and written concepts in CLIP**, Materzyńska, Torralba, Bau, CVPR 2022 ([arXiv 2206.07835](https://arxiv.org/pdf/2206.07835)). CLIP's image encoder has a separable "spelling" subspace and responds strongly even to nonsense words. *Prediction:* CLIP may reward any text-looking marks, including pseudo-Indic glyphs.
- **Parrot Captions Teach CLIP to Spot Text**, Lin et al., ECCV 2024 ([paper](https://ecva.net/papers/eccv_2024/papers_ECCV/html/5898_ECCV_2024_paper.php)). About half of LAION-2B images contain text, and captions often copy it, so CLIP learns to spot text. That training text is overwhelmingly Latin, so the spotting ability may not transfer to Indic.
- **Tokenizer unfairness**: Petrov et al., NeurIPS 2023 ([proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/74bb24dca8334adce292883b4b651eda-Abstract.html)); "The Tokenizer Tax" ([2607.24276](https://tldr.takara.ai/p/2607.24276)), which reports Indic text at about 8× the English token cost under cl100k and 13× for Malayalam. CLIP's 77-token context truncates Indic text early, and its byte-level BPE breaks Indic characters into byte fragments. *Prediction:* the CLIPScore text side barely encodes the requested Indic word, so it cannot check spelling.

## 5. Cultural and India-centric T2I evaluation (background)

- **CulturalFrames**, EMNLP Findings 2025 ([ICML page](https://icml.cc/virtual/2025/50224)): 10 countries, 983 prompts, over 10k annotations. Metrics correlate poorly with human cultural judgments. **This covers the "cultural track" claim.**
- **IndicTTI / Navigating T2I Generative Bias across Indic Languages**, ECCV 2024 ([paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/12020_ECCV_2024_paper.php)): Indic-language *prompts* across 30 languages. Generation quality and consistency drop compared with English.
- Also in this area: Culture in Action ([2511.05681](https://arxiv.org/pdf/2511.05681)), Exposing Blindspots ([project](https://seochan99.github.io/ECB/)), a debiasing evaluator via cultural reward ([2607.15740](https://arxiv.org/html/2607.15740v1)), DRISHTIKON ([2509.19274](https://arxiv.org/pdf/2509.19274)), and geographic inclusion in T2I evaluation ([2405.04457](https://arxiv.org/pdf/2405.04457)).

## 6. Gap and positioning

**What exists:**
- Evaluators are unreliable for cultural content (CulturalFrames).
- VLM judges degrade when the *judging language* changes (Lost in Translation).
- OCR and MLLMs miss *structural* glyph damage in Chinese and English (TextPecker).
- VLMs autocorrect perturbed text (FaithC4).
- Indic OCR is weak (Devanagari stress test, IndicVisionBench, GlotOCR).

**What was not found:** any controlled test of whether **T2I evaluators** (CLIPScore, VQAScore, VLM judges, OCR metrics, rendering rewards) **detect errors in Indic-script text inside images**, against a matched Latin control. Also none on whether this lets wrong text through **best-of-N selection**, or on whether cheap fixes close the gap.

**Proposed contribution statement (draft):**
1. **Glyph minimal pairs**: a benchmark covering N Indic scripts plus Latin, with a linguistically grounded error taxonomy (below), so ground truth comes by construction.
2. The first measurement of evaluator sensitivity to Indic text errors: per evaluator, per script, per error type and per word length, using TS2-style meta-metrics.
3. Evidence of downstream harm: evaluator-driven best-of-N selection on real generator outputs, labelled by native readers.
4. A fix and an evaluation of it: OCR-grounded judging and/or a small judge trained on the synthetic pairs, tested on the real-image set.

**Main threat to novelty:** TextPecker or FaithC4 extending to Indic scripts. Move quickly, and cite both prominently as the foundation rather than hiding them.

## 7. Design implications taken from the survey

- **Error taxonomy, specific to Indic and realistic for generators:**
  1. A vowel sign (matra) deleted or swapped.
  2. Pre-base matra misordering: ि drawn after the consonant. This is a real shaping bug in rendering software.
  3. Text rendered **without shaping**, so conjuncts and halant forms fall apart. This is a realistic failure and is easy to make by disabling HarfBuzz/RAQM.
  4. A visually similar letter substituted.
  5. A neighbouring-script swap: Bengali for Assamese, Gujarati for Devanagari, Tamil for Malayalam.
  6. Pseudo-glyphs that look Indic.
  7. The word transliterated into Latin script.

  Each is matched to a Latin analogue where one exists (types 1, 4 and 6; type 7 has its own control).
- **Evaluators:**
  - CLIPScore, including a multilingual CLIP variant (MetaCLIP-2 or M-CLIP).
  - VQAScore.
  - Two or three VLM judges, open and hosted.
  - An Indic-capable OCR plus CER.
  - The TextPecker 8B model.
- **Meta-metrics:** pairwise accuracy (P[correct scored above corrupted]), TS2 Spearman/KS where there are graded error counts, and AUROC.
- **Stratify by word length.** FaithC4 found autocorrection concentrated in short words.
- **Human labels shrink:** native readers only need to confirm that each corruption type really is an error and that clean renders are correct, for example 50 pairs per script.

## 8. Still to do

- [ ] Read in full: TextPecker, FaithC4, Lost in Translation, LingT2I, T2IScoreScore, CulturalFrames, MultiTextEdit, the Devanagari OCR stress test.
- [ ] Confirm the language coverage of Lost in Translation and LingT2I. If either includes Indic text *rendered in images*, revise the positioning.
- [ ] Check the licences of the IndicSTR12, BanglaWild and GlotOCR rendering assets.
- [ ] Search Semantic Scholar or Google Scholar citation graphs forward from TextPecker and FaithC4, for anything after 2026-08.
