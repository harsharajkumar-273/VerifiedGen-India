"""Model-backed evaluators: OCR, VLM judges, VQAScore and OpenCLIP.

All of them read the requested text from ``metadata["expected_text"]`` (set by
the glyph-pair and study manifests) and the prompt from ``candidate.prompt``.
Heavy dependencies are imported lazily so the rest of the package stays
standard-library + numpy + Pillow.

* ``ocr_tesseract`` – Tesseract in the language of the *requested* script,
  score = 1 - CER(requested text, best reading). Six readings per image (raw and
  Otsu-binarised, page-segmentation modes 7, 6 and 13) and the one closest to
  the requested text is used, so OCR gets its best shot; this is symmetric for
  the two images of a pair. With ``crop=True`` it reads
  only the sign region recorded by the renderer (``metadata["text_box"]``):
  an oracle text detector, so it is an upper bound for OCR-based checking.
* ``vlm_judge`` – a vision-language model asked the standard judge question;
  score = P("yes"). Backends: ``hf`` (local transformers model) or ``openai``
  (any OpenAI-compatible chat endpoint with logprobs: vLLM, OpenRouter, ...).
  ``ocr_grounded=True`` adds the Tesseract reading to the question, which is
  one of the fixes under test.
* ``vqascore`` – VQAScore through the ``t2v_metrics`` package.
* ``openclip`` – CLIPScore with an OpenCLIP checkpoint, e.g. the multilingual
  ``xlm-roberta-large-ViT-H-14``.
"""
from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import tempfile
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .evaluators import Evaluator
from .language_metrics import character_error_rate
from .schema import ImageCandidate

TESSDATA_DIR = Path(__file__).resolve().parents[2] / "data" / "tessdata"
TESSDATA_URL = "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/{lang}.traineddata"

# Unicode block start -> Tesseract language.
BLOCK_LANG = {0x0900: "hin", 0x0980: "ben", 0x0A00: "pan", 0x0A80: "guj", 0x0B00: "ori",
              0x0B80: "tam", 0x0C00: "tel", 0x0C80: "kan", 0x0D00: "mal"}

JUDGE_PROMPT = (
    'This image was generated for the prompt: "{prompt}". Would you show this image to the person who typed this '
    "prompt? Check that any requested text is spelled correctly in the requested script, that the setting and "
    "cultural details match the prompt, and that requested objects are present. Answer yes or no."
)
OCR_HINT = ' An OCR system read the text in the image as: "{ocr}". The OCR may itself be wrong.'


def requested_text(candidate: ImageCandidate) -> str:
    expected = (candidate.metadata or {}).get("expected_text") or []
    return " ".join(expected) if isinstance(expected, list) else str(expected)


def tesseract_lang(text: str) -> str:
    """Tesseract language for the dominant script of ``text`` (Latin -> eng)."""
    counts: Dict[str, int] = {}
    for char in text:
        if char.isspace() or unicodedata.category(char).startswith("P"):
            continue
        block = (ord(char) // 0x80) * 0x80
        lang = BLOCK_LANG.get(block, "eng")
        counts[lang] = counts.get(lang, 0) + 1
    return max(counts, key=counts.get) if counts else "eng"


def fetch_tessdata(langs: Sequence[str], tessdata_dir: Path = TESSDATA_DIR) -> None:
    import urllib.request

    tessdata_dir.mkdir(parents=True, exist_ok=True)
    for lang in langs:
        target = tessdata_dir / f"{lang}.traineddata"
        if not target.exists():
            with urllib.request.urlopen(TESSDATA_URL.format(lang=lang), timeout=120) as response:
                target.write_bytes(response.read())


def _squash(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFC", text) if not ch.isspace() and ch not in "|_~")


def binarize(image):
    """Otsu threshold, dark text on a white background with a white margin."""
    import numpy as np
    from PIL import Image, ImageOps

    array = np.asarray(image.convert("L"), dtype=float)
    hist, _ = np.histogram(array, bins=256, range=(0, 256))
    prob = hist / max(1, hist.sum())
    weight = np.cumsum(prob)
    mean = np.cumsum(prob * np.arange(256))
    between = (mean[-1] * weight - mean) ** 2 / (weight * (1 - weight) + 1e-12)
    mask = array > int(np.nanargmax(between))
    if mask.mean() < 0.5:  # text is the minority: make the majority white
        mask = ~mask
    return ImageOps.expand(Image.fromarray((mask * 255).astype("uint8")), border=20, fill=255)


class TesseractReader:
    """Runs Tesseract once per (image, language, crop) and caches the readings."""

    def __init__(self, tessdata_dir: Path = TESSDATA_DIR, psms: Sequence[int] = (7, 6, 13), crop: bool = True,
                 cache_path: Optional[str] = None, upscale: int = 2, modes: Sequence[str] = ("raw", "otsu")):
        self.tessdata_dir = Path(tessdata_dir)
        self.psms = tuple(psms)
        self.modes = tuple(modes)
        self.crop = crop
        self.upscale = upscale
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: Dict[str, List[str]] = {}
        if self.cache_path and self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))

    def readings(self, image_path: str, lang: str, box: Optional[Sequence[int]]) -> List[str]:
        from PIL import Image

        key = f"{image_path}|{lang}|{list(box) if (self.crop and box) else None}|{self.psms}|{self.modes}"
        if key in self.cache:
            return self.cache[key]
        if not (self.tessdata_dir / f"{lang}.traineddata").exists():
            fetch_tessdata([lang], self.tessdata_dir)
        image = Image.open(image_path).convert("L")
        if self.crop and box:
            image = image.crop(tuple(box))
        if self.upscale > 1:
            image = image.resize((image.width * self.upscale, image.height * self.upscale))
        out: List[str] = []
        for mode in self.modes:
            out.extend(self._run(binarize(image) if mode == "otsu" else image, lang))
        self.cache[key] = out
        return out

    def _run(self, image, lang: str) -> List[str]:
        out: List[str] = []
        with tempfile.NamedTemporaryFile(suffix=".png") as handle:
            image.save(handle.name)
            for psm in self.psms:
                # One thread per process: parallelism comes from the worker pool, and
                # Tesseract's own OpenMP threads would otherwise oversubscribe the CPU.
                result = subprocess.run(
                    ["tesseract", handle.name, "-", "--tessdata-dir", str(self.tessdata_dir), "-l", lang, "--psm", str(psm)],
                    capture_output=True, text=True, check=False, env={**os.environ, "OMP_THREAD_LIMIT": "1"},
                )
                out.append(result.stdout.strip())
        return out

    def save(self) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")

    def best_reading(self, candidate: ImageCandidate) -> str:
        target = requested_text(candidate)
        readings = self.readings(candidate.image_path, tesseract_lang(target), (candidate.metadata or {}).get("text_box"))
        return min(readings, key=lambda r: character_error_rate(_squash(target), _squash(r))) if readings else ""


class TesseractOcrEvaluator(Evaluator):
    name = "ocr_tesseract"

    def __init__(self, tessdata_dir: str = str(TESSDATA_DIR), crop: bool = True, cache_path: Optional[str] = None,
                 name: Optional[str] = None):
        self.reader = TesseractReader(Path(tessdata_dir), crop=crop, cache_path=cache_path)
        if name:
            self.name = name
        elif not crop:
            self.name = "ocr_tesseract_fullimage"

    def score(self, candidate: ImageCandidate) -> Optional[float]:
        target = requested_text(candidate)
        if not target:
            return None
        return max(0.0, 1.0 - character_error_rate(_squash(target), _squash(self.reader.best_reading(candidate))))

    def score_all(self, candidates, workers: int = os.cpu_count() or 4):
        from concurrent.futures import ThreadPoolExecutor

        candidates = list(candidates)
        # Tesseract runs in subprocesses, so threads parallelise the OCR itself.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(self.reader.best_reading, candidates))
        scores = super().score_all(candidates)
        self.reader.save()
        return scores


def _yes_probability(token_logprobs: Dict[str, float]) -> Optional[float]:
    import math

    yes = sum(math.exp(lp) for tok, lp in token_logprobs.items() if tok.strip().lower() == "yes")
    no = sum(math.exp(lp) for tok, lp in token_logprobs.items() if tok.strip().lower() == "no")
    return yes / (yes + no) if yes + no > 0 else None


class VlmJudgeEvaluator(Evaluator):
    """P("yes") from a vision-language model asked the standard judge question."""

    name = "vlm_judge"

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", backend: str = "hf", name: Optional[str] = None,
                 base_url: Optional[str] = None, api_key_env: str = "OPENAI_API_KEY", ocr_grounded: bool = False,
                 tessdata_dir: str = str(TESSDATA_DIR), device: Optional[str] = None, prompt_template: str = JUDGE_PROMPT):
        self.model_name = model_name
        self.backend = backend
        self.prompt_template = prompt_template
        self.ocr = TesseractReader(Path(tessdata_dir)) if ocr_grounded else None
        self.name = name or ("vlm_judge_ocr" if ocr_grounded else "vlm_judge") + ":" + model_name.split("/")[-1]
        if backend == "hf":
            self._load_hf(device)
        elif backend == "openai":
            self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
            self.api_key = os.environ.get(api_key_env, "")
        else:
            raise ValueError(f"Unknown backend {backend!r}")

    def question(self, candidate: ImageCandidate) -> str:
        text = self.prompt_template.format(prompt=candidate.prompt)
        if self.ocr is not None:
            text += OCR_HINT.format(ocr=self.ocr.best_reading(candidate))
        return text

    # -- local transformers model
    def _load_hf(self, device: Optional[str]) -> None:
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("The hf backend needs torch and transformers>=4.45.") from exc
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(self.model_name, torch_dtype=dtype).to(self.device).eval()
        tokenizer = self.processor.tokenizer
        self.yes_ids = sorted({tokenizer.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes", " yes", " Yes")})
        self.no_ids = sorted({tokenizer.encode(w, add_special_tokens=False)[0] for w in ("no", "No", " no", " No")})

    def _score_hf(self, candidate: ImageCandidate) -> float:
        from PIL import Image

        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.question(candidate)}]}]
        chat = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        image = Image.open(candidate.image_path).convert("RGB")
        inputs = self.processor(text=[chat], images=[image], return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            logits = self.model(**inputs).logits[0, -1].float()
        probs = self.torch.softmax(logits, dim=-1)
        yes, no = float(probs[self.yes_ids].sum()), float(probs[self.no_ids].sum())
        return yes / (yes + no) if yes + no > 0 else 0.5

    # -- OpenAI-compatible endpoint
    def _score_openai(self, candidate: ImageCandidate) -> Optional[float]:
        import urllib.request

        data = base64.b64encode(Path(candidate.image_path).read_bytes()).decode("ascii")
        mime = "image/png" if candidate.image_path.endswith(".png") else "image/jpeg"
        body = {
            "model": self.model_name, "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
                {"type": "text", "text": self.question(candidate)},
            ]}],
        }
        request = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(body).encode("utf-8"),
                                         headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(request, timeout=120) as response:
            choice = json.loads(response.read())["choices"][0]
        content = ((choice.get("logprobs") or {}).get("content") or [])
        if content:
            top = {item["token"]: item["logprob"] for item in content[0].get("top_logprobs", [])}
            value = _yes_probability(top)
            if value is not None:
                return value
        answer = (choice.get("message", {}).get("content") or "").strip().lower()
        return 1.0 if answer.startswith("yes") else (0.0 if answer.startswith("no") else None)

    def score(self, candidate: ImageCandidate) -> Optional[float]:
        return self._score_hf(candidate) if self.backend == "hf" else self._score_openai(candidate)


class VQAScoreEvaluator(Evaluator):
    """VQAScore (Lin et al., 2024) via ``pip install t2v_metrics``."""

    name = "vqascore"

    def __init__(self, model_name: str = "clip-flant5-xxl", name: Optional[str] = None):
        try:
            import t2v_metrics
        except ImportError as exc:
            raise RuntimeError("VQAScoreEvaluator needs `pip install t2v_metrics`.") from exc
        self.metric = t2v_metrics.VQAScore(model=model_name)
        self.name = name or f"vqascore:{model_name}"

    def score(self, candidate: ImageCandidate) -> float:
        return float(self.metric(images=[candidate.image_path], texts=[candidate.prompt]).item())


class OpenClipScoreEvaluator(Evaluator):
    """Cosine similarity with an OpenCLIP checkpoint (e.g. multilingual XLM-R CLIP)."""

    name = "openclip"

    def __init__(self, model_name: str = "xlm-roberta-large-ViT-H-14", pretrained: str = "frozen_laion5b_s13b_b90k",
                 device: Optional[str] = None, name: Optional[str] = None):
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise RuntimeError("OpenClipScoreEvaluator needs `pip install open_clip_torch`.") from exc
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.name = name or f"openclip:{model_name}"

    def score(self, candidate: ImageCandidate) -> float:
        from PIL import Image

        image = self.preprocess(Image.open(candidate.image_path).convert("RGB")).unsqueeze(0).to(self.device)
        text = self.tokenizer([candidate.prompt]).to(self.device)
        with self.torch.no_grad():
            image_embeds = self.model.encode_image(image)
            text_embeds = self.model.encode_text(text)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return float((image_embeds * text_embeds).sum())
