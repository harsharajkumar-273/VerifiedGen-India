"""A learned fix: a low-rank bilinear text-reading head on frozen CLIP features.

The score is ``s(image, prompt) = (U^T i) . (V^T t)`` where ``i`` and ``t`` are
frozen, L2-normalised image and text embeddings. It is trained with a pairwise
logistic loss so that, for every glyph pair, the image matching the prompt
scores higher. Both directions (forward and mirrored) are used, so the head
cannot win by preferring one kind of image regardless of the prompt.

Evaluation is cross-fitted: concepts are split into folds and every pair is
scored by a head that never saw its word, in any script. With
``--holdout-script`` the head is also never trained on that script, which
tests whether a fix learned on some scripts transfers to an unseen one.

Feature extraction needs torch + transformers (run it on a GPU machine);
training and scoring are numpy only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .evaluators import write_score_file


def extract_clip_features(manifest: str, out_path: str, model_name: str = "openai/clip-vit-large-patch14",
                          batch_size: int = 32, device: Optional[str] = None) -> Path:
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    from .cli import load_manifest

    candidates = load_manifest(manifest)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    images = sorted({c.image_path for c in candidates})
    prompts = sorted({c.prompt for c in candidates})

    def embed_images(paths: Sequence[str]) -> np.ndarray:
        out = []
        for start in range(0, len(paths), batch_size):
            batch = [Image.open(p).convert("RGB") for p in paths[start:start + batch_size]]
            with torch.no_grad():
                emb = model.get_image_features(**processor(images=batch, return_tensors="pt").to(device))
            out.append(torch.nn.functional.normalize(emb, dim=-1).float().cpu().numpy())
        return np.concatenate(out)

    def embed_texts(texts: Sequence[str]) -> np.ndarray:
        out = []
        for start in range(0, len(texts), batch_size):
            inputs = processor(text=list(texts[start:start + batch_size]), return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                emb = model.get_text_features(**inputs)
            out.append(torch.nn.functional.normalize(emb, dim=-1).float().cpu().numpy())
        return np.concatenate(out)

    image_index = {p: i for i, p in enumerate(images)}
    prompt_index = {t: i for i, t in enumerate(prompts)}
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, image_emb=embed_images(images), text_emb=embed_texts(prompts),
        candidate_ids=np.array([c.candidate_id for c in candidates]),
        candidate_image=np.array([image_index[c.image_path] for c in candidates]),
        candidate_text=np.array([prompt_index[c.prompt] for c in candidates]),
        model_name=np.array(model_name),
    )
    return out


class BilinearHead:
    def __init__(self, dim_image: int, dim_text: int, rank: int = 32, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.U = rng.normal(0, 1 / np.sqrt(dim_image), (dim_image, rank))
        self.V = rng.normal(0, 1 / np.sqrt(dim_text), (dim_text, rank))

    def score(self, image: np.ndarray, text: np.ndarray) -> np.ndarray:
        return np.sum((image @ self.U) * (text @ self.V), axis=-1)

    def fit(self, img_pos: np.ndarray, img_neg: np.ndarray, text: np.ndarray, epochs: int = 300,
            lr: float = 0.5, l2: float = 1e-3) -> "BilinearHead":
        """Full-batch gradient descent on mean log(1 + exp(-(s_pos - s_neg)))."""
        n = len(text)
        diff_img = img_pos - img_neg
        for _ in range(epochs):
            a = diff_img @ self.U  # (n, r)
            b = text @ self.V  # (n, r)
            margin = np.sum(a * b, axis=1)
            weight = -1.0 / (1.0 + np.exp(margin)) / n  # d loss / d margin
            grad_u = diff_img.T @ (weight[:, None] * b) + l2 * self.U
            grad_v = text.T @ (weight[:, None] * a) + l2 * self.V
            self.U -= lr * grad_u
            self.V -= lr * grad_v
        return self


def _concept_folds(concepts: Sequence[str], k: int, seed: int) -> Dict[str, int]:
    unique = sorted(set(concepts))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    return {unique[i]: int(rank % k) for rank, i in enumerate(order)}


def cross_fit_scores(pairs: Sequence[Dict], features: Dict[str, np.ndarray], folds: int = 6, rank: int = 32,
                     epochs: int = 300, seed: int = 0, holdout_script: Optional[str] = None) -> Dict[str, float]:
    """Score every candidate with a head trained on other concepts (and optionally other scripts)."""
    lookup = {cid: i for i, cid in enumerate(features["candidate_ids"].tolist())}
    image_emb, text_emb = features["image_emb"], features["text_emb"]
    cand_img, cand_txt = features["candidate_image"], features["candidate_text"]

    def emb(cid: str) -> Tuple[np.ndarray, np.ndarray]:
        i = lookup[cid]
        return image_emb[cand_img[i]], text_emb[cand_txt[i]]

    usable = [p for p in pairs if f"{p['pair_id']}:clean" in lookup and f"{p['pair_id']}:corrupt" in lookup]
    fold_of = _concept_folds([p["concept"] for p in usable], folds, seed)
    scores: Dict[str, float] = {}
    for fold in range(folds):
        train = [p for p in usable if fold_of[p["concept"]] != fold and p["is_error"]
                 and (holdout_script is None or p["script"] != holdout_script)]
        test = [p for p in usable if fold_of[p["concept"]] == fold
                and (holdout_script is None or p["script"] == holdout_script)]
        if not train or not test:
            continue
        pos = np.stack([emb(f"{p['pair_id']}:clean")[0] for p in train])
        neg = np.stack([emb(f"{p['pair_id']}:corrupt")[0] for p in train])
        txt = np.stack([emb(f"{p['pair_id']}:clean")[1] for p in train])
        head = BilinearHead(image_emb.shape[1], text_emb.shape[1], rank, seed + fold).fit(pos, neg, txt, epochs)
        for p in test:
            for role in ("clean", "corrupt"):
                cid = f"{p['pair_id']}:{role}"
                image, text = emb(cid)
                scores[cid] = float(head.score(image[None], text[None])[0])
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Learned text-reading head on frozen CLIP features (cross-fitted).")
    sub = parser.add_subparsers(dest="command", required=True)
    ext = sub.add_parser("extract", help="Embed images and prompts with CLIP (needs torch).")
    ext.add_argument("--manifest", required=True)
    ext.add_argument("--out", required=True)
    ext.add_argument("--model", default="openai/clip-vit-large-patch14")
    fit = sub.add_parser("score", help="Cross-fit the head and write a score file.")
    fit.add_argument("--pairs", required=True)
    fit.add_argument("--features", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--folds", type=int, default=6)
    fit.add_argument("--rank", type=int, default=32)
    fit.add_argument("--epochs", type=int, default=300)
    fit.add_argument("--holdout-script", help="Never train on this script; score only it (transfer test).")
    fit.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.command == "extract":
        print(f"[probe] wrote {extract_clip_features(args.manifest, args.out, args.model)}")
        return
    pairs = [json.loads(line) for line in Path(args.pairs).read_text(encoding="utf-8").splitlines() if line.strip()]
    features = dict(np.load(args.features, allow_pickle=False))
    scores = cross_fit_scores(pairs, features, args.folds, args.rank, args.epochs, args.seed, args.holdout_script)
    model = str(features.get("model_name", "clip")).split("/")[-1]
    name = f"probe:{model}" + (f":holdout-{args.holdout_script}" if args.holdout_script else "")
    print(f"[probe] {name}: {write_score_file(args.output, name, scores)} scores")


if __name__ == "__main__":
    main()
