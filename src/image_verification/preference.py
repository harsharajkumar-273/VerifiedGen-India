"""Preference-data and reward-model training for VerifiedGen-India.

The first trainable component is a small pairwise reward model. It learns to
rank candidates from verifier features and human preferences. This is safer
and more data-efficient than immediately fine-tuning the full diffusion model.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

FEATURE_NAMES = [
    "overall_score", "prompt_alignment", "visual_quality", "artifact_risk",
    "safety", "text_rendering", "cultural_context", "reward",
]

def _axis_map(report: Dict) -> Dict[str, float]:
    return {axis["name"]: float(axis["score"]) for axis in report.get("axis_scores", [])}

def report_features(report: Dict) -> List[float]:
    axes = _axis_map(report)
    return [
        float(report.get("overall_score", 0.0)),
        axes.get("prompt_alignment", 0.5),
        axes.get("visual_quality", 0.5),
        axes.get("artifact_risk", 0.5),
        axes.get("safety", 1.0),
        axes.get("text_rendering", 0.5),
        axes.get("cultural_context", 0.5),
        float(report.get("reward", report.get("overall_score", 0.0))),
    ]

def build_preference_pairs(scored_jsonl: str | Path, output_jsonl: str | Path) -> List[Dict]:
    rows = [json.loads(line) for line in Path(scored_jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]
    groups: Dict[str, List[Dict]] = {}
    for row in rows:
        groups.setdefault(row["prompt_id"], []).append(row)
    pairs = []
    for prompt_id, candidates in groups.items():
        ranked = sorted(candidates, key=lambda row: (row.get("human_rank") is None, row.get("human_rank") or 10**9, -row.get("reward", 0.0)))
        if len(ranked) < 2:
            continue
        for better, worse in zip(ranked, ranked[1:]):
            if better.get("human_rank") is None and worse.get("human_rank") is None:
                continue
            pairs.append({
                "prompt_id": prompt_id,
                "chosen_id": better["candidate_id"],
                "rejected_id": worse["candidate_id"],
                "chosen_features": report_features(better.get("report", better)),
                "rejected_features": report_features(worse.get("report", worse)),
                "reasons": worse.get("failure_reasons", []),
            })
    out = Path(output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(pair, ensure_ascii=False) for pair in pairs) + ("\n" if pairs else ""), encoding="utf-8")
    return pairs

def split_pairs(pairs: List[Dict], validation_fraction: float = 0.2, seed: int = 7) -> Tuple[List[Dict], List[Dict]]:
    items = list(pairs); random.Random(seed).shuffle(items)
    cut = max(1, int(len(items) * (1.0 - validation_fraction))) if items else 0
    return items[:cut], items[cut:]

def train_reward_model(pairs_jsonl: str | Path, output_dir: str | Path, epochs: int = 20, learning_rate: float = 1e-2, seed: int = 7) -> Dict:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install torch to train the reward model.") from exc
    pairs = [json.loads(line) for line in Path(pairs_jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]
    train, valid = split_pairs(pairs, seed=seed)
    if not train:
        raise ValueError("No preference pairs found. Add human_rank labels before training.")
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(len(FEATURE_NAMES), 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.LogSigmoid()
    history = []
    for epoch in range(epochs):
        total = 0.0
        for pair in train:
            chosen = torch.tensor(pair["chosen_features"], dtype=torch.float32).unsqueeze(0)
            rejected = torch.tensor(pair["rejected_features"], dtype=torch.float32).unsqueeze(0)
            loss = -loss_fn(model(chosen) - model(rejected)).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step(); total += float(loss.detach())
        history.append({"epoch": epoch + 1, "train_loss": total / len(train)})
    def accuracy(items):
        if not items: return None
        correct = 0
        with torch.no_grad():
            for pair in items:
                a = model(torch.tensor(pair["chosen_features"], dtype=torch.float32)).item()
                b = model(torch.tensor(pair["rejected_features"], dtype=torch.float32)).item()
                correct += int(a > b)
        return correct / len(items)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "feature_names": FEATURE_NAMES}, out / "reward_model.pt")
    summary = {"train_pairs": len(train), "validation_pairs": len(valid), "train_accuracy": accuracy(train), "validation_accuracy": accuracy(valid), "history": history}
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

def main():
    parser = argparse.ArgumentParser(description="Build preference pairs or train a reward model.")
    sub = parser.add_subparsers(dest="command", required=True)
    pairs = sub.add_parser("pairs"); pairs.add_argument("--scored-jsonl", required=True); pairs.add_argument("--output-jsonl", required=True)
    train = sub.add_parser("train"); train.add_argument("--pairs-jsonl", required=True); train.add_argument("--output-dir", required=True); train.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    if args.command == "pairs": print(f"[preference] wrote {len(build_preference_pairs(args.scored_jsonl, args.output_jsonl))} pairs")
    else: print(json.dumps(train_reward_model(args.pairs_jsonl, args.output_dir, args.epochs), indent=2))

if __name__ == "__main__": main()
