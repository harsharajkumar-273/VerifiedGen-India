"""Power of the pooled Latin-vs-Indic reading-accuracy gap (concept-clustered bootstrap)."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.image_verification.glyph_eval import simulate_power

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--concepts", type=int, default=18)
    p.add_argument("--variants", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--acc-latin", type=float, default=0.85)
    p.add_argument("--acc-indic", type=float, nargs="+", default=[0.80, 0.75, 0.70])
    p.add_argument("--concept-sd", type=float, default=0.5)
    p.add_argument("--sims", type=int, default=200)
    a = p.parse_args()
    for v in a.variants:
        for acc in a.acc_indic:
            r = simulate_power(a.concepts, v, a.acc_latin, acc, concept_sd=a.concept_sd, sims=a.sims)
            print(json.dumps({"variants": v, "acc_indic": acc, "power": r["power"], "pairs_per_sim": r["n_pairs_per_sim"]}))
