import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.image_verification.cli import load_manifest
from src.image_verification.reward import constrained_decision, reward_from_report
from src.image_verification.verifier import verify_candidate

def main():
    p=argparse.ArgumentParser(); p.add_argument("--manifest",required=True); p.add_argument("--output-jsonl",required=True); a=p.parse_args()
    rows=[]
    for candidate in load_manifest(a.manifest):
        report=verify_candidate(candidate); rows.append({"candidate_id":candidate.candidate_id,"prompt_id":candidate.prompt_id,"decision":report.decision,"constrained_decision":constrained_decision(report),"reward":reward_from_report(report),"report":report.to_dict()})
    out=Path(a.output_jsonl); out.parent.mkdir(parents=True,exist_ok=True); out.write_text("\n".join(json.dumps(row,ensure_ascii=False) for row in rows)+"\n"); print(f"[score] wrote {len(rows)} reports -> {out}")
if __name__ == "__main__": main()
