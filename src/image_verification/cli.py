import argparse, json
from pathlib import Path
from .schema import ImageCandidate
from .verifier import ThresholdConfig, verify_candidate, rank_reports

def load_manifest(path):
    root=Path(path).parent; rows=[]
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row=json.loads(line); p=Path(row["image_path"]); row["image_path"]=str(p if p.is_absolute() else root/p); rows.append(ImageCandidate.from_dict(row))
    return rows

def run_verification(manifest, output_jsonl, summary_json, config=None):
    reports=[verify_candidate(c, config or ThresholdConfig()) for c in load_manifest(manifest)]
    Path(output_jsonl).parent.mkdir(parents=True,exist_ok=True); Path(output_jsonl).write_text("\n".join(json.dumps(r.to_dict(),ensure_ascii=False) for r in reports)+"\n")
    groups={}
    for r in reports: groups.setdefault(r.prompt_id,[]).append(r)
    summary={"candidate_count":len(reports),"selected_candidates":[rank_reports(rs)[0].to_dict() for rs in groups.values()]}
    Path(summary_json).parent.mkdir(parents=True,exist_ok=True); Path(summary_json).write_text(json.dumps(summary,indent=2,ensure_ascii=False)); return summary

def main():
    p=argparse.ArgumentParser(); p.add_argument("--manifest",required=True); p.add_argument("--output-jsonl",required=True); p.add_argument("--summary-json",required=True); args=p.parse_args(); run_verification(args.manifest,args.output_jsonl,args.summary_json)
if __name__ == "__main__": main()

