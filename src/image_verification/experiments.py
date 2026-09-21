import argparse, json
from pathlib import Path
from .cli import load_manifest
from .metrics import evaluate_reports
from .verifier import ThresholdConfig, verify_candidate

def run_experiment_suite(manifest,output_dir,methods=None,config=None,**kwargs):
    methods=methods or ["verifiedgen"]; config=config or ThresholdConfig(); candidates=load_manifest(manifest); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); payload={"manifest":str(manifest),"methods":{}}
    for method in methods:
        reports=[verify_candidate(c,config) for c in candidates]; payload["methods"][method]={"metrics":evaluate_reports(reports)}
    (out/"summary.json").write_text(json.dumps(payload,indent=2)); (out/"results.md").write_text("# VerifiedGen Results\n\n"+json.dumps(payload,indent=2)); return payload
def main():
    p=argparse.ArgumentParser(); p.add_argument("--manifest",required=True); p.add_argument("--output-dir",required=True); p.add_argument("--config-file"); p.add_argument("--bootstrap-samples",type=int,default=0); a=p.parse_args(); run_experiment_suite(a.manifest,a.output_dir)
if __name__ == "__main__": main()

