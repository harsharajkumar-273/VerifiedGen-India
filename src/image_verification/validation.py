import argparse, json
from pathlib import Path
from .cli import load_manifest
def validate_manifest(path):
    try:
        rows=load_manifest(path); errors=[]
        for row in rows:
            if not Path(row.image_path).exists(): errors.append(f"missing image: {row.image_path}")
        return {"valid":not errors,"candidate_count":len(rows),"errors":errors}
    except Exception as e: return {"valid":False,"candidate_count":0,"errors":[str(e)]}
def main():
    p=argparse.ArgumentParser(); p.add_argument("--manifest",required=True); p.add_argument("--output-json",required=True); a=p.parse_args(); r=validate_manifest(a.manifest); Path(a.output_json).parent.mkdir(parents=True,exist_ok=True); Path(a.output_json).write_text(json.dumps(r,indent=2)); print(r)
if __name__ == "__main__": main()

