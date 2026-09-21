import argparse, json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
def main():
    from src.image_verification.language_metrics import evaluate_ocr_rows
    p=argparse.ArgumentParser(); p.add_argument("--manifest",required=True); p.add_argument("--output-json",required=True); a=p.parse_args()
    rows=[json.loads(x) for x in Path(a.manifest).read_text(encoding="utf-8").splitlines() if x.strip()]
    result=evaluate_ocr_rows(rows); Path(a.output_json).parent.mkdir(parents=True,exist_ok=True); Path(a.output_json).write_text(json.dumps(result,indent=2,ensure_ascii=False)); print(json.dumps(result,indent=2,ensure_ascii=False))
if __name__ == "__main__": main()
