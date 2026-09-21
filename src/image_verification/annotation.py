import argparse, json
from pathlib import Path
from .cli import load_manifest
def export_annotation_tasks(manifest, output):
    rows=[]
    for c in load_manifest(manifest): rows.append({"candidate_id":c.candidate_id,"prompt_id":c.prompt_id,"prompt":c.prompt,"image_path":c.image_path,"overall_label":None,"prompt_alignment_score":None,"visual_quality_score":None,"artifact_severity":None,"failure_types":[]})
    Path(output).parent.mkdir(parents=True,exist_ok=True); Path(output).write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in rows)+"\n"); return rows
def main():
    p=argparse.ArgumentParser(); p.add_argument("command",choices=["export"]); p.add_argument("--manifest",required=True); p.add_argument("--output-jsonl",required=True); a=p.parse_args(); print(f"[annotation] wrote {len(export_annotation_tasks(a.manifest,a.output_jsonl))} tasks")
if __name__ == "__main__": main()

