import argparse, json
from pathlib import Path
from .prompt_bank import load_prompt_bank

EXT={".png",".jpg",".jpeg",".webp",".ppm"}
def build_manifest_from_folder(prompt_bank,image_dir,output_manifest,generator_name="external"):
    prompts={p.prompt_id:p for p in load_prompt_bank(prompt_bank)}; root=Path(image_dir); rows=[]
    for pid,prompt in prompts.items():
        folder=root/pid
        if not folder.exists(): continue
        for i,path in enumerate(sorted(x for x in folder.iterdir() if x.suffix.lower() in EXT),1):
            rows.append({"prompt_id":pid,"candidate_id":f"{pid}-{generator_name}-{i:02d}","generator":generator_name,"prompt":prompt.prompt,"category":prompt.category,"expected_objects":prompt.expected_objects,"image_path":str(path),**prompt.metadata})
    out=Path(output_manifest); out.parent.mkdir(parents=True,exist_ok=True); out.write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in rows)+"\n"); return rows
def main():
    p=argparse.ArgumentParser(); p.add_argument("--prompt-bank",required=True); p.add_argument("--image-dir",required=True); p.add_argument("--generator-name",default="external"); p.add_argument("--output-manifest",required=True); a=p.parse_args(); print(f"[manifest] wrote {len(build_manifest_from_folder(a.prompt_bank,a.image_dir,a.output_manifest,a.generator_name))} candidates")
if __name__ == "__main__": main()

