"""Decision-level evaluation of the VerifiedGen verifier against gold labels.

For the evaluator-reliability study (matched India/control prompts, human
panels, baselines, bootstrap intervals) use ``scripts/run_study.py``.
"""
import argparse, json
from pathlib import Path
from .cli import load_manifest
from .metrics import evaluate_reports
from .verifier import ThresholdConfig, verify_candidate


def threshold_config_from_file(path):
    if not path:
        return ThresholdConfig()
    data = json.loads(Path(path).read_text(encoding="utf-8")).get("thresholds", {})
    known = set(ThresholdConfig.__dataclass_fields__)
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown threshold keys in {path}: {sorted(unknown)}")
    return ThresholdConfig(**data)


def run_experiment_suite(manifest, output_dir, config=None):
    config = config or ThresholdConfig()
    reports = [verify_candidate(c, config) for c in load_manifest(manifest)]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"manifest": str(manifest), "thresholds": config.__dict__, "metrics": evaluate_reports(reports)}
    (out / "summary.json").write_text(json.dumps(payload, indent=2))
    return payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--config-file")
    a = p.parse_args()
    print(json.dumps(run_experiment_suite(a.manifest, a.output_dir, threshold_config_from_file(a.config_file)), indent=2))


if __name__ == "__main__": main()
