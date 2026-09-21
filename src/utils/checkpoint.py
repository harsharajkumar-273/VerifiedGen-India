import json
import os
from datetime import datetime, timezone
from pathlib import Path

def _now(): return datetime.now(timezone.utc).isoformat()

class RunCheckpoint:
    def __init__(self, path, stage):
        self.path = Path(path)
        self.state = {"stage": stage, "started_at": _now(), "updated_at": _now(), "completed": {}, "skipped": {}, "failures": {}}
        self.load()
    def load(self):
        if self.path.exists(): self.state.update(json.loads(self.path.read_text()))
    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True); self.state["updated_at"] = _now(); temp = self.path.with_suffix(".tmp"); temp.write_text(json.dumps(self.state, indent=2)); os.replace(temp, self.path)
    def output_completed(self, item_id, output_path, overwrite=False):
        if overwrite: return False
        path = Path(output_path)
        if path.exists() and path.stat().st_size > 0:
            self.state["skipped"][item_id] = {"output_path": str(path), "reason": "output_exists"}; self.save(); return True
        return False
    def mark_completed(self, item_id, output_path, metadata=None):
        self.state["completed"][item_id] = {"output_path": str(output_path), "metadata": metadata or {}, "completed_at": _now()}; self.save()
    def mark_failed(self, item_id, error, metadata=None):
        self.state["failures"][item_id] = {"error": str(error), "metadata": metadata or {}, "failed_at": _now()}; self.save()
    def summary(self): return {key: len(self.state.get(key, {})) for key in ("completed", "skipped", "failures")}

