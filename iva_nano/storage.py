"""Local artifact storage; no Cloudflare R2 dependency."""
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


class LocalArtifactStore:
    def __init__(self, root="artifacts"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, name, checkpoint, torch_module):
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                torch_module.save(checkpoint, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, target)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        item = {
            "file": str(target.relative_to(self.root)),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "bytes": target.stat().st_size,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "storage": "local_filesystem",
        }
        manifest_path = self.root / "artifact_manifest.json"
        items = []
        if manifest_path.exists():
            try:
                items = json.loads(manifest_path.read_text(encoding="utf-8")).get("artifacts", [])
            except (OSError, json.JSONDecodeError):
                items = []
        items = [old for old in items if old.get("file") != item["file"]]
        items.append(item)
        manifest_path.write_text(json.dumps({"artifacts": items}, ensure_ascii=False, indent=2), encoding="utf-8")
        return target
