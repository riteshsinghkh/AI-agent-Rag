"""
Simple helper to persist latest uploaded file names.
"""

import json
from pathlib import Path
from typing import List


LATEST_UPLOADS_FILE = ".latest_uploads.json"


def record_latest_uploads(filenames: List[str], upload_dir: Path) -> None:
    """Persist the latest uploaded filenames."""
    upload_dir.mkdir(parents=True, exist_ok=True)
    payload = {"files": filenames}
    (upload_dir / LATEST_UPLOADS_FILE).write_text(json.dumps(payload), encoding="utf-8")


def get_latest_uploads(upload_dir: Path) -> List[Path]:
    """Return latest uploaded file paths (if present and still on disk)."""
    state_path = upload_dir / LATEST_UPLOADS_FILE
    if not state_path.exists():
        return []

    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    filenames = payload.get("files", [])
    paths = []
    for name in filenames:
        candidate = upload_dir / name
        if candidate.exists():
            paths.append(candidate)

    return paths
