"""Harvest measured samples for the OCULUS decision taxonomy.

Design rules, each one paid for by a real failure:

* APPEND AS WE GO. Progress is written after every batch, so a crash or a rate
  limit costs one batch, never the run.
* NEVER OVERWRITE GOOD DATA WITH EMPTY DATA. A rate-limited or errored retry
  returns nothing; writing that over a good batch file is silent loss. A batch is
  only written when it has samples, and an existing non-empty batch is never
  replaced by an empty one.
* RESUME SAFELY. A batch already on disk with the right count is skipped, so a
  re-run continues instead of starting over.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
BATCH_DIR = os.path.join(HERE, "datasets", "batches")
PROGRESS = os.path.join(HERE, "harvest_progress.json")
BATCH_SIZE = 100


def _load_progress() -> Dict[str, Any]:
    try:
        with open(PROGRESS) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {"batches": {}, "started": None, "updated": None}


def _save_progress(p: Dict[str, Any]) -> None:
    p["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    tmp = PROGRESS + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(p, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, PROGRESS)


def batch_path(leaf_id: str, index: int) -> str:
    return os.path.join(BATCH_DIR, f"{leaf_id}.{index:04d}.json")


def write_batch(leaf_id: str, index: int, samples: List[Dict[str, Any]]) -> str:
    """Write one batch. Refuses to destroy a non-empty batch with an empty one."""
    os.makedirs(BATCH_DIR, exist_ok=True)
    path = batch_path(leaf_id, index)
    if not samples:
        raise ValueError(
            f"refusing to write an empty batch for {leaf_id} #{index} — "
            "an empty result is a failed attempt, not data"
        )
    if os.path.exists(path):
        try:
            with open(path) as fh:
                existing = json.load(fh)
        except (OSError, json.JSONDecodeError):
            existing = None  # unreadable file: a real write is an improvement
        if isinstance(existing, list) and len(samples) < len(existing):
            # NOT just the empty case. A rate-limited or half-failed retry returns a
            # PARTIAL batch, and replacing 100 good samples with 1 is the same silent
            # loss as replacing them with 0. Only a batch at least as large may land.
            raise ValueError(
                f"refusing to shrink {os.path.basename(path)} from {len(existing)} "
                f"samples to {len(samples)} — a partial batch is a failed attempt, not data"
            )
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(samples, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def already_done(leaf_id: str, index: int, expect: int = BATCH_SIZE) -> bool:
    path = batch_path(leaf_id, index)
    try:
        with open(path) as fh:
            return len(json.load(fh)) >= expect
    except (OSError, json.JSONDecodeError):
        return False


def progress_summary() -> Dict[str, Any]:
    p = _load_progress()
    n = sum(1 for v in p.get("batches", {}).values() if v.get("count"))
    total = sum(v.get("count", 0) for v in p.get("batches", {}).values())
    return {"batches_with_data": n, "samples": total}


if __name__ == "__main__":
    print(json.dumps({"progress": progress_summary(), "batch_dir": BATCH_DIR}, indent=2))
