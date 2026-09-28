"""Which model should do this work — the layer neither Laya nor Jev provides.

`backends.py` answers questions. `decide.py` turns an answer into a gate. This module
answers a different question entirely: given a task, which model should run it?

The distinction matters because the two decisions have different inputs and different
failure modes. A decision problem has a fixed option set the caller wrote down. A
routing problem has an option set the ROUTER owns, an external constraint (the model
must be able to perceive the input at all), and an objective that is not accuracy —
it is "cheapest option still good enough", where "good enough" is a bar the caller
may move.

Two properties are load-bearing and are enforced here rather than described:

* **Capability is a hard filter, applied before cost.** A model that cannot accept
  audio is not a cheap option, it is not an option. There is no budget at which it
  becomes correct, so it must never appear in the shortlist.
* **A ceiling is a ceiling.** If the caller asks for the cheapest band and nothing in
  it can do the job, the right answer is to say so and name what would work — not to
  quietly spend past the ceiling the caller chose.

Both come from `laya_router`, which is imported by path so this package keeps its own
dependency list. If the router is not installed, the tools report that plainly; they do
not fall back to guessing, because a routing answer with no measurement behind it is
worse than no answer.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

_ROUTER_DIRS = [
    os.environ.get("LAYA_ROUTER_DIR"),
    "/home/roni/Roni_workspace/laya_router",
]

_router = None
_import_error: Optional[str] = None


def _load_router():
    """Import `router.py` from whichever directory provides it, once.

    Imported lazily and cached: the module reads a 1,900-row dataset to build its
    task index, which is real work that a caller who only wants `backend_status`
    should not pay for.
    """
    global _router, _import_error
    if _router is not None or _import_error is not None:
        return _router
    for d in _ROUTER_DIRS:
        if not d:
            continue
        path = Path(d) / "router.py"
        if not path.exists():
            continue
        try:
            spec = importlib.util.spec_from_file_location("laya_router_router", path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["laya_router_router"] = mod
            if d not in sys.path:
                sys.path.insert(0, d)      # router.py imports bands/modalities as siblings
            spec.loader.exec_module(mod)
            _router = mod
            return _router
        except Exception as e:                                  # noqa: BLE001
            _import_error = f"{type(e).__name__}: {e}"
    if _import_error is None:
        _import_error = (f"router.py not found in any of: {[d for d in _ROUTER_DIRS if d]}")
    return None


def router_status() -> Dict[str, Any]:
    """Whether routing is available here, and why not when it is not."""
    r = _load_router()
    if r is None:
        return {"available": False, "reason": _import_error,
                "searched": [d for d in _ROUTER_DIRS if d]}
    try:
        import bands
        return {"available": True, "repo": str(r.Path(r.__file__).resolve().parent),
                "bands": {b: len(bands.candidates(b)) for b in ("low", "medium", "high")},
                "adapter": bool(getattr(r, "DEFAULT_ADAPTER", None))}
    except Exception as e:                                      # noqa: BLE001
        return {"available": False, "reason": f"{type(e).__name__}: {e}"}


def _pick_dict(p) -> Dict[str, Any]:
    return {
        "model": p.model,
        "band": p.band,
        "cost_usd": None if p.cost_usd == float("inf") else p.cost_usd,
        "confidence": p.confidence,
        "source": p.source,
        "note": p.note,
        "ranked": [{"model": m, "score": round(s, 4)} for m, s in p.ranked[:8]],
    }


def pick_model(task: str, band: str = "auto",
               capabilities: Optional[List[str]] = None,
               spec: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The cheapest model that is still good enough for `task`.

    `spec` is the full task definition when the caller has one — a dict carrying the
    task's `check` block, the same shape the router's task files use. Pass it if you have
    it: the adapter is trained on a DIGEST derived from that spec, and `task` alone forces
    the router to recover a weaker digest from prose. The two are not equivalent, and the
    response says which one was used so a caller can tell them apart.

    Returns `model: None` with a `note` explaining why when nothing in the requested
    band can meet the capability — that is an answer, not an error, and the note names
    the cheapest band and model that would work.
    """
    r = _load_router()
    if r is None:
        return {"error": "router_unavailable", "reason": _import_error}
    if not isinstance(task, str) or not task.strip():
        return {"error": "empty_task",
                "reason": "task is the input the router reads; an empty one has no answer"}
    caps = frozenset(capabilities or ())
    known = {"text", "image", "audio", "video", "file"}
    unknown = caps - known
    if unknown:
        return {"error": "unknown_capability", "unknown": sorted(unknown),
                "known": sorted(known)}
    router = r.Router()
    p = router.pick(task, band, caps or None, spec=spec)
    out = _pick_dict(p)
    # Report what the router actually read. A digest recovered from prose is weaker than
    # one taken from a spec, and the caller is the only one who can supply the spec — so
    # say which happened rather than letting both paths look identical.
    out["state"] = getattr(router, "_last_state", None)
    out["state_from"] = "spec" if spec else "text"
    return out


def pick_many(tasks: List[str], band: str = "auto",
              specs: Optional[List[Optional[Dict[str, Any]]]] = None) -> Dict[str, Any]:
    """Route several independent tasks in one call.

    Each task is routed on its own; batching is an I/O saving here, not a shared
    forward pass, because the option set and the target differ per task. Kept in the
    same shape as `decide_many` so a caller who knows that idiom is not surprised.

    `specs` is optional and positional: `specs[i]` is the full task definition for
    `tasks[i]`, or None to fall back to prose for that entry. ★ WITHOUT THIS, THE ONE
    PATH THAT ROUTES MANY TASKS COULD NOT USE THE BETTER INPUT — `pick_model` gained a
    `spec` because the adapter is trained on a digest derived from a task's definition,
    and routing from prose alone yields a weaker state. A caller holding real task
    definitions (the farm's tasks have a `check` block) had to drop them to batch.
    """
    if not isinstance(tasks, list) or not tasks:
        return {"error": "empty_tasks", "reason": "pass a non-empty list of task strings"}
    if specs is not None and len(specs) != len(tasks):
        return {"error": "specs_length_mismatch",
                "reason": f"got {len(specs)} specs for {len(tasks)} tasks"}
    out = []
    for i, t in enumerate(tasks):
        spec = specs[i] if specs is not None else None
        out.append({"task": t[:120], **pick_model(t, band, spec=spec)})
    return {"n": len(out), "band": band, "results": out}
