"""Serve the trained per-leaf adapters from ONE loaded base model.

The tree walk held no adapter at all: `tree_walk.py` calls `layajev.decide.decide`, which
goes through the base Laya model, so every adapter trained here was orphaned on disk. This
is the piece that connects them.

Two constraints shape it:

1. The base model is 425.7M parameters and takes ~75s to load. Loading one per leaf is not
   an option across 54 leaves, so the base is loaded ONCE and adapters are attached and
   detached around each call. Attaching a LoRA is cheap - measured 0.6s to swap.

2. Laya is a NARROW-task model. One adapter per leaf is the shape Bob specified: "remember
   laya is amazing at narrow tasks, but if u make it general it completely fails". So the
   adapter is chosen BY LEAF, never one pooled adapter over the tree.

An adapter that is missing is not an error - most leaves have no adapter yet. The caller
falls through to the base model and the result says which was used, because a silent
fallback would make a base-model answer look like a fine-tuned one.
"""
from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import Any, Dict, List, Mapping, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
ADAPTERS = os.path.join(_HERE, "adapters")

_LOCK = threading.Lock()
_AGENT = None
_ATTACHED: Optional[str] = None


def adapter_dir(leaf_id: str) -> str:
    return os.path.join(ADAPTERS, f"t3_{leaf_id}")


def has_adapter(leaf_id: str) -> bool:
    """A directory alone is not an adapter; the weights file has to be there too."""
    d = adapter_dir(leaf_id)
    return os.path.isfile(os.path.join(d, "adapter_model.safetensors"))


def available_leaves() -> List[str]:
    if not os.path.isdir(ADAPTERS):
        return []
    out = []
    for name in sorted(os.listdir(ADAPTERS)):
        if not name.startswith("t3_"):
            continue
        leaf_id = name[len("t3_"):]
        if has_adapter(leaf_id):
            out.append(leaf_id)
    return out


def agent():
    """The one base model, loaded on first use."""
    global _AGENT
    if _AGENT is None:
        import laya
        _AGENT = laya.load()
    return _AGENT


@contextmanager
def use_adapter(leaf_id: Optional[str]):
    """Attach `leaf_id`'s adapter for the duration of the block, then remove it.

    Yields the leaf id actually in force, or None when there is no adapter and the base
    model is being used. The caller is expected to report that distinction.
    """
    global _ATTACHED
    if not leaf_id or not has_adapter(leaf_id):
        yield None
        return

    from peft import PeftModel
    with _LOCK:
        a = agent()
        model = a.model
        if _ATTACHED is not None and _ATTACHED != leaf_id:
            # Detach the previous leaf before attaching the next. Leaving two LoRA
            # wrappers stacked makes the second one score through the first.
            if hasattr(model, "unload"):
                try:
                    model.unload()
                except Exception:
                    pass
            _ATTACHED = None
        if _ATTACHED is None:
            wrapped = PeftModel.from_pretrained(model, adapter_dir(leaf_id))
            wrapped.eval()
            _ATTACHED = leaf_id
        try:
            yield _ATTACHED
        finally:
            pass


def decide_leaf(state: Any, leaf: Mapping[str, Any], opts: List[Dict[str, str]],
                policy: Any, caller: Optional[str] = None) -> Dict[str, Any]:
    """One leaf decision, through its own adapter when one exists.

    Falls back to the base model and SAYS SO in `adapter_used`, so a base answer is never
    mistaken for a fine-tuned one.
    """
    from layajev.decide import decide

    leaf_id = leaf.get("id")
    with use_adapter(leaf_id) as used:
        res = decide(state, opts, leaf["question"], policy=policy, context_label="state")
    res = dict(res)
    res["adapter_used"] = used
    res["leaf"] = leaf_id
    if used is None:
        res.setdefault("notes", [])
        if isinstance(res["notes"], list):
            res["notes"].append(
                f"no adapter for {leaf_id}; answered by the base model")
    return res


def status() -> Dict[str, Any]:
    return {"attached": _ATTACHED, "base_loaded": _AGENT is not None,
            "adapters": available_leaves()}
