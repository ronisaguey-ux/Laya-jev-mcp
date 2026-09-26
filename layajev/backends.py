"""Backends: where the probabilities actually come from.

Two implementations of one contract:

    predict(state, questions) -> {"answers": {<name>: <answer>}, ...}

`LayaBackend` runs the model on this machine. It is free and private, and it is
slower to start because the checkpoint has to load.

`JevBackend` posts to a remote TypeSafe-compatible endpoint (`/v1/systemone`).
It needs an API key and costs money per call, but it needs no local weights.

Both are normalised into the same answer shape, so everything above this module
is written once. Where the two genuinely disagree the fields are kept separate
rather than averaged — a blended number nobody can audit is worse than two
honest ones.

`auto_backend()` picks Laya when it is importable and Jev when it is configured,
and prefers Laya because a local call costs nothing and does not send the
caller's state to a third party.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from typing import Any, Dict, List, Mapping, Optional

from .questions import validate_questions

# TypeSafe's documented endpoint. OpenRouter mirrors the same product under a
# `typesafe/` model namespace and accepts these calls at its own base URL, which
# is how a caller who already has an OpenRouter key can use Jev with no new
# credential.
JEV_DEFAULT_BASE = "https://api.typesafe.ai/v1"
JEV_DEFAULT_MODEL = "jev-latest"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

_API_KEY_FILES = ("~/.config/typesafe/key", "~/.config/jev/key")


class BackendError(RuntimeError):
    """A backend could not answer. Never swallowed into an empty result — the
    caller has to be able to tell "the model said no" from "the model never ran"."""


class Backend:
    """The contract everything above this module is written against."""

    name = "backend"

    def available(self) -> bool:
        raise NotImplementedError

    def predict(self, state: Any, questions: Mapping[str, Any], **kw: Any) -> Dict[str, Any]:
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        """What this backend is and whether it can run — for the `status` tool."""
        return {"name": self.name, "available": self.available()}


# ─────────────────────────────────────────────────────────────────────────────
# Laya — local, free, no key
# ─────────────────────────────────────────────────────────────────────────────

class LayaBackend(Backend):
    """The local model. Imported lazily so this package installs without torch."""

    name = "laya"

    _runner: Any = None
    _load_error: Optional[str] = None
    _lock = threading.Lock()

    def __init__(self, model: Optional[str] = None, device: Optional[str] = None):
        self.model = model or os.environ.get("LAYA_MODEL") or ""
        self.device = device or os.environ.get("LAYA_DEVICE") or ""

    def available(self) -> bool:
        try:
            import laya  # noqa: F401
            return True
        except Exception as e:  # pragma: no cover - depends on the install
            self._load_error = f"laya is not importable: {e}"
            return False

    def _load(self) -> Any:
        # One load per process. The checkpoint is hundreds of MB; loading it per
        # call would make the tool useless.
        with LayaBackend._lock:
            if LayaBackend._runner is not None:
                return LayaBackend._runner
            try:
                import laya
            except Exception as e:
                raise BackendError(
                    "the Laya backend needs the `laya` package — install it with "
                    "`pip install layajev-mcp[laya]`, or point this at Jev instead"
                ) from e
            kw: Dict[str, Any] = {}
            if self.model:
                kw["model"] = self.model
            if self.device:
                kw["device"] = self.device
            try:
                LayaBackend._runner = laya.load(**kw)
            except TypeError:
                # A different laya version without those kwargs: load with defaults
                # rather than failing a call over an optional knob.
                LayaBackend._runner = laya.load()
            except Exception as e:
                raise BackendError(f"could not load the Laya checkpoint: {e}") from e
            return LayaBackend._runner

    def predict(self, state: Any, questions: Mapping[str, Any], **kw: Any) -> Dict[str, Any]:
        # Validate BEFORE loading. The checkpoint is hundreds of MB and takes
        # seconds; a caller whose question is malformed should be told that
        # immediately, not after paying for a model load to learn it.
        qs = validate_questions(questions)
        runner = self._load()
        try:
            out = runner.predict(state, qs)
        except Exception as e:
            raise BackendError(f"Laya predict failed: {e}") from e
        if not isinstance(out, Mapping):
            raise BackendError(f"Laya returned {type(out).__name__}, expected an object")
        result = dict(out)
        result.setdefault("model", self.model or "laya")
        return result

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d["kind"] = "local"
        d["cost"] = "free"
        d["note"] = "runs on this machine; the state never leaves it"
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Jev — remote, key-gated, billed
# ─────────────────────────────────────────────────────────────────────────────

def _load_key_from_file() -> Optional[str]:
    for tmpl in _API_KEY_FILES:
        p = os.path.expanduser(tmpl)
        try:
            with open(p, "r", encoding="utf-8") as fh:
                key = fh.read().strip()
            if key:
                return key
        except OSError:
            continue
    return None


class JevBackend(Backend):
    """A remote TypeSafe-compatible `/v1/systemone` endpoint."""

    name = "jev"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY") or os.environ.get("JEV_API_KEY")
        if not self.api_key:
            self.api_key = _load_key_from_file()
        # An OpenRouter base + key is a first-class way to reach the same product,
        # so honour it instead of insisting on a dedicated TypeSafe credential.
        self.base_url = (
            base_url
            or os.environ.get("JEV_BASE_URL")
            or os.environ.get("TYPESAFE_BASE_URL")
            or (OPENROUTER_BASE if os.environ.get("OPENROUTER_API_KEY") and not self.api_key else JEV_DEFAULT_BASE)
        )
        if not self.api_key and os.environ.get("OPENROUTER_API_KEY") and "openrouter" in self.base_url:
            self.api_key = os.environ["OPENROUTER_API_KEY"]
        self.model = model or os.environ.get("JEV_MODEL") or JEV_DEFAULT_MODEL
        self.timeout = timeout

    def available(self) -> bool:
        return bool(self.api_key and self.base_url)

    def predict(self, state: Any, questions: Mapping[str, Any], **kw: Any) -> Dict[str, Any]:
        # Same order as the local backend: a malformed question must not cost a
        # network round trip to discover.
        qs = validate_questions(questions)
        if not self.available():
            raise BackendError(
                "the Jev backend needs an API key — set TYPESAFE_API_KEY (or OPENROUTER_API_KEY "
                "with an openrouter.ai base URL), or write one to ~/.config/typesafe/key"
            )
        # The engine reads the question ids as opaque keys, so a Name is optional —
        # but a stable key is what makes the response map back to the caller's names.
        payload = {"model": self.model, "state": state, "questions": qs}
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url.rstrip("/") + "/systemone",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace")[:400]
            except Exception:
                pass
            # A message that says what the server said is worth more than "HTTP 401".
            raise BackendError(f"Jev endpoint returned HTTP {e.code}: {detail or e.reason}") from e
        except urllib.error.URLError as e:
            raise BackendError(f"could not reach the Jev endpoint at {self.base_url}: {e.reason}") from e
        except Exception as e:
            raise BackendError(f"Jev request failed: {e}") from e

        try:
            out = json.loads(raw)
        except json.JSONDecodeError as e:
            raise BackendError(f"Jev returned non-JSON: {raw[:200]!r}") from e
        if not isinstance(out, Mapping):
            raise BackendError(f"Jev returned {type(out).__name__}, expected an object")
        result = dict(out)
        result.setdefault("model", self.model)
        return result

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d["kind"] = "remote"
        d["cost"] = "billed per call"
        d["baseUrl"] = self.base_url
        d["model"] = self.model
        d["note"] = "the state is sent to a third party; check that is acceptable before using it"
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Selection
# ─────────────────────────────────────────────────────────────────────────────

def available_backends() -> List[Dict[str, Any]]:
    """Every backend this install could use, and whether each one can run now."""
    return [LayaBackend().describe(), JevBackend().describe()]


def auto_backend(prefer: Optional[str] = None) -> Backend:
    """Pick a backend, or explain why none can run.

    Preference order is Laya, then Jev. Laya first because a local call costs
    nothing and does not hand the caller's state to anybody — the cheaper and
    more private option is the right default, not merely the cheaper one.

    `prefer` pins a choice; an explicitly requested backend that cannot run is an
    error rather than a silent fallback, because quietly answering with a
    different model than the caller asked for is how a wrong answer looks right.
    """
    laya_b = LayaBackend()
    jev_b = JevBackend()

    if prefer:
        wanted = prefer.strip().lower()
        if wanted == "laya":
            if not laya_b.available():
                raise BackendError("laya was requested but is not installed in this environment")
            return laya_b
        if wanted in ("jev", "typesafe"):
            if not jev_b.available():
                raise BackendError("jev was requested but no API key is configured")
            return jev_b
        raise BackendError(f"unknown backend {prefer!r} — use 'laya', 'jev', or omit it")

    if laya_b.available():
        return laya_b
    if jev_b.available():
        return jev_b
    raise BackendError(
        "no decision backend is available: Laya is not installed and no Jev API key is set. "
        "Install with `pip install layajev-mcp[laya]`, or set TYPESAFE_API_KEY."
    )
