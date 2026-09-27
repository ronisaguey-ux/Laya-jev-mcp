"""MCP stdio server: the decision tools, exposed to any MCP client.

Speaks MCP over stdio, one JSON-RPC message per line. Hand-rolled rather than
depending on the MCP SDK, for one reason that matters here: this server has to
run in the same environment as a locally-loaded model without dragging a
dependency tree along with it. The protocol surface it needs is small and stable.

Tools:

  decide           one situation and a list of options -> probabilities + a gate
  decide_many      several questions about ONE state -> one pass, all answers
  triage_options   score a long list of options, then decide over the best few
  judge            a yes/no with a probability, for the asymmetric-cost case
  backend_status   which engines can run here, and why not

Contract notes (MCP 2025-06-18):
  - each tool declares an `outputSchema`, so results also carry a
    `structuredContent` object that conforms to it;
  - the same JSON is ALSO serialised into a text block, because pre-2025-06-18
    clients ignore `structuredContent` and would otherwise see nothing;
  - `isError` is set on a tool-level failure, with the reason in the text, so the
    caller can tell a refusal from an empty result.

Every tool is read-only and touches nothing outside this process's own model, so
they are all annotated accordingly — a client should not have to ask permission
to obtain a probability.
"""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any, Callable, Dict, List, Mapping, Optional

from . import __version__
from .backends import BackendError, JevBackend, LayaBackend, auto_backend, available_backends
from .decide import DecisionPolicy, decide, decide_many
from .questions import QuestionError
from .routing import pick_model, pick_many, router_status

SERVER_NAME = "layajev"
# The revision this server implements. A client that speaks a newer one still
# works — nothing here depends on a newer-only feature.
PROTOCOL_VERSION = "2025-06-18"
DEFAULT_PROTOCOL = "2024-11-05"

# Refused rather than truncated, so a caller cannot accidentally hand a whole file
# to the model and wonder why the decision is about something else.
MAX_STATE_CHARS = 32_000


# ── output schemas ───────────────────────────────────────────────────────────
# Declared per tool because declaring one creates a hard obligation to conform —
# the schema is the contract, and these are built to match what the code returns.

_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["act", "escalate"],
                     "description": "act when the winner cleared the gate; escalate when it did not"},
        "reason": {"type": "string", "description": "why the gate decided that way"},
        "recommendation": {"type": ["string", "null"], "description": "the leading option"},
        "confidence": {"type": ["number", "null"], "description": "probability on the recommendation"},
        "margin": {"type": ["number", "null"], "description": "lead over the runner-up"},
        "runner_up": {"type": "string"},
        "probabilities": {"type": "object", "additionalProperties": {"type": "number"},
                          "description": "label -> probability, every option the caller offered"},
        "backend": {"type": "string"},
        "model": {"type": ["string", "null"]},
    },
    "required": ["decision", "reason", "probabilities", "backend"],
}

_MANY_SCHEMA = {
    "type": "object",
    "properties": {
        "answers": {"type": "object", "description": "question name -> answer"},
        "backend": {"type": "string"},
        "model": {"type": ["string", "null"]},
        "usage": {"type": "object"},
    },
    "required": ["answers", "backend"],
}

_STATUS_SCHEMA = {
    "type": "object",
    "properties": {
        "version": {"type": "string"},
        "default": {"type": ["string", "null"], "description": "the backend used when none is named"},
        "backends": {"type": "array", "items": {"type": "object"}},
    },
    "required": ["version", "backends"],
}


def _text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QuestionError("must be non-empty text")
    return value.strip()


def _state_arg(raw: Any) -> Any:
    """Accept a JSON object/array, or a string, and refuse an oversized one."""
    if raw is None:
        raise QuestionError("`state` is required")
    if isinstance(raw, str):
        if len(raw) > MAX_STATE_CHARS:
            raise QuestionError(
                f"`state` is {len(raw)} characters, over the {MAX_STATE_CHARS} limit — "
                "pass the relevant part instead of the whole document"
            )
        return raw
    if not isinstance(raw, (Mapping, list, tuple, int, float, bool)):
        raise QuestionError("`state` must be a string, object, array or scalar")
    if isinstance(raw, Mapping):
        import json as _j
        if len(_j.dumps(raw)) > MAX_STATE_CHARS:
            raise QuestionError(
                f"`state` is larger than the {MAX_STATE_CHARS} character limit — "
                "pass the relevant part instead of the whole document"
            )
    return raw


def _policy(args: Mapping[str, Any]) -> DecisionPolicy:
    pol = DecisionPolicy()
    for key in ("act_at", "margin", "noul_act_at"):
        if key in args and args[key] is not None:
            try:
                setattr(pol, key, float(args[key]))
            except (TypeError, ValueError):
                raise QuestionError(f"`{key}` must be a number, got {args[key]!r}")
    for key in ("act_at", "margin", "noul_act_at"):
        v = getattr(pol, key)
        if not 0.0 <= v <= 1.0:
            raise QuestionError(f"`{key}` is a probability and must be between 0 and 1, got {v}")
    return pol


# ── tools ────────────────────────────────────────────────────────────────────

def tool_decide(args: Mapping[str, Any]) -> Dict[str, Any]:
    state = _state_arg(args.get("state"))
    options = args.get("options")
    if not isinstance(options, list) or not options:
        raise QuestionError("`options` must be a non-empty list of labels or {label, description} objects")
    instruction = _text(args.get("instruction"))
    return decide(
        state,
        options,
        instruction,
        backend_name=args.get("backend"),
        policy=_policy(args),
        context_label=args.get("state_label") or "state",
    )


def tool_decide_many(args: Mapping[str, Any]) -> Dict[str, Any]:
    state = _state_arg(args.get("state"))
    questions = args.get("questions")
    if not isinstance(questions, Mapping) or not questions:
        raise QuestionError("`questions` must be a non-empty object keyed by question name")
    return decide_many(
        state,
        questions,
        backend_name=args.get("backend"),
        context_label=args.get("state_label") or "state",
    )


def tool_triage_options(args: Mapping[str, Any]) -> Dict[str, Any]:
    """Narrow a long list of options, then decide over the survivors.

    Choice questions lose accuracy past roughly twenty options, so a caller with
    fifty candidates gets a bad answer rather than an error. This scores each
    candidate on a scale first, keeps the strongest few, and runs the real
    decision over those — two passes, both cheap, and the shortlist is reported
    so the caller can see what was discarded rather than trusting a silent cut.
    """
    state = _state_arg(args.get("state"))
    options = args.get("options")
    if not isinstance(options, list) or not options:
        raise QuestionError("`options` must be a non-empty list")
    instruction = _text(args.get("instruction"))
    keep = int(args.get("keep") or 5)
    if not 2 <= keep <= 20:
        raise QuestionError("`keep` must be between 2 and 20")

    labels: List[str] = []
    criteria: Dict[str, str] = {}
    for o in options:
        if isinstance(o, Mapping):
            lbl = str(o.get("label") or o.get("id") or "")
            if not lbl:
                raise QuestionError(f"an option needs a label or an id: {o!r}")
            criteria[lbl] = str(o.get("description") or lbl)
        else:
            lbl = str(o)
            criteria[lbl] = lbl
        labels.append(lbl)
    if len(set(labels)) != len(labels):
        raise QuestionError(f"option labels must be unique, got {labels}")
    if len(labels) <= keep:
        # Nothing to narrow — deciding directly is the honest answer, and it avoids
        # a second model call that could only make things worse.
        return decide(state, labels, instruction, backend_name=args.get("backend"),
                      policy=_policy(args))

    from .questions import choice as _choice
    q = {"shortlist": _choice(instruction, criteria)}
    be = auto_backend(args.get("backend"))
    result = decide_many(state, q, backend=be, context_label=args.get("state_label") or "state")

    dist = (result["answers"].get("shortlist") or {}).get("probabilities") or {}
    ranked = sorted(labels, key=lambda l: float(dist.get(l, 0.0)), reverse=True)
    shortlist = ranked[:keep]

    out = decide(state, shortlist, instruction, backend=be, policy=_policy(args))
    out["shortlisted"] = shortlist
    out["discarded"] = [l for l in labels if l not in shortlist]
    out["stage1_probabilities"] = dist
    out["backend"] = result.get("backend") or out.get("backend")
    return out


def tool_judge(args: Mapping[str, Any]) -> Dict[str, Any]:
    """A yes/no with a probability, for a decision with asymmetric costs.

    The bar is the caller's to set and defaults lower than a multi-option
    decision, because the usual question here is a risk check: "might this
    destroy something?" should trip at a much lower probability than "is this the
    best of five?" would.
    """
    state = _state_arg(args.get("state"))
    question = _text(args.get("question"))
    from .questions import noul as _noul
    be = auto_backend(args.get("backend"))
    pol = _policy(args)
    if "noul_act_at" not in args and "act_at" in args:
        pol.noul_act_at = pol.act_at

    result = decide_many(
        state, {"q": _noul(question)}, backend=be,
        context_label=args.get("state_label") or "state",
    )
    answer = result["answers"].get("q") or {}
    try:
        p_true = float(answer.get("value"))
    except (TypeError, ValueError):
        raise BackendError(f"the backend returned no probability for the question: {answer}")

    verdict = pol.verdict_noul(p_true)
    return {
        "probability": p_true,
        "decision": verdict["decision"],
        "reason": verdict["reason"],
        "yes": p_true >= pol.noul_act_at,
        "threshold": pol.noul_act_at,
        "backend": result.get("backend"),
        "model": result.get("model"),
    }


def tool_pick_model(args: Mapping[str, Any]) -> Dict[str, Any]:
    """Cheapest model still good enough for one task."""
    return pick_model(args.get("task", ""), args.get("band", "auto"),
                      args.get("capabilities"))


def tool_pick_many(args: Mapping[str, Any]) -> Dict[str, Any]:
    """Route several independent tasks in one call."""
    return pick_many(args.get("tasks", []), args.get("band", "auto"))


def tool_router_status(_args: Mapping[str, Any]) -> Dict[str, Any]:
    """Whether routing is available, and why not when it is not."""
    return router_status()


def tool_backend_status(_args: Mapping[str, Any]) -> Dict[str, Any]:
    backends = available_backends()
    default = None
    for b in backends:
        if b.get("available"):
            default = b.get("name")
            break
    return {"version": __version__, "default": default, "backends": backends}


# name -> (handler, definition)
def _tools() -> Dict[str, Dict[str, Any]]:
    return {
        "decide": {
            "handler": tool_decide,
            "description": (
                "Choose between several courses of action and get a gated decision. "
                "Use when the task is ambiguous: more than one option is plausible, or none "
                "clearly is. Returns probabilities over every option plus 'act' or 'escalate' — "
                "escalate means it is too close to call and a human or a stronger model should "
                "decide. Cheap: a local call costs nothing and no state leaves this machine."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "state": {"description": "The situation to decide on: a string, or an object of named facts."},
                    "options": {
                        "type": "array",
                        "description": "The candidate actions. Give a description whenever the label alone could be read two ways.",
                        "items": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "object", "properties": {
                                    "label": {"type": "string"},
                                    "description": {"type": "string"}},
                                    "required": ["label"]},
                            ]
                        },
                    },
                    "instruction": {"type": "string",
                                    "description": "What is being decided, phrased as a question about the state."},
                    "backend": {"type": "string", "enum": ["laya", "jev"],
                                "description": "Force an engine. Omit to use the default. An engine that cannot run is an error, never a silent substitute."},
                    "act_at": {"type": "number",
                               "description": "Probability the winner needs before acting. Default 0.65. Raise to make the agent ask more; lower to make it act more."},
                    "margin": {"type": "number",
                               "description": "Lead over the runner-up needed before acting. Default 0.15. This is what stops a 55/45 split being treated as a decision."},
                    "state_label": {"type": "string",
                                    "description": "Key to file a plain-string state under, when the questions refer to it by name. Default 'state'."},
                },
                "required": ["state", "options", "instruction"],
            },
            "outputSchema": _DECISION_SCHEMA,
        },
        "decide_many": {
            "handler": tool_decide_many,
            "description": (
                "Ask several independent questions about ONE state in a single pass. "
                "Question ids stay local, so N questions cost one model call, not N — this is "
                "the cheap way to get many judgments at once. Types: 'choice' (label + criteria "
                "map), 'score' (ordered list of 2-10 steps), 'noul' (a yes/no, returns P(true))."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "state": {"description": "The situation: a string, or an object of named facts."},
                    "questions": {
                        "type": "object",
                        "description": "question name -> {type, instructions, criteria}",
                        "additionalProperties": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "enum": ["choice", "score", "noul"]},
                                "instructions": {"type": "string"},
                                "criteria": {"description": "choice: label->description map; score: ordered list; noul: optional true/false descriptions"},
                            },
                            "required": ["type", "instructions"],
                        },
                    },
                    "backend": {"type": "string", "enum": ["laya", "jev"]},
                    "state_label": {"type": "string"},
                },
                "required": ["state", "questions"],
            },
            "outputSchema": _MANY_SCHEMA,
        },
        "triage_options": {
            "handler": tool_triage_options,
            "description": (
                "Narrow a LONG list of options and then decide over the survivors. Use when you "
                "have more than about twenty candidates: a choice question loses accuracy past "
                "that, so this scores them all, keeps the strongest few, and decides over those. "
                "The shortlist and what was discarded are both returned, so the cut is visible."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "state": {"description": "The situation: a string, or an object of named facts."},
                    "options": {"type": "array", "description": "All candidates (labels or {label, description}).",
                                "items": {"anyOf": [{"type": "string"}, {"type": "object"}]}},
                    "instruction": {"type": "string"},
                    "keep": {"type": "integer", "description": "How many to carry into the decision. Default 5, max 20."},
                    "backend": {"type": "string", "enum": ["laya", "jev"]},
                    "act_at": {"type": "number"},
                    "margin": {"type": "number"},
                },
                "required": ["state", "options", "instruction"],
            },
            "outputSchema": _DECISION_SCHEMA,
        },
        "judge": {
            "handler": tool_judge,
            "description": (
                "Ask one yes/no question and get P(true). Use for a risk check with asymmetric "
                "costs — 'might this destroy something?', 'is this irreversible?', 'does this "
                "violate the contract?' — where the rare yes matters more than the common no. "
                "Set the threshold yourself with noul_act_at; it defaults low (0.5)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "state": {"description": "The situation: a string, or an object of named facts."},
                    "question": {"type": "string", "description": "The yes/no question about the state."},
                    "backend": {"type": "string", "enum": ["laya", "jev"]},
                    "noul_act_at": {"type": "number",
                                    "description": "P(true) needed to answer yes. Default 0.5; lower is more cautious."},
                    "state_label": {"type": "string"},
                },
                "required": ["state", "question"],
            },
            "outputSchema": {
                "type": "object",
                "properties": {
                    "probability": {"type": "number"},
                    "decision": {"type": "string", "enum": ["act", "escalate"]},
                    "reason": {"type": "string"},
                    "yes": {"type": "boolean"},
                    "threshold": {"type": "number"},
                    "backend": {"type": "string"},
                },
                "required": ["probability", "decision", "yes", "backend"],
            },
        },
        "pick_model": {
            "handler": tool_pick_model,
            "description": (
                "Pick the CHEAPEST model that is still good enough for a task, from 14 "
                "measured models. Not 'best model' and not 'cheapest model': every option "
                "was scored 1-100 by hand on real academic work at this product's traffic "
                "mix, and the target is the cheapest one clearing the quality bar. Measured "
                "on this data, the most expensive model has the WORST quality and the "
                "goldilocks pick beats a fixed default by ~5 quality points at lower cost. "
                "`band` is a ceiling the caller chooses: a hard task on 'low' gets the best "
                "low model; a trivial task on 'high' gets the CHEAPEST high model; 'auto' "
                "has no ceiling. `capabilities` names a modality the task needs (image, "
                "audio, video, file) and is a HARD filter applied before cost — a model "
                "that cannot perceive the input is not a cheap option. If nothing in the "
                "band can do it, the answer names the cheapest band that would."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task": {"type": "string",
                             "description": "The task text. This is what the router reads, so give the real prompt."},
                    "band": {"type": "string", "enum": ["auto", "low", "medium", "high"],
                             "description": "Cost ceiling. Default 'auto' (no ceiling)."},
                    "capabilities": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Modalities the task requires: image, audio, video, file. Omit for text.",
                    },
                },
                "required": ["task"],
            },
        },
        "pick_many": {
            "handler": tool_pick_many,
            "description": (
                "Route several independent tasks in one call. Each task is routed on its "
                "own — the option set and the right answer differ per task, so this is an "
                "I/O saving rather than a shared forward pass. Use it to plan a batch of "
                "work before spending anything on it."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "tasks": {"type": "array", "items": {"type": "string"},
                              "description": "Task texts to route."},
                    "band": {"type": "string", "enum": ["auto", "low", "medium", "high"]},
                },
                "required": ["tasks"],
            },
        },
        "router_status": {
            "handler": tool_router_status,
            "description": (
                "Whether the model router is available in this environment, which bands "
                "have which models, and the reason when it is NOT available. Call this "
                "before relying on pick_model in a new environment — the router is a "
                "separate component and its absence is reported honestly rather than "
                "answered by guessing."
            ),
            "inputSchema": {"type": "object", "properties": {}},
        },
        "backend_status": {
            "handler": tool_backend_status,
            "description": (
                "Which decision engines can run here and why not, if one cannot. Call this first "
                "when a decision tool fails: it separates 'the model declined' from 'no engine is "
                "installed or configured', which need different fixes."
            ),
            "inputSchema": {"type": "object", "properties": {}},
            "outputSchema": _STATUS_SCHEMA,
        },
    }


def list_tool_definitions() -> List[Dict[str, Any]]:
    return [
        {
            "name": name,
            "description": spec["description"],
            "inputSchema": spec["inputSchema"],
            # Optional: the decision tools declare a typed result, the routing tools
            # return a computed answer whose shape depends on whether a band could meet
            # the capability, so they have no single schema worth publishing. Indexing
            # it unconditionally made tools/list fail with a KeyError that surfaced as
            # "internal error" — one missing key breaking the entire tool list.
            **({"outputSchema": spec["outputSchema"]} if "outputSchema" in spec else {}),
            "annotations": {
                # Read-only and idempotent: these compute a probability and touch
                # nothing, so a client should not need to prompt for permission.
                "title": name.replace("_", " ").title(),
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": bool(name in ("decide", "decide_many", "triage_options", "judge")),
            },
        }
        for name, spec in _tools().items()
    ]


# ── JSON-RPC plumbing ────────────────────────────────────────────────────────

def _result(msg_id: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _error(msg_id: Any, code: int, message: str, data: Any = None) -> Dict[str, Any]:
    err: Dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": msg_id, "error": err}


def _tool_result(payload: Any, *, is_error: bool = False) -> Dict[str, Any]:
    """A tool result carrying both channels.

    `structuredContent` is what a typed client reads; the text block is the same
    JSON serialised, because a client that predates structured output would
    otherwise receive nothing at all.
    """
    text = json.dumps(payload, indent=2, default=str)
    out: Dict[str, Any] = {"content": [{"type": "text", "text": text}]}
    if isinstance(payload, Mapping) and not is_error:
        # Only claims conformance when it has an object to conform with.
        out["structuredContent"] = payload
    if is_error:
        out["isError"] = True
    return out


def handle_message(msg: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """One JSON-RPC message in, one out. None for a notification."""
    if not isinstance(msg, Mapping):
        return _error(None, -32600, "invalid request: not an object")

    method = msg.get("method")
    msg_id = msg.get("id")
    is_notification = msg_id is None

    if method == "initialize":
        params = msg.get("params") or {}
        client_version = params.get("protocolVersion") or DEFAULT_PROTOCOL
        if is_notification:
            return None
        return _result(msg_id, {
            "protocolVersion": client_version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": __version__},
            "instructions": (
                "Calibrated probabilities for an agent's own options. Reach for `decide` when a "
                "task is ambiguous — several actions plausible, or none clearly right — and act "
                "on the answer when it says 'act'. When it says 'escalate' the call is too close "
                "to make, so ask a human or a stronger model instead. `decide_many` answers many "
                "questions about one state in a single pass; `judge` answers a yes/no for a risk "
                "check; `triage_options` narrows a long candidate list."
            ),
        })

    if is_notification:
        # notifications/initialized and friends: nothing to answer.
        return None

    if method == "ping":
        return _result(msg_id, {})

    if method == "tools/list":
        return _result(msg_id, {"tools": list_tool_definitions()})

    if method == "tools/call":
        params = msg.get("params") or {}
        name = params.get("name")
        args = params.get("arguments") or {}
        spec = _tools().get(name)
        if spec is None:
            # An unknown tool is a protocol-level error, not a tool result — the
            # caller named something that does not exist.
            return _error(msg_id, -32602, f"unknown tool: {name!r}",
                          {"available": list(_tools())})
        if not isinstance(args, Mapping):
            return _result(msg_id, _tool_result(
                {"error": "arguments must be an object"}, is_error=True))
        try:
            payload = spec["handler"](args)
        except (QuestionError, BackendError) as e:
            # Expected failure: reported as a tool result so the model sees the
            # reason and can correct itself, rather than as a transport error it
            # cannot act on.
            return _result(msg_id, _tool_result({"error": str(e), "kind": type(e).__name__},
                                                is_error=True))
        except Exception as e:  # pragma: no cover - a bug in this server
            print(f"[{SERVER_NAME}] {name} failed: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return _result(msg_id, _tool_result(
                {"error": f"{name} failed unexpectedly: {e}", "kind": "InternalError"},
                is_error=True))
        return _result(msg_id, _tool_result(payload))

    if method in ("resources/list", "prompts/list"):
        return _result(msg_id, {} if method == "resources/list" else {"prompts": []})

    return _error(msg_id, -32601, f"method not found: {method}")


def serve(stdin: Any = None, stdout: Any = None) -> None:
    """Read newline-delimited JSON-RPC from stdin, answer on stdout.

    stdout is the wire, so nothing else may ever be written to it — one stray
    print corrupts the stream and the client reports a parse error that looks
    like a broken server. Diagnostics go to stderr.
    """
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            _write(stdout, _error(None, -32700, f"parse error: {e}"))
            continue
        try:
            out = handle_message(msg)
        except Exception as e:  # pragma: no cover
            traceback.print_exc(file=sys.stderr)
            out = _error(msg.get("id") if isinstance(msg, Mapping) else None,
                         -32603, f"internal error: {e}")
        if out is not None:
            _write(stdout, out)


def _write(stdout: Any, payload: Mapping[str, Any]) -> None:
    stdout.write(json.dumps(payload, default=str) + "\n")
    stdout.flush()


def main() -> None:
    try:
        serve()
    except (BrokenPipeError, KeyboardInterrupt):
        # The client went away. Nothing to report and nothing to clean up.
        pass


if __name__ == "__main__":
    main()
