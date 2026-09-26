"""`layajev` command line — the same decisions, without an MCP client.

    layajev decide --state ./situation.txt --option fix --option revert --instruction "..."
    layajev judge --state "..." --question "is this irreversible?"
    layajev status
    layajev serve          # the MCP server on stdio

Flags rather than a config file: a decision is usually a one-off, and making the
caller write a file to think about a choice would defeat the point.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, List, Optional

from . import __version__
from .backends import BackendError, available_backends
from .decide import DecisionPolicy, decide
from .mcp_server import serve as serve_mcp
from .questions import QuestionError


def _read_state(value: Optional[str], path: Optional[str]) -> Any:
    if path:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        # A JSON file is passed through as structure — the questions can then refer
        # to named fields instead of the whole blob being one string.
        if path.endswith(".json"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
        return text
    if not value:
        raise SystemExit("need a state: pass --state or --state-file")
    return value


def _policy(args: argparse.Namespace) -> DecisionPolicy:
    pol = DecisionPolicy()
    if args.act_at is not None:
        pol.act_at = args.act_at
    if args.margin is not None:
        pol.margin = args.margin
    if args.threshold is not None:
        pol.noul_act_at = args.threshold
    return pol


def cmd_decide(args: argparse.Namespace) -> int:
    options: List[Any] = []
    for raw in args.option or []:
        # "label::description" keeps a one-off decision a one-liner, while the
        # description is still available when the label is ambiguous.
        if "::" in raw:
            label, desc = raw.split("::", 1)
            options.append({"label": label.strip(), "description": desc.strip()})
        else:
            options.append(raw)
    if len(options) < 2:
        raise SystemExit("need at least 2 --option values to have a decision")

    try:
        result = decide(
            _read_state(args.state, args.state_file),
            options,
            args.instruction,
            backend_name=args.backend,
            policy=_policy(args),
            context_label=args.state_label,
        )
    except (QuestionError, BackendError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_human(result)
    # The gate is the output, so it is also the exit code: a caller driving this
    # from a script should not have to parse stdout to learn it must stop.
    return 0 if result.get("decision") == "act" else 3


def cmd_judge(args: argparse.Namespace) -> int:
    from .questions import noul
    from .backends import auto_backend
    from .decide import decide_many

    try:
        out = decide_many(
            _read_state(args.state, args.state_file),
            {"q": noul(args.question)},
            backend_name=args.backend,
            context_label=args.state_label,
        )
        answer = out["answers"].get("q") or {}
        p = float(answer.get("value"))
    except (QuestionError, BackendError, TypeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    threshold = args.threshold if args.threshold is not None else 0.5
    payload = {"probability": p, "yes": p >= threshold, "threshold": threshold,
               "backend": out.get("backend"), "model": out.get("model")}
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"P(true) = {p:.4f}   (threshold {threshold}) -> {'YES' if payload['yes'] else 'NO'}")
    return 0 if payload["yes"] else 3


def cmd_status(_args: argparse.Namespace) -> int:
    payload = {"version": __version__, "backends": available_backends()}
    if _args.json:
        print(json.dumps(payload, indent=2))
        return 0
    print(f"layajev {__version__}")
    for b in payload["backends"]:
        mark = "ready  " if b.get("available") else "unusable"
        print(f"  [{mark}] {b['name']:<6} {b.get('kind', ''):<7} {b.get('cost', '')}")
        if not b.get("available"):
            print(f"           {b.get('note', '') or 'not configured'}")
    return 0


def cmd_serve(_args: argparse.Namespace) -> int:
    serve_mcp()
    return 0


def _print_human(r: dict) -> None:
    print(f"{r['decision'].upper()}   {r.get('recommendation')}")
    print(f"  {r.get('reason')}")
    print()
    for label, p in sorted((r.get("probabilities") or {}).items(), key=lambda kv: -kv[1]):
        bar = "#" * max(1, int(round(p * 30)))
        print(f"  {p:6.3f}  {bar:<30} {label}")
    print()
    print(f"  backend: {r.get('backend')} ({r.get('model')})")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="layajev", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"layajev {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("decide", help="choose between options and gate the answer")
    d.add_argument("--state", help="the situation, inline")
    d.add_argument("--state-file", help="the situation, read from a file (.json is parsed)")
    d.add_argument("--option", action="append", help="a candidate; repeat. Use label::description for an explanation")
    d.add_argument("--instruction", required=True, help="what is being decided")
    d.add_argument("--backend", choices=["laya", "jev"], help="force an engine")
    d.add_argument("--act-at", type=float, dest="act_at", help="probability needed to act (default 0.65)")
    d.add_argument("--margin", type=float, help="lead over the runner-up needed to act (default 0.15)")
    d.add_argument("--threshold", type=float, help="alias for --act-at on the noul gate")
    d.add_argument("--state-label", dest="state_label", default="state")
    d.add_argument("--json", action="store_true", help="machine-readable output")
    d.set_defaults(func=cmd_decide)

    j = sub.add_parser("judge", help="ask one yes/no and get P(true)")
    j.add_argument("--state")
    j.add_argument("--state-file")
    j.add_argument("--question", required=True)
    j.add_argument("--backend", choices=["laya", "jev"])
    j.add_argument("--threshold", type=float, help="P(true) needed for YES (default 0.5)")
    j.add_argument("--state-label", dest="state_label", default="state")
    j.add_argument("--json", action="store_true")
    j.set_defaults(func=cmd_judge)

    s = sub.add_parser("status", help="which engines can run here")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_status)

    sv = sub.add_parser("serve", help="run the MCP server on stdio")
    sv.set_defaults(func=cmd_serve)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
