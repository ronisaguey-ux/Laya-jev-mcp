"""Tests for layajev — no model, no network, no API key.

Both backends are faked at `predict()`, which is the boundary that matters: the
decision logic, the gate, the validation and the JSON-RPC surface are all
exercised for real, while the checkpoint and the remote endpoint are not needed.

The gate tests are the ones worth reading. A gate that only ever says "escalate"
passes every "is it cautious" test while being useless, so each threshold is
tested from BOTH sides.
"""

from __future__ import annotations

import json
import subprocess
import sys
from io import StringIO
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from layajev import (  # noqa: E402
    DecisionPolicy,
    QuestionError,
    decide,
    decide_many,
    choice,
    noul,
    score,
    validate_questions,
)
from layajev.backends import Backend, BackendError, available_backends, auto_backend  # noqa: E402
from layajev import mcp_server  # noqa: E402


class FakeBackend(Backend):
    """Returns a canned distribution, and records what it was asked."""

    name = "fake"

    def __init__(self, probs=None, noul_value=0.8, score_value=2, fail=None):
        self.probs = probs or {}
        self.noul_value = noul_value
        self.score_value = score_value
        self.fail = fail
        self.calls = []

    def available(self):
        return True

    def predict(self, state, questions, **kw):
        self.calls.append({"state": state, "questions": questions})
        if self.fail:
            raise BackendError(self.fail)
        answers = {}
        for name, spec in questions.items():
            if spec["type"] == "choice":
                labels = list(spec["criteria"])
                dist = {}
                for i, lbl in enumerate(labels):
                    if i < len(self.probs):
                        dist[lbl] = self.probs[i]
                answers[name] = {"type": "choice", "choice": labels[0],
                                 "probabilities": dist, "confidence": dist.get(labels[0], 0.0)}
            elif spec["type"] == "noul":
                answers[name] = {"type": "noul", "noul": self.noul_value,
                                 "confidence": self.noul_value}
            else:
                answers[name] = {"type": "score", "score": self.score_value,
                                 "probabilities": {str(i): 0.0 for i in range(len(spec["criteria"]))}}
        return {"model": "fake-1", "answers": answers, "usage": {"input_tokens": 5}}


# ── the gate: both sides of every threshold ──────────────────────────────────

def test_clear_winner_acts():
    be = FakeBackend(probs=[0.9, 0.07, 0.03])
    r = decide("s", ["a", "b", "c"], "which?", backend=be)
    assert r["decision"] == "act"
    assert r["recommendation"] == "a"
    assert r["confidence"] == pytest.approx(0.9)


def test_a_coin_flip_escalates_even_though_the_leader_wins():
    # 0.52 clears act_at=0.5 but not margin — this is the case a bare
    # "is the leader above X" check would wrongly call a decision.
    be = FakeBackend(probs=[0.52, 0.48])
    r = decide("s", ["a", "b"], "which?", backend=be,
               policy=DecisionPolicy(act_at=0.5, margin=0.15))
    assert r["decision"] == "escalate"
    assert "margin" in r["reason"]
    assert r["runner_up"] == "b"


def test_low_absolute_but_clear_lead_escalates_on_the_absolute_threshold():
    # 0.45 with the rivals far behind: a clear winner, but not confident in
    # absolute terms. Conservative by default, which is the point for an agent
    # that will act without asking.
    be = FakeBackend(probs=[0.45, 0.10, 0.05])
    r = decide("s", ["a", "b", "c"], "which?", backend=be)
    assert r["decision"] == "escalate"
    assert "below" in r["reason"]


def test_a_lower_threshold_makes_the_same_input_act():
    # Same numbers, different policy -> different verdict. Proves the threshold
    # is what decides, not some hidden constant.
    be = FakeBackend(probs=[0.52, 0.48])
    r = decide("s", ["a", "b"], "which?", backend=be,
               policy=DecisionPolicy(act_at=0.5, margin=0.01))
    assert r["decision"] == "act"


def test_every_offered_option_is_reported_even_when_unscored():
    # An option the engine did not mention must appear at 0.0 — a silently
    # missing option reads as "impossible" when it was never scored.
    be = FakeBackend(probs=[0.9])
    r = decide("s", ["a", "b", "c"], "which?", backend=be)
    assert set(r["probabilities"]) == {"a", "b", "c"}
    assert r["probabilities"]["c"] == 0.0


# ── judge / noul ─────────────────────────────────────────────────────────────

def test_noul_gate_acts_above_and_escalates_below():
    above = decide_many("s", {"q": noul("safe?")}, backend=FakeBackend(noul_value=0.9))
    assert above["answers"]["q"]["value"] == pytest.approx(0.9)

    be = FakeBackend(noul_value=0.9)
    assert DecisionPolicy(noul_act_at=0.5).verdict_noul(0.9)["decision"] == "act"
    assert DecisionPolicy(noul_act_at=0.5).verdict_noul(0.4)["decision"] == "escalate"
    # An asymmetric-cost check wants a low bar: 0.2 is enough to stop and look.
    assert DecisionPolicy(noul_act_at=0.15).verdict_noul(0.2)["decision"] == "act"


# ── question construction and validation ─────────────────────────────────────

def test_choice_accepts_labels_and_maps():
    assert choice("q", ["a", "b"])["criteria"] == {"a": "a", "b": "b"}
    m = choice("q", {"a": "first", "b": "second"})["criteria"]
    assert m == {"a": "first", "b": "second"}


def test_a_single_option_is_refused():
    # One option is not a decision, and letting it through would return a
    # confident 1.0 for a question with no alternative.
    with pytest.raises(QuestionError):
        choice("q", ["only"])


def test_too_many_options_is_refused_with_a_usable_message():
    with pytest.raises(QuestionError) as e:
        choice("q", [f"o{i}" for i in range(25)])
    assert "20" in str(e.value)


def test_score_needs_an_ordered_list_of_2_to_10():
    assert score("q", ["low", "high"])["criteria"] == ["low", "high"]
    with pytest.raises(QuestionError):
        score("q", ["only-one"])
    with pytest.raises(QuestionError):
        score("q", [f"s{i}" for i in range(11)])
    with pytest.raises(QuestionError):
        score("q", "low, high")  # a string is not a scale


def test_validation_rejects_a_blank_instruction_and_a_bad_type():
    with pytest.raises(QuestionError):
        validate_questions({"q": {"type": "choice", "instructions": "  ", "criteria": {"a": "x", "b": "y"}}})
    with pytest.raises(QuestionError):
        validate_questions({"q": {"type": "nonsense", "instructions": "x"}})
    with pytest.raises(QuestionError):
        validate_questions({})


def test_validation_strips_private_keys():
    out = validate_questions({"q": {"type": "score", "instructions": "x",
                                    "criteria": ["a", "b"], "_name": "leak"}})
    assert "_name" not in out["q"]


# ── decide_many: one pass for N questions ────────────────────────────────────

def test_decide_many_sends_every_question_in_one_call():
    be = FakeBackend(probs=[0.6, 0.4], noul_value=0.7)
    out = decide_many("s", {
        "a": choice("pick", ["x", "y"]),
        "b": noul("is it safe?"),
    }, backend=be)
    assert len(be.calls) == 1, "N questions must cost ONE call, not N"
    assert set(out["answers"]) == {"a", "b"}
    assert out["answers"]["b"]["value"] == pytest.approx(0.7)


def test_decide_many_reports_each_answer_with_its_kind():
    be = FakeBackend(probs=[0.6, 0.4])
    out = decide_many("s", {"a": choice("pick", ["x", "y"]),
                            "b": score("how bad?", ["mild", "bad"])}, backend=be)
    assert out["answers"]["a"]["kind"] == "choice"
    assert out["answers"]["b"]["kind"] == "score"


# ── failure is never an empty result ─────────────────────────────────────────

def test_a_backend_failure_raises_rather_than_returning_nothing():
    # An empty result would be indistinguishable from "the model abstained",
    # which is why this raises.
    with pytest.raises(BackendError):
        decide("s", ["a", "b"], "which?", backend=FakeBackend(fail="laya exploded"))


def test_a_response_with_no_distribution_raises():
    class NoDist(Backend):
        name = "nodist"
        def available(self): return True
        def predict(self, state, questions, **kw):
            return {"answers": {k: {"type": "choice", "confidence": 0.5} for k in questions}}

    with pytest.raises(BackendError):
        decide("s", ["a", "b"], "which?", backend=NoDist())


def test_a_bare_string_state_is_wrapped_so_questions_can_bind_to_it():
    be = FakeBackend(probs=[0.9, 0.1])
    decide("some text", ["a", "b"], "which?", backend=be)
    assert be.calls[0]["state"] == {"state": "some text"}
    # An object state is passed through untouched.
    be2 = FakeBackend(probs=[0.9, 0.1])
    decide({"k": "v"}, ["a", "b"], "which?", backend=be2)
    assert be2.calls[0]["state"] == {"k": "v"}


def test_naming_an_unavailable_backend_is_an_error_not_a_substitution():
    with pytest.raises(BackendError):
        auto_backend("nonsense")
    with pytest.raises(BackendError):
        auto_backend("jev")  # no key in the test environment


def test_available_backends_reports_both_with_a_reason():
    got = {b["name"]: b for b in available_backends()}
    assert set(got) == {"laya", "jev"}
    assert got["jev"]["kind"] == "remote"
    assert got["laya"]["kind"] == "local"


# ── the MCP surface ──────────────────────────────────────────────────────────

def test_every_tool_declaring_an_output_schema_is_listed_with_one():
    # The decision tools return a fixed result shape, so they declare an outputSchema
    # and must conform to it. The routing tools do not: their answer depends on whether
    # a band could meet the capability (a Pick with a model, or an empty one with a
    # note), so there is no single schema worth publishing. The invariant is therefore
    # conditional, not blanket: IF a tool declares one, it must be well-formed.
    defs = mcp_server.list_tool_definitions()
    names = {d["name"] for d in defs}
    assert {"decide", "decide_many", "triage_options", "judge",
            "pick_model", "pick_many", "router_status", "backend_status"} <= names
    for d in defs:
        assert d["inputSchema"]["type"] == "object"
        assert d["annotations"]["readOnlyHint"] is True
        if "outputSchema" in d:
            assert d["outputSchema"]["type"] == "object"
    for d in defs:
        if d["name"] in {"decide", "decide_many", "triage_options", "judge"}:
            assert d.get("outputSchema"), f"{d['name']} returns a fixed shape and must declare it"


def test_routing_tools_are_read_only_and_do_not_need_a_backend():
    # Routing is a lookup against measured data; it must work with NO key and NO model,
    # because that is the whole point of it being the cheap tier.
    got = mcp_server.tool_pick_model({"task": "Summarise this article in 3 sentences."})
    assert got.get("error") is None, got
    assert got["model"], got
    # an unknown capability is a caller error and must be named, not guessed at
    bad = mcp_server.tool_pick_model({"task": "x", "capabilities": ["smell"]})
    assert bad["error"] == "unknown_capability" and bad["unknown"] == ["smell"]


def test_pick_many_routes_each_task_independently():
    got = mcp_server.tool_pick_many(
        {"tasks": ["Explain photosynthesis in 90 words.",
                   "Read the bar chart in the attached image."]})
    assert got["n"] == 2
    assert all(r["model"] for r in got["results"])
    # a task with no text cannot be routed, and each row says so rather than returning
    # one shared answer — the defect this batched form replaced
    empty = mcp_server.tool_pick_model({"task": ""})
    assert empty["error"] == "empty_task"


def test_initialize_echoes_the_client_protocol_version():
    out = mcp_server.handle_message({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {}}})
    assert out["result"]["protocolVersion"] == "2025-06-18"
    assert out["result"]["serverInfo"]["name"] == "layajev"


def test_a_notification_gets_no_reply():
    assert mcp_server.handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_unknown_tool_is_a_protocol_error_and_unknown_method_too():
    err = mcp_server.handle_message({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                                     "params": {"name": "nope", "arguments": {}}})
    assert err["error"]["code"] == -32602
    assert "nope" in err["error"]["message"]

    err2 = mcp_server.handle_message({"jsonrpc": "2.0", "id": 3, "method": "does/not/exist"})
    assert err2["error"]["code"] == -32601


def test_a_question_error_is_reported_as_an_iserror_result_not_a_transport_error():
    # The model needs the reason so it can correct itself; a transport error
    # gives it nothing to act on.
    out = mcp_server.handle_message({
        "jsonrpc": "2.0", "id": 4, "method": "tools/call",
        "params": {"name": "decide", "arguments": {"state": "s", "options": ["only-one"],
                                                   "instruction": "which?"}}})
    assert "error" not in out, "a caller mistake is not a protocol failure"
    assert out["result"]["isError"] is True
    assert "at least 2 options" in out["result"]["content"][0]["text"]


def test_an_oversized_state_is_refused_by_name():
    big = "x" * (mcp_server.MAX_STATE_CHARS + 1)
    out = mcp_server.handle_message({
        "jsonrpc": "2.0", "id": 5, "method": "tools/call",
        "params": {"name": "decide", "arguments": {"state": big, "options": ["a", "b"],
                                                   "instruction": "which?"}}})
    assert out["result"]["isError"] is True
    assert "limit" in out["result"]["content"][0]["text"]


def test_a_tool_result_carries_both_channels():
    out = mcp_server._tool_result({"decision": "act", "n": 1})
    assert out["structuredContent"] == {"decision": "act", "n": 1}
    assert json.loads(out["content"][0]["text"]) == {"decision": "act", "n": 1}


def test_serve_answers_over_stdio_and_writes_nothing_else_to_stdout():
    # stdout is the wire: one stray print and the client sees a parse error.
    payload = "\n".join([
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {"protocolVersion": "2025-06-18"}}),
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
        json.dumps({"jsonrpc": "2.0", "id": 3, "method": "ping"}),
    ]) + "\n"

    stdin, stdout = StringIO(payload), StringIO()
    mcp_server.serve(stdin=stdin, stdout=stdout)

    lines = [l for l in stdout.getvalue().splitlines() if l.strip()]
    assert len(lines) == 3, f"expected 3 replies, got {len(lines)}"
    parsed = [json.loads(l) for l in lines]
    for p in parsed:
        assert p["jsonrpc"] == "2.0"
    assert parsed[1]["result"]["tools"]
    assert parsed[2]["result"] == {}


def test_malformed_json_gets_a_parse_error_not_a_crash():
    stdin, stdout = StringIO("{not json\n"), StringIO()
    mcp_server.serve(stdin=stdin, stdout=stdout)
    out = json.loads(stdout.getvalue().strip())
    assert out["error"]["code"] == -32700


def test_backend_status_works_with_no_model_and_no_key():
    out = mcp_server.tool_backend_status({})
    assert out["version"]
    assert {b["name"] for b in out["backends"]} == {"laya", "jev"}


# ── the CLI ──────────────────────────────────────────────────────────────────

def _cli(*args):
    return subprocess.run([sys.executable, "-m", "layajev.cli", *args],
                          capture_output=True, text=True, cwd=str(ROOT))


def test_cli_status_runs_without_a_model():
    r = _cli("status")
    assert r.returncode == 0, r.stderr
    assert "laya" in r.stdout and "jev" in r.stdout


def test_cli_decide_refuses_a_missing_state_or_one_option():
    r = _cli("decide", "--instruction", "which?")
    assert r.returncode != 0
    r2 = _cli("decide", "--state", "s", "--option", "only", "--instruction", "which?")
    assert r2.returncode != 0
    assert "at least 2" in (r2.stderr + r2.stdout)


def test_cli_reports_an_unavailable_backend_clearly():
    r = _cli("decide", "--state", "s", "--option", "a", "--option", "b",
             "--instruction", "which?", "--backend", "jev")
    assert r.returncode == 2
    assert "API key" in (r.stderr + r.stdout)


def test_gate_thresholds_are_probabilities():
    with pytest.raises(QuestionError):
        mcp_server._policy({"act_at": 1.5})
