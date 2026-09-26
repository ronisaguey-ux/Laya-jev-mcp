# layajev-mcp

**Calibrated probabilities for an agent's own options.**

An agent reaches a decision it cannot resolve from the information it has: several
courses of action are plausible, or none is clearly right. Asking a chat model is
the slow, expensive way to settle that — and it returns prose, so the caller
cannot tell a confident answer from a coin flip.

This is an MCP server that answers the question cheaply and returns **numbers**.
Give it a situation and a list of candidate actions; it comes back with a
probability for every option and a **gate**: `act` on the leader, or `escalate`
because the call is too close to make.

```jsonc
// decide(state="the test fails on a malformed email", options=["fix the email",
//        "drop the field", "refuse the sale"])
{
  "decision": "act",                       // or "escalate" — the useful half
  "recommendation": "drop the field",
  "confidence": 0.971,
  "margin": 0.943,
  "probabilities": {
    "fix the email": 0.028,
    "drop the field": 0.971,
    "refuse the sale": 0.001
  },
  "backend": "laya"
}
```

## Why a gate and not just the top option

An agent that always takes the argmax will act on a 51/49 split and never notice
it guessed. `escalate` gives an ambiguous situation somewhere to go that is not a
coin flip, and the thresholds are the caller's to set.

Two numbers are checked, because they catch different failures:

| | catches | default |
|---|---|---|
| `act_at` | the leader is not confident in absolute terms | `0.65` |
| `margin` | the leader is barely ahead of the runner-up | `0.15` |

A 0.55 leader over a 0.45 second option passes a bare `act_at` test while being a
coin flip; `margin` is what refuses it.

## Two engines, one idiom

| | **Laya** | **Jev (TypeSafe)** |
|---|---|---|
| runs | on this machine | remote HTTP |
| costs | nothing | billed per call |
| needs | `pip install layajev-mcp[laya]` | an API key |
| privacy | the state never leaves the box | the state is sent to a third party |
| speed | one local forward pass | network round trip |

`auto` prefers Laya: it is free, private, and fast enough. Naming an engine that
cannot run is an **error, never a silent substitution** — quietly answering with a
different model than the caller asked for is how a wrong answer looks right.

Jev is reached through `TYPESAFE_API_KEY`, or through an `OPENROUTER_API_KEY`
with an `openrouter.ai` base URL, since OpenRouter mirrors the same product and
that is a way to use Jev with no new credential.

## Install

```bash
# local engine (pulls torch + the checkpoints)
pip install "layajev-mcp[laya]"

# remote engine only — no local weights, tiny install
pip install layajev-mcp
export TYPESAFE_API_KEY=...
```

Wire it into any MCP client:

```json
{ "mcpServers": { "layajev": { "command": "layajev-mcp" } } }
```

## Tools

| tool | use it when |
|---|---|
| `decide` | several actions are plausible, or none is clearly right |
| `decide_many` | you have many questions about **one** state — one pass, all answers |
| `triage_options` | you have more than ~20 candidates and need to narrow first |
| `judge` | one yes/no with asymmetric costs — "is this irreversible?" |
| `backend_status` | a decision call failed and you need to know why |

**`decide_many` is the efficiency argument for using this at all.** Question ids
stay on the caller's side and are never sent to the model, so N questions about
one state cost **one forward pass, not N**. That is the difference between this
and asking a chat model each question in turn.

Question types: `choice` (a label set), `score` (an ordered 2–10 scale), `noul`
(a yes/no returning `P(true)`).

```jsonc
// decide_many — three independent judgments, one call
{
  "state": { "diff": "...", "test": "failing" },
  "questions": {
    "risk":       { "type": "score",  "instructions": "How risky is `diff` to ship?",  "criteria": ["safe", "minor", "moderate", "dangerous"] },
    "reversible": { "type": "noul",   "instructions": "Is the change in `diff` reversible?" },
    "owner":      { "type": "choice", "instructions": "Which area does `diff` touch?", "criteria": { "api": "the HTTP surface", "db": "schema or queries", "ui": "the front end" } }
  }
}
```

## Command line

The same decisions without an MCP client. The gate is also the **exit code**
(`0` = act, `3` = escalate), so a shell script can drive it without parsing
anything.

```bash
layajev status
layajev decide --state "the test fails on a malformed email" \
  --option "fix the email::resolve and pass the real address" \
  --option "drop the field::Stripe does not require an email" \
  --instruction "Which fix is correct for 'state'?"
layajev judge --state "<diff>" --question "is this change reversible?"
```

## Honest limits

Read this before trusting a number.

- **It is a decision model, not an oracle.** The base checkpoint scores around
  0.36 on typed decisions against 0.77 when fine-tuned per domain. Treat a
  confident answer as a strong prior, not as ground truth — and `escalate` is the
  branch for when it matters.
- **Confidence is not correctness.** The `confidence` field describes how peaked
  the distribution is. A peaked distribution over the wrong options is still
  wrong, and it will look decisive.
- **Laya's shipped checkpoint warns on load** that some of its temperatures are
  out of range, so treat confidence from the affected entries as uncalibrated.
  The warning is surfaced, not hidden — see it in stderr.
- **Non-Latin scripts collapse** in the English checkpoint (measured: near-zero
  accuracy at high confidence). Use the multilingual checkpoint for those.
- **Past ~20 options a `choice` degrades**, because options share a fixed token
  budget. `triage_options` exists for exactly this — and it reports what it
  discarded rather than cutting silently.
- **The threshold is always yours.** No engine here returns a boolean, and none
  should. `act_at`/`margin`/`noul_act_at` are how you say what a decision is
  worth acting on.

## Not a duplicate of `laya-mcp-server`

Upstream Laya ships its own MCP server exposing the raw primitives —
`laya_predict`, `laya_route`, `laya_preset`, `laya_status`. If those are what you
want, run that; it is installed by `laya[mcp]` and this package deliberately does
not re-implement it.

This is the layer above: what to **ask**, and what to **do** with the answer.

## Protocol

MCP `2025-06-18`. Every tool declares an `outputSchema` and returns a matching
`structuredContent`, with the same JSON mirrored into a text block so clients
that predate structured output still receive something. Failures set `isError`
with the reason in the text, so a caller can tell a refusal from an empty result.

## Tests

```bash
python -m pytest tests/ -q
```

No model, network, or API key is needed: both backends are faked at the
`predict()` boundary, so the suite covers the decision logic, the gate, the
validation, and the JSON-RPC surface without loading a checkpoint.

## Licence

MIT.
