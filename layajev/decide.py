"""The agent layer: turn a situation and a list of options into a gate decision.

`backends` answers questions. This module decides what to ASK and what to DO with
the answer, which is the part an agent actually needs and the part neither engine
provides.

The shape of the problem: an agent has several plausible courses of action, or
none that is clearly right. The useful output is not "here are probabilities" —
it is "act on this one, confidently enough" or "this is too close to call, go
ask someone". A probability without a threshold is not a decision, so the
threshold is explicit, per call, and the caller's to set.

Why a gate rather than always taking the top option: an agent that always picks
the argmax will happily act on a 51/49 split and never notice it guessed. The
`escalate` branch exists so an ambiguous situation has somewhere to go that is
not a coin flip.

    decide(state, options, instruction)
      -> {"decision": "act"|"escalate", "recommendation": <label>, "probabilities": {...}}

`decide_many` answers several independent questions about ONE state in a single
call, which is the efficiency the engines are built for: question ids are not
sent to the model, so N questions cost one forward pass, not N.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .backends import Backend, BackendError, auto_backend
from .questions import (
    QTYPE_CHOICE,
    QTYPE_NOUL,
    QTYPE_SCORE,
    QuestionError,
    choice as _choice,
    noul as _noul,
    score as _score,
    validate_questions,
)


@dataclass
class DecisionPolicy:
    """When to act and when to hand the decision back.

    `act_at` is the probability the leading option must reach to act on it.
    `margin` is how far ahead of the RUNNER-UP it must be. They catch different
    failures, which is why both exist:

      - a 0.55 lead over a 0.45 second option passes a bare `act_at` check while
        being a coin flip — `margin` refuses it;
      - a 0.60 option with four rivals at 0.10 each is a clear winner despite a
        low absolute number — `act_at` refuses it, which is the conservative and
        correct default for an autonomous agent.

    Lower `act_at` to make the agent act more, raise it to make it ask more.
    """

    act_at: float = 0.65
    margin: float = 0.15
    # A `noul` question is a probability, not a set of options, so it is gated on
    # its own threshold — the point of a yes/no here is usually the asymmetric case
    # ("is this irreversible?"), where the caller wants a much lower bar to act.
    noul_act_at: float = 0.5

    def verdict(self, probs: Mapping[str, float]) -> Dict[str, Any]:
        if not probs:
            return {"decision": "escalate", "reason": "no probabilities were returned"}

        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        top_label, top_p = ranked[0]
        runner_p = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = top_p - runner_p

        if top_p >= self.act_at and margin >= self.margin:
            return {
                "decision": "act",
                "reason": f"leading option at {top_p:.3f} with a {margin:.3f} margin",
                "recommendation": top_label,
                "confidence": top_p,
                "margin": margin,
            }
        why = []
        if top_p < self.act_at:
            why.append(f"leader {top_p:.3f} is below the {self.act_at} act threshold")
        if margin < self.margin:
            why.append(f"margin {margin:.3f} over {ranked[1][0]!r} is inside the {self.margin} band")
        return {
            "decision": "escalate",
            "reason": "; ".join(why) or "too close to call",
            "recommendation": top_label,
            "confidence": top_p,
            "margin": margin,
            "runner_up": ranked[1][0] if len(ranked) > 1 else None,
        }

    def verdict_noul(self, p_true: float) -> Dict[str, Any]:
        if p_true >= self.noul_act_at:
            return {
                "decision": "act",
                "reason": f"P(true) {p_true:.3f} clears the {self.noul_act_at} bar",
                "recommendation": True,
                "confidence": p_true,
            }
        return {
            "decision": "escalate",
            "reason": f"P(true) {p_true:.3f} is below the {self.noul_act_at} bar",
            "recommendation": False,
            "confidence": p_true,
        }


def _probs_from_answer(answer: Mapping[str, Any]) -> Dict[str, float]:
    """Pull a label->probability map out of whichever shape the engine returned.

    Both engines report a distribution for a choice, but they do not agree on the
    field name, and a caller reading the wrong one gets an empty dict that looks
    like "the model abstained". Every observed shape is handled explicitly, and an
    unrecognised one raises rather than returning nothing.
    """
    for key in ("probabilities", "probs", "distribution"):
        v = answer.get(key)
        if isinstance(v, Mapping) and v:
            out: Dict[str, float] = {}
            for k, p in v.items():
                try:
                    out[str(k)] = float(p)
                except (TypeError, ValueError):
                    continue
            if out:
                return out
    # Some responses carry the chosen label and a confidence but no distribution.
    picked = answer.get("choice")
    if picked is not None:
        conf = answer.get("answer_confidence", answer.get("confidence"))
        try:
            p = float(conf)
        except (TypeError, ValueError):
            p = None
        if p is not None:
            rest = max(0.0, 1.0 - p)
            # Two labels is the common case; the remainder is reported rather than
            # split, so the caller sees the real number instead of an invention.
            return {str(picked): p, "__other__": rest}
    raise BackendError(
        f"the backend returned no probability distribution for this question: {sorted(answer)}"
    )


def decide(
    state: Any,
    options: Sequence[Any],
    instruction: str,
    *,
    backend: Optional[Backend] = None,
    backend_name: Optional[str] = None,
    policy: Optional[DecisionPolicy] = None,
    context_label: str = "state",
    extra_questions: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Ask which of `options` fits `state`, and gate the answer.

    `options` is a list of labels, or of `{"id"|"label", "description"}` objects.
    A description is what the engine reads, so give one whenever the label alone
    could be read two ways.
    """
    # Validate the caller's own arguments BEFORE resolving a backend or touching a
    # model. The order matters: a malformed request should be reported as malformed
    # regardless of what engines happen to be installed, and it should never cost a
    # checkpoint load or a network round trip to discover a typo.
    labels = _labels(options)
    question = _choice(instruction, _criteria(options))
    validated_extra: Dict[str, Any] = {}
    if extra_questions:
        validated_extra = validate_questions(dict(extra_questions))

    be = backend or auto_backend(backend_name)
    pol = policy or DecisionPolicy()

    questions: Dict[str, Any] = {"recommended_option": question}
    questions.update(validated_extra)

    out = be.predict(_prepare_state(state, context_label), questions)
    answers = out.get("answers")
    if not isinstance(answers, Mapping):
        raise BackendError(f"the backend returned no answers object: {sorted(out)}")

    first = answers.get("recommended_option")
    if not isinstance(first, Mapping):
        # A choice question that came back as a bare label is still usable.
        first = {"choice": first} if first is not None else {}
    probs = _probs_from_answer(first)

    # Report every label the caller offered, including any the engine did not
    # mention — a silently missing option reads as impossible when it was never
    # scored, and those are different facts.
    full = {lbl: float(probs.get(lbl, probs.get(lbl.strip(), 0.0))) for lbl in labels}
    verdict = pol.verdict(full)

    result: Dict[str, Any] = {
        "decision": verdict["decision"],
        "reason": verdict["reason"],
        "recommendation": verdict.get("recommendation"),
        "confidence": verdict.get("confidence"),
        "margin": verdict.get("margin"),
        "probabilities": full,
        "backend": getattr(be, "name", "unknown"),
        "model": out.get("model"),
        "state_label": context_label,
    }
    if verdict.get("runner_up"):
        result["runner_up"] = verdict["runner_up"]
    if out.get("usage"):
        result["usage"] = out["usage"]

    # Extra questions are answered from the same pass, so they cost nothing extra
    # and there is no reason to make the caller ask them separately.
    if extra_questions:
        result["extra"] = {k: _summarise_answer(answers.get(k)) for k in extra_questions}
    return result


def decide_many(
    state: Any,
    questions: Mapping[str, Any],
    *,
    backend: Optional[Backend] = None,
    backend_name: Optional[str] = None,
    context_label: str = "state",
) -> Dict[str, Any]:
    """Answer several independent questions about one state in a single call.

    This is the efficiency the engines are built for — question ids stay on the
    caller's side — so N questions cost one pass, not N.
    """
    be = backend or auto_backend(backend_name)
    qs = validate_questions(questions)
    out = be.predict(_prepare_state(state, context_label), qs)
    answers = out.get("answers")
    if not isinstance(answers, Mapping):
        raise BackendError(f"the backend returned no answers object: {sorted(out)}")
    return {
        "answers": {k: _summarise_answer(answers.get(k)) for k in qs},
        "backend": getattr(be, "name", "unknown"),
        "model": out.get("model"),
        "usage": out.get("usage"),
        "state_label": context_label,
    }


def _summarise_answer(answer: Any) -> Dict[str, Any]:
    """One question's answer, in a stable shape."""
    if not isinstance(answer, Mapping):
        return {"value": answer, "kind": "raw"}
    kind = answer.get("type") or "unknown"
    out: Dict[str, Any] = {"kind": kind}
    if kind == QTYPE_CHOICE or "choice" in answer:
        out["value"] = answer.get("choice")
        try:
            out["probabilities"] = _probs_from_answer(answer)
        except BackendError:
            pass
    elif kind == QTYPE_NOUL or "noul" in answer:
        v = answer.get("noul")
        try:
            out["value"] = float(v)
        except (TypeError, ValueError):
            out["value"] = v
    elif kind == QTYPE_SCORE or "score" in answer:
        out["value"] = answer.get("score")
        try:
            out["probabilities"] = _probs_from_answer(answer)
        except BackendError:
            pass
    else:
        out["value"] = answer.get("value")
    for k in ("confidence", "answer_confidence"):
        if k in answer:
            out[k] = answer[k]
    return out


def _labels(options: Sequence[Any]) -> List[str]:
    if not options:
        raise QuestionError("decide needs at least one option")
    labels: List[str] = []
    for o in options:
        if isinstance(o, Mapping):
            lbl = o.get("label") or o.get("id")
            if not lbl:
                raise QuestionError(f"an option needs a label or an id: {o!r}")
            labels.append(str(lbl))
        else:
            labels.append(str(o))
    if len(set(labels)) != len(labels):
        raise QuestionError(f"option labels must be unique, got {labels}")
    return labels


def _criteria(options: Sequence[Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for o in options:
        if isinstance(o, Mapping):
            lbl = str(o.get("label") or o.get("id"))
            out[lbl] = str(o.get("description") or o.get("label") or o.get("id"))
        else:
            out[str(o)] = str(o)
    return out


def _prepare_state(state: Any, label: str) -> Any:
    """Pass the state through unchanged when it is already structured.

    A bare string is wrapped so the engine has a key to read: the question
    templates refer to the state by name (`request`, `message`, `context`), and a
    bare string gives them nothing to bind to.
    """
    if isinstance(state, Mapping):
        return dict(state)
    if isinstance(state, str):
        return {label: state}
    # A list or scalar is a legitimate state too — hand it over as-is rather than
    # guessing a shape the caller did not choose.
    return state
