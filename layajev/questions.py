"""Question builders and validation.

A `question` is the unit both engines understand. Three types, and the type is
not cosmetic — it decides how the answer is scored and rendered:

  choice   pick one label from a named set. `criteria` is a map of
           label -> description, and the description is what the engine actually
           reads, so a vague description makes a vague decision.
  score    place the state on an ordered scale. `criteria` is an ORDERED list,
           lowest first.
  noul     a yes/no. Returns P(true) directly, which is the one type whose
           number needs no interpretation.

The ids are the caller's own keys and are never sent to the model — several
questions about one state cost one forward pass, not one per question. That is
the efficiency argument for using this at all instead of asking a chat model
each question separately.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

QTYPE_CHOICE = "choice"
QTYPE_SCORE = "score"
QTYPE_NOUL = "noul"
VALID_TYPES = (QTYPE_CHOICE, QTYPE_SCORE, QTYPE_NOUL)

# A choice with more options than this degrades: options share a fixed token
# budget, and the engine's own docs measure the drop-off past ~20. Callers who
# genuinely have more should pre-filter (see `layajev.shortlist`).
MAX_CHOICE_OPTIONS = 20
MAX_SCORE_STEPS = 10


class QuestionError(ValueError):
    """A malformed question. Raised rather than silently defaulted, because a
    question the engine cannot read produces a confident wrong answer."""


def choice(
    instruction: str,
    options: Any,
    *,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """A `choice` question.

    `options` is either an ordered sequence of labels, or a mapping of
    label -> description. A bare list is accepted for convenience and each label
    becomes its own description — but a description is what the engine decides
    from, so prefer the mapping form for anything where the label alone could be
    read two ways.
    """
    criteria = _normalise_choice(options, instruction)
    q: Dict[str, Any] = {"type": QTYPE_CHOICE, "instructions": _text(instruction, "instruction"),
                         "criteria": criteria}
    if name:
        q["_name"] = name
    return q


def score(
    instruction: str,
    steps: Sequence[str],
    *,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """A `score` question. `steps` is ordered lowest -> highest, 2..10 entries."""
    if isinstance(steps, (str, bytes)) or not isinstance(steps, (list, tuple)):
        raise QuestionError("score steps must be an ordered list of descriptions")
    if len(steps) < 2:
        raise QuestionError("score needs at least 2 steps to be a scale")
    if len(steps) > MAX_SCORE_STEPS:
        raise QuestionError(
            f"score supports at most {MAX_SCORE_STEPS} steps, got {len(steps)} — "
            "collapse the scale or use a choice"
        )
    q: Dict[str, Any] = {
        "type": QTYPE_SCORE,
        "instructions": _text(instruction, "instruction"),
        "criteria": [_text(s, "score step") for s in steps],
    }
    if name:
        q["_name"] = name
    return q


def noul(
    instruction: str,
    *,
    name: Optional[str] = None,
    true_description: Optional[str] = None,
    false_description: Optional[str] = None,
) -> Dict[str, Any]:
    """A yes/no question. Returns P(true).

    The asymmetric-cost case is the reason this exists: "is this action
    irreversible?" is worth asking even when the answer is usually no, because
    the cost of the rare yes is not the cost of the common no.
    """
    q: Dict[str, Any] = {"type": QTYPE_NOUL, "instructions": _text(instruction, "instruction")}
    if true_description or false_description:
        # Expressible when the engine supports the disambiguation; omitted rather
        # than faked when it does not, so a caller never trusts a field that was
        # silently dropped.
        crit: Dict[str, str] = {}
        if false_description:
            crit["false"] = _text(false_description, "description")
        if true_description:
            crit["true"] = _text(true_description, "description")
        q["criteria"] = crit
    if name:
        q["_name"] = name
    return q


def validate_questions(questions: Any) -> Dict[str, Dict[str, Any]]:
    """Validate a caller-supplied question map and strip private keys.

    Accepts what a JSON-RPC client sends over the wire, so it is permissive about
    types it then enforces — the point is to fail with a usable message rather
    than to pass something the engine will misread.
    """
    if not isinstance(questions, dict) or not questions:
        raise QuestionError("questions must be a non-empty object keyed by question name")

    cleaned: Dict[str, Dict[str, Any]] = {}
    for name, spec in questions.items():
        if not isinstance(name, str) or not name.strip():
            raise QuestionError(f"question name must be a non-empty string, got {name!r}")
        if not isinstance(spec, dict):
            raise QuestionError(f"questions[{name!r}] must be an object")

        qtype = spec.get("type")
        if qtype not in VALID_TYPES:
            raise QuestionError(
                f"questions[{name!r}].type must be one of {list(VALID_TYPES)}, got {qtype!r}"
            )
        instructions = spec.get("instructions")
        if not isinstance(instructions, str) or not instructions.strip():
            raise QuestionError(f"questions[{name!r}].instructions must be non-empty text")

        entry: Dict[str, Any] = {"type": qtype, "instructions": instructions.strip()}
        criteria = spec.get("criteria")

        if qtype == QTYPE_CHOICE:
            entry["criteria"] = _normalise_choice(criteria, f"questions[{name!r}]")
        elif qtype == QTYPE_SCORE:
            if isinstance(criteria, (str, bytes)) or not isinstance(criteria, (list, tuple)):
                raise QuestionError(f"questions[{name!r}].criteria must be an ordered list")
            if not 2 <= len(criteria) <= MAX_SCORE_STEPS:
                raise QuestionError(
                    f"questions[{name!r}].criteria needs 2..{MAX_SCORE_STEPS} steps, got {len(criteria)}"
                )
            entry["criteria"] = [_text(c, "score step") for c in criteria]
        elif criteria is not None:
            # noul criteria is optional and only meaningful as a map.
            if isinstance(criteria, Mapping):
                entry["criteria"] = {str(k): _text(v, "description") for k, v in criteria.items()}

        cleaned[name] = entry
    return cleaned


def _normalise_choice(options: Any, where: str) -> Dict[str, str]:
    if isinstance(options, Mapping):
        if not options:
            raise QuestionError(f"{where}: choice needs at least one option")
        out = {str(k): _text(v, "option description") for k, v in options.items()}
    elif isinstance(options, (list, tuple)):
        if not options:
            raise QuestionError(f"{where}: choice needs at least one option")
        out = {}
        for o in options:
            if isinstance(o, Mapping):
                # {id, label, description} — the shape the decide() helpers build.
                label = o.get("label") or o.get("id")
                if not label:
                    raise QuestionError(f"{where}: an option needs a label or an id")
                out[str(label)] = _text(o.get("description") or label, "option description")
            else:
                label = _text(o, "option")
                out[label] = label
    else:
        raise QuestionError(f"{where}: choice options must be a list or an object")

    if len(out) < 2:
        raise QuestionError(f"{where}: a choice needs at least 2 options to be a decision")
    if len(out) > MAX_CHOICE_OPTIONS:
        raise QuestionError(
            f"{where}: {len(out)} options exceeds the {MAX_CHOICE_OPTIONS} that read reliably — "
            "filter the list first (see layajev.shortlist)"
        )
    return out


def _text(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QuestionError(f"{what} must be non-empty text, got {value!r}")
    return value.strip()


def options_from_list(options: Iterable[Any]) -> List[str]:
    """The label order of a choice, for rendering probabilities back in order."""
    return [str(o.get("label") or o.get("id")) if isinstance(o, Mapping) else str(o) for o in options]
