"""layajev — calibrated probability judgments for an LLM agent's own options.

The problem this solves is narrow and specific. An agent reaches a decision it
cannot resolve from the information it has: several courses of action are
plausible, or none is clearly right. Asking a language model to pick is the
expensive, slow, and often arbitrary way to settle it — and it produces prose,
not a number, so the caller cannot tell a confident answer from a coin flip.

Laya and TypeSafe Jev both answer the same question cheaply: given a `state`
and a set of typed `questions`, return probabilities. Laya runs locally and
free; Jev is remote and billed. They share a wire format.

What this package adds on top of either one is the AGENT-shaped layer: take a
situation and a list of candidate actions, and return not just probabilities but
a gated decision — act, or escalate — with the confidence that earned it.

    from layajev import decide

    result = decide(
        state={"test": "checkout_session_created raises InvalidRequestError"},
        options=[
            {"id": "fix_email", "label": "Pass a real email to Stripe"},
            {"id": "omit_email", "label": "Create the customer with no email"},
        ],
        instruction="Which fix is correct for `state`?",
    )
    result["decision"]        # "act" | "escalate"
    result["recommendation"]  # "omit_email"
    result["probabilities"]   # {"fix_email": 0.03, "omit_email": 0.97}

Deliberately NOT a re-implementation of `laya-mcp-server`: that server already
exposes the raw primitives (`laya_predict`, `laya_route`, `laya_preset`,
`laya_status`). If you want those, run it. This is the layer above them.
"""

from .decide import decide, decide_many, DecisionPolicy
from .backends import Backend, BackendError, LayaBackend, JevBackend, auto_backend, available_backends
from .questions import (
    MAX_CHOICE_OPTIONS,
    MAX_SCORE_STEPS,
    QuestionError,
    choice,
    noul,
    score,
    validate_questions,
)

__version__ = "0.1.0"

__all__ = [
    "decide",
    "decide_many",
    "DecisionPolicy",
    "Backend",
    "BackendError",
    "LayaBackend",
    "JevBackend",
    "auto_backend",
    "available_backends",
    "choice",
    "score",
    "noul",
    "validate_questions",
    "QuestionError",
    "MAX_CHOICE_OPTIONS",
    "MAX_SCORE_STEPS",
    "__version__",
]
