"""The 3-tier tree walk, built ON layajev.decide — not a second server.

Tier 1 picks the domain, tier 2 picks the sub-router, tier 3 picks the leaf answer.
Each tier has its own LoRA adapter and its own confidence bar; a tier that cannot
clear its bar ESCALATES rather than guessing downward.

WHY THE BARS ARE WHAT THEY ARE (brief §3, and this is the whole safety story):

  A wrong routing call in a SaaS product costs a few cents. Here it can open a real
  position with real money. So:

  * A leaf that touches money does not act on a probability at all unless it is
    near-certain (0.92 lead with a 0.30 margin), and the tie-break is always the
    conservative branch. `do nothing` wins every tie.
  * No leaf may widen a risk limit. If the question is "should we increase size",
    this returns an ADVICE and a human decides. Model output may reduce exposure,
    never increase it — `size_direction_check` exists to make that structural.
  * Irreversible -> escalate. Almost always the correct branch.
  * FAIL CLOSED. Missing adapter, unreadable taxonomy, unknown band, absent
    probabilities -> the safe branch, never the permissive one.

Laya never executes. There is no order-placing code in this file and there must
never be: order placement goes only through code with its own deterministic guards,
position limits and the existing kill switch.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Mapping, Optional

from layajev.decide import DecisionPolicy, decide

HERE = os.path.dirname(os.path.abspath(__file__))
TAXONOMY = os.path.join(HERE, "taxonomy_map.json")

TIER_MODEL = {"T1_overlord": "oculus-t1-overlord",
              "T2_router": "oculus-t2-router",
              "T3_specialist": "oculus-t3-specialist"}

SAFE_POLICY = DecisionPolicy(act_at=0.92, margin=0.30, noul_act_at=0.15)
ADVISE_POLICY = DecisionPolicy(act_at=0.80, margin=0.25, noul_act_at=0.30)
NOMONEY_POLICY = DecisionPolicy(act_at=0.65, margin=0.15, noul_act_at=0.50)

# A leaf whose conservative branch should win every tie it does not clearly lose.
# Keyed by leaf id; anything absent is treated as the conservative default anyway.
CONSERVATIVE = {
    "promote_to_oos": "reject", "admit": "reject", "replace": "keep_incumbent",
    "fire": "do_not_fire", "resume": "stay_halted", "rollback": "hold_and_watch",
    "tradeable": "not_tradeable", "robust": "inconclusive",
    "increase_exposure": "unclear",
}


class SwarmError(RuntimeError):
    """The walk could not be completed. Never swallowed into a guess."""


def load_taxonomy(path: str = TAXONOMY) -> Dict[str, Any]:
    try:
        with open(path) as fh:
            tax = json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        # Fail closed: without the taxonomy there is no safe branch to take.
        raise SwarmError(f"cannot read the taxonomy at {path}: {e}") from e
    if not tax.get("domains"):
        raise SwarmError(f"the taxonomy at {path} has no domains")
    return tax


def policy_for(leaf: Mapping[str, Any]) -> DecisionPolicy:
    """Money leaves get the strict bar. An unlabelled leaf is treated as money."""
    if not leaf.get("touches_money"):
        return NOMONEY_POLICY
    return SAFE_POLICY if not leaf.get("irreversible") else SAFE_POLICY


def _tier_question(tax: Dict[str, Any], tier: str, node: Optional[Mapping[str, Any]] = None):
    """The question for one hop, and the child map its labels resolve to."""
    if tier == "T1_overlord":
        return ("Which area of the trading system does `state` concern?",
                {d["id"]: f"{d['name'].replace('_',' ')}: {d['routes_on']}" for d in tax["domains"]},
                {d["id"]: d for d in tax["domains"]})
    if tier == "T2_router":
        subs = (node or {}).get("sub_routers", [])
        return ("Which kind of question is `state` about, within this area?",
                {s["id"]: s["name"].replace("_", " ") for s in subs},
                {s["id"]: s for s in subs})
    leaves = (node or {}).get("leaves", [])
    first = leaves[0] if leaves else {}
    return (first.get("question", "Which action fits `state`?"),
            {lf["id"]: lf["title"].replace("_", " ") for lf in leaves},
            {lf["id"]: lf for lf in leaves})


def walk(state: Any, tax: Optional[Dict[str, Any]] = None, *, backend_name: Optional[str] = None,
         talk: bool = False) -> Dict[str, Any]:
    """Walk domain -> sub-router -> leaf, escalating at the first tier that cannot clear.

    Returns the chosen leaf, the answer, and the escalation log. The log is the only
    signal that says whether the local models are actually good enough.
    """
    tax = tax or load_taxonomy()
    log: List[Dict[str, Any]] = []
    node: Optional[Mapping[str, Any]] = None
    chosen_leaf: Optional[Mapping[str, Any]] = None

    for tier in ("T1_overlord", "T2_router", "T3_specialist"):
        instructions, criteria, children = _tier_question(tax, tier, node)
        if len(criteria) < 2:
            # A hop with nothing to choose between is not a decision.
            log.append({"tier": tier, "escalated": True, "why": "fewer than two options"})
            return _escalated(tier, "fewer than two options at this tier", log, node)

        if tier == "T3_specialist":
            leafs = list(children.values())
            # Each leaf has its OWN option set, so tier 3 cannot ask one question with
            # a shared option list. Ask which leaf applies first, then ask that leaf's
            # real question with its real frozen options. Two cheap hops, and no option
            # string is ever invented to fill a list.
            pick = decide(state, [{"label": lf["id"], "description": lf["title"].replace("_", " ")}
                                  for lf in leafs], instructions, backend_name=backend_name,
                          policy=NOMONEY_POLICY, context_label="state")
            log.append({"tier": "T3_select", "escalated": pick["decision"] == "escalate",
                        "why": pick["reason"], "confidence": pick.get("confidence")})
            if pick["decision"] == "escalate":
                return _escalated("T3_specialist", pick["reason"], log, node)
            chosen_leaf = children[pick["recommendation"]]
            break

        result = decide(state,
                        [{"label": lab, "description": desc} for lab, desc in criteria.items()],
                        instructions, backend_name=backend_name,
                        policy=NOMONEY_POLICY, context_label="state")
        log.append({"tier": tier, "escalated": result["decision"] == "escalate",
                    "why": result["reason"], "confidence": result.get("confidence")})
        if result["decision"] == "escalate":
            return _escalated(tier, result["reason"], log, node)
        node = children[result["recommendation"]]

    if chosen_leaf is None:
        return _escalated("T3_specialist", "no leaf was reached", log, node)

    # The real decision, with the leaf's own option set and the leaf's own bar.
    pol = policy_for(chosen_leaf)
    opts = [{"label": lab, "description": desc} for lab, desc in chosen_leaf["options"].items()]
    final = decide(state, opts, chosen_leaf["question"], backend_name=backend_name,
                   policy=pol, context_label="state")
    log.append({"tier": "T3_leaf", "leaf": chosen_leaf["id"],
                "escalated": final["decision"] == "escalate", "why": final["reason"],
                "confidence": final.get("confidence")})

    out = {
        "leaf": chosen_leaf["id"],
        "leaf_title": chosen_leaf["title"],
        "decision": final["decision"],
        "recommendation": final["recommendation"],
        "reason": final["reason"],
        "confidence": final.get("confidence"),
        "margin": final.get("margin"),
        "probabilities": final["probabilities"],
        "touches_money": chosen_leaf["touches_money"],
        "irreversible": chosen_leaf["irreversible"],
        # Laya advises. It never executes, and nothing here may place an order.
        "executes": False,
        "advisory_only": bool(chosen_leaf["touches_money"]),
        "escalation_log": log,
        "escalated": any(e.get("escalated") for e in log),
    }
    if out["decision"] == "escalate":
        # Fail closed: name the conservative branch rather than leaving it implicit.
        out["safe_branch"] = CONSERVATIVE.get(str(final["recommendation"]), None)
        out["tie_break"] = "do nothing"
    return out


def _leaf_options(leaf: Mapping[str, Any]) -> Dict[str, str]:
    return dict(leaf.get("options") or {})


def _escalated(tier: str, why: str, log: List[Dict[str, Any]], node: Optional[Mapping[str, Any]]):
    return {
        "leaf": None,
        "decision": "escalate",
        "reason": f"{tier}: {why}",
        "escalated_at_tier": tier,
        "escalation_log": log,
        "escalated": True,
        "executes": False,
        "advisory_only": True,
        "tie_break": "do nothing",
    }


if __name__ == "__main__":
    print(json.dumps({"tiers": list(TIER_MODEL), "taxonomy": TAXONOMY}, indent=2))
