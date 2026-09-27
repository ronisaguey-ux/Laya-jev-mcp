"""Add the ORCHESTRATION tier to the taxonomy.

Bob, 2026-09-27: the Laya agents "will be used alot for orchestration, like the audit,
cross eval, execution agents, and you will use them". So the tree needs the decisions an
agent makes about WORK, not just about the market.

Same machinery as the trading leaves - closed option set, scored by the shared scorer,
gated with a threshold. These are the calls that are expensive and wrong today.
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_taxonomy as bt

leaf = bt.leaf

ORCH = [
 {"id": "D7", "name": "agent_routing",
  "routes_on": "Work exists and the question is who should do it, and how.",
  "sub_routers": [
    {"id": "D7.1", "name": "assignment", "leaves": [
      leaf("D7.1.1", "agent_choice",
           "Which kind of worker should take the task in `state`?",
           {"local_fast_lane": "a cheap local lane; the task is narrow and well specified",
            "strong_reasoning_lane": "a strong model; the task needs real reasoning or design",
            "specialist_tool": "a purpose-built tool rather than an agent; deterministic work",
            "human_owner": "a human; the task needs authority a model does not have",
            "no_worker": "nothing should take it; it is not actionable"},
           "AI-coding-orchestrator, orch_lanes.py", bt.NOMONEY),
      leaf("D7.1.2", "task_specificity",
           "Is the task in `state` specified well enough to hand to a worker?",
           {"ready": "it names the files, the change and the done condition",
            "underspecified": "it is missing the target or the done condition",
            "contradictory": "its parts cannot all be satisfied at once",
            "already_done": "the work it describes is already present"},
           "AI-coding-orchestrator, execute.py", bt.NOMONEY),
      leaf("D7.1.3", "parallelism_verdict",
           "Can the items in `state` be worked in parallel?",
           {"independent": "they touch disjoint files and can run together",
            "serial_required": "they touch the same files and must be ordered",
            "batch_safe": "they are independent enough to batch with care",
            "single_thread": "do one at a time; the risk of collision is high"},
           "AI-coding-orchestrator, orch_queue.py", bt.NOMONEY),
    ]},
    {"id": "D7.2", "name": "effort_band", "leaves": [
      leaf("D7.2.1", "task_difficulty",
           "How hard is the task in `state`?",
           {"trivial": "a lookup or a one-line change; no reasoning needed",
            "routine": "a small well-understood change in familiar code",
            "moderate": "several steps, some judgement about approach",
            "hard": "multi-file reasoning, design decisions, or novel debugging"},
           "AI-coding-orchestrator, lane budgets", bt.NOMONEY),
      leaf("D7.2.2", "effort_justified",
           "Does the effort spent on `state` match the value of the result?",
           {"proportional": "the cost is reasonable for what the work is worth",
            "over_invested": "more effort has gone in than the outcome warrants",
            "under_invested": "the task is important and is being under-resourced",
            "stop_spending": "no further effort is justified; stop it"},
           "AI-coding-orchestrator", bt.NOMONEY),
    ]},
  ]},
 {"id": "D8", "name": "work_triage",
  "routes_on": "A finding, a failure or a step needs a verdict on what happens to it next.",
  "sub_routers": [
    {"id": "D8.1", "name": "finding_verdict", "leaves": [
      leaf("D8.1.1", "finding_actionable",
           "Is the finding in `state` worth acting on?",
           {"actionable": "a real defect with a concrete fix available",
            "noise": "not a real problem; exclude it from the work",
            "already_fixed": "the described problem is already resolved in the code",
            "needs_investigation": "cannot tell yet whether it is real; dig first"},
           "audit agents, audits_plans/", bt.NOMONEY),
      leaf("D8.1.2", "finding_severity",
           "How severe is the finding in `state` if left alone?",
           {"cosmetic": "style or tidiness; nothing breaks",
            "minor": "a real defect with no money or data consequence",
            "serious": "it can corrupt data, state or a result",
            "critical": "it can lose money, leak credentials, or halt trading"},
           "audit agents, regression_risk_auditor.py", bt.ADVISE),
    ]},
    {"id": "D8.2", "name": "step_lifecycle", "leaves": [
      leaf("D8.2.1", "step_disposition",
           "What should happen to the step in `state` now?",
           {"retry": "a real attempt failed on a transient cause; try again",
            "escalate": "the worker cannot do it; a human or stronger model must",
            "retire_green": "the work is verifiably present; close it as done",
            "retire_yellow": "it cannot be done as specified; close it with the reason"},
           "AI-coding-orchestrator, execute.py", bt.NOMONEY),
      leaf("D8.2.2", "failure_cause",
           "Why did the attempt in `state` fail?",
           {"transient": "a timeout, a rate limit or a dropped connection",
            "worker_incapable": "the worker was not able to do this kind of task",
            "bad_instruction": "the task itself was wrong or underspecified",
            "code_defect": "our own engine or tooling broke, not the task"},
           "AI-coding-orchestrator, transport errors", bt.NOMONEY),
    ]},
  ]},
 {"id": "D9", "name": "agent_trust",
  "routes_on": "A claim, a disagreement or a result needs to be believed or not.",
  "sub_routers": [
    {"id": "D9.1", "name": "conflict_resolution", "leaves": [
      leaf("D9.1.1", "disagreement_kind",
           "What kind of disagreement is described in `state`?",
           {"real_conflict": "the agents reached genuinely incompatible conclusions",
            "wording_only": "they agree and are describing it differently",
            "different_scope": "they answered different questions; both can be right",
            "one_is_wrong": "one is demonstrably wrong on the evidence"},
           "cross-eval agents", bt.NOMONEY),
      leaf("D9.1.2", "arbitration",
           "How should the disagreement in `state` be settled?",
           {"evidence_decides": "check the artefact itself; the fact settles it",
            "more_samples": "gather more runs before deciding",
            "human_decision": "escalate; the trade-off is not mechanical",
            "prefer_conservative": "take the branch with the smaller blast radius"},
           "cross-eval agents", bt.NOMONEY),
    ]},
    {"id": "D9.2", "name": "evidence_quality", "leaves": [
      leaf("D9.2.1", "claim_supported",
           "Does the evidence in `state` actually support the claim made with it?",
           {"supported": "the artefact shows what the claim says it shows",
            "unsupported": "the claim goes beyond what the evidence shows",
            "contradicted": "the artefact shows the opposite of the claim",
            "unverifiable": "the claim cannot be checked from the evidence given"},
           "review agents, verification contract", bt.NOMONEY),
      leaf("D9.2.2", "completion_truth",
           "Is the work in `state` actually finished?",
           {"verified_done": "driven and observed working; the claim rests on a real result",
            "claimed_only": "reported done with no observation behind it",
            "partially_done": "some of the promised scope is missing",
            "skipped_verification": "finished but the check was never run"},
           "verification contract, oculus-verify", bt.NOMONEY),
      leaf("D9.2.3", "summary_trust",
           "Should a summary in `state` be taken at face value?",
           {"trust_with_check": "use it as a lead but confirm the key fact",
            "trust_directly": "it is a mechanical report from a tool, not a model's prose",
            "distrust": "a model summary of its own work; verify independently",
            "ignore": "it is stale or about a different revision"},
           "verification contract", bt.NOMONEY),
    ]},
  ]},
 {"id": "D10", "name": "scope_control",
  "routes_on": "A change or a request needs a boundary verdict.",
  "sub_routers": [
    {"id": "D10.1", "name": "change_boundary", "leaves": [
      leaf("D10.1.1", "change_scope",
           "Is the change in `state` inside the scope it was asked to touch?",
           {"in_scope": "it does what was asked and nothing else",
            "scope_creep": "it also changed adjacent things that were not asked for",
            "wrong_target": "it changed something unrelated to the request",
            "boundary_violation": "it crosses an architecture boundary"},
           "ocs/architecture_boundaries.md, .import-linter", bt.NOMONEY),
      leaf("D10.1.2", "reversibility",
           "How reversible is the action in `state`?",
           {"reversible": "it can be undone completely and cheaply",
            "costly_to_undo": "it can be undone but at real cost",
            "irreversible": "it cannot be undone; a human must authorise it",
            "external_side_effect": "it leaves this machine, so it cannot be recalled"},
           "outward-facing action policy", bt.SAFE),
      leaf("D10.1.3", "authority_check",
           "Who is allowed to authorise the action in `state`?",
           {"agent_may_proceed": "within standing authority; no approval needed",
            "owner_must_decide": "it is the owner's call, not the agent's",
            "blocked_until_evidence": "no authority until a specific check passes",
            "forbidden": "not permitted by any standing rule"},
           "standing rules, owner authority", bt.SAFE),
    ]},
  ]},
]

if __name__ == "__main__":
    bt.TREE["domains"].extend(ORCH)
    leaves = bt.walk()
    ids = [l["id"] for _, _, l in leaves]
    assert len(ids) == len(set(ids)), "duplicate leaf ids after adding orchestration"
    for _, _, lf in leaves:
        ops = lf["options"]
        assert 2 <= len(ops) <= 9, f"{lf['id']}: {len(ops)} options"
        for k, v in ops.items():
            assert isinstance(v, str) and v.strip() and len(v) <= 190
    dom = len(bt.TREE["domains"])
    subs = sum(len(d["sub_routers"]) for d in bt.TREE["domains"])
    bt.TREE["counts"] = {"domains": dom, "sub_routers": subs, "leaves": len(leaves)}
    bt.TREE["money_leaves"] = [l["id"] for _, _, l in leaves if l["touches_money"]]
    bt.TREE["irreversible_leaves"] = [l["id"] for _, _, l in leaves if l["irreversible"]]
    bt.TREE["tiers"] = {"trading_domains": ["D1","D2","D3","D4","D5","D6"],
                        "orchestration_domains": ["D7","D8","D9","D10"]}
    json.dump(bt.TREE, open(os.path.join(HERE, "taxonomy_map.json"), "w"), indent=2)
    print(f"  domains={dom} sub_routers={subs} leaves={len(leaves)}")
    print(f"  trading={len(bt.TREE['tiers']['trading_domains'])} orchestration={len(bt.TREE['tiers']['orchestration_domains'])}")
