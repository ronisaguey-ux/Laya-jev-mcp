"""Emit taxonomy_map.json — the frozen 3-tier decision tree for OCULUS.

Grounded in modules confirmed by reading the repo at f87b134f6, not invented:
  oculus/fitness_calculator.py (1393)  evolution/genome.py (629)
  evolution/genetics.py (1255)         evolution/evaluators.py (593)
  evolution/oos_metrics.py (458)       evolution/champion_registry.py (345)
  oculus/regime.py (533)               oculus/regime_signal.py (186)
  oculus/data_validation.py (792)      oculus/asset_universe.py (706)
  live/kill_switch.py (27)             live/kill_switch_guard.py (106)
  core/sot_guardrails.py (698)

Option strings are FROZEN. build_sequence() caps every option at 48 tokens, so
each description is written to fit and still discriminate.
"""
import json

# MONEY leaves may not act on a probability. They escalate unless near-certain and
# the tie-break is always the conservative branch (brief §3).
SAFE = {"act_at": 0.92, "margin": 0.30, "noul_act_at": 0.15}
ADVISE = {"act_at": 0.80, "margin": 0.25, "noul_act_at": 0.30}
NOMONEY = {"act_at": 0.65, "margin": 0.15, "noul_act_at": 0.50}


def leaf(lid, title, question, options, touched_by, policy, irreversible=False):
    return {
        "id": lid,
        "title": title,
        "question": question,
        "options": options,
        "touched_by": touched_by,
        "policy": policy,
        "touches_money": policy in (SAFE, ADVISE),
        "irreversible": irreversible,
    }


TREE = {
  "version": 1,
  "source_commit": "f87b134f6",
  "frozen_option_strings": True,
  "policy_tiers": {
    "money_safe": SAFE, "advisor": ADVISE, "nomoney": NOMONEY,
  },
  "domains": [
    {"id": "D1", "name": "strategy_evolution",
     "routes_on": "A genome or population is being scored, ranked, or advanced.",
     "sub_routers": [
       {"id": "D1.1", "name": "genome_triage", "leaves": [
         leaf("D1.1.1", "genome_promote_verdict",
              "What should happen to the mutated genome described in `state` next?",
              {"promote_to_oos": "send it onward to out-of-sample review; in-sample gain looks real",
               "reject": "discard it; the gain is noise or below the survival threshold",
               "needs_more_generations": "keep evolving this lineage; it is not converged yet",
               "quarantine": "hold it aside; the evidence contradicts itself and needs a closer look"},
              "evolution/genetics.py, evolution/genome.py", NOMONEY),
         leaf("D1.1.2", "survival_threshold",
              "Did the genome in `state` clear the survival threshold for its generation?",
              {"cleared": "fitness is at or above the threshold required this generation",
               "below": "fitness sits under the threshold and it will not survive",
               "borderline_review": "it sits on the line; a human should set the bar"},
              "evolution/genetics.py", NOMONEY),
         leaf("D1.1.3", "mutation_operator_choice",
              "Which mutation operator best fits the genome lineage in `state`?",
              {"weight_perturb": "nudge numeric genes only; structure is sound",
               "structural": "add or remove a gene; the search is stuck in a local optimum",
               "crossover": "recombine with another lineage; progress has stalled",
               "restart_lineage": "abandon this lineage; it is converged or degenerate"},
              "evolution/genetics.py, evolution/genome.py", NOMONEY),
         leaf("D1.1.4", "diversity_triage",
              "Is the population in `state` diverse enough to keep searching?",
              {"healthy": "spread is adequate; continue as planned",
               "converged": "the population has collapsed onto one lineage and needs injection",
               "fragmented": "too few viable members survive; relax selection briefly"},
              "evolution/genetics.py", NOMONEY),
       ]},
       {"id": "D1.2", "name": "champion_admission", "leaves": [
         leaf("D1.2.1", "champion_admit_verdict",
              "Should the candidate in `state` be admitted to the champion registry?",
              {"admit": "add it to the registry as the current champion",
               "admit_provisional": "register it as a challenger with a bounded trial",
               "reject": "do not register it; it does not beat the incumbent",
               "escalate_human": "the call is contestable and a human should decide"},
              "evolution/champion_registry.py", ADVISE, irreversible=True),
         leaf("D1.2.2", "champion_replace",
              "Should the registered champion in `state` be replaced by this challenger?",
              {"replace": "the challenger is better on every gated metric",
               "keep_incumbent": "the champion still wins; the challenger is not better",
               "run_both": "carry both while evidence accumulates",
               "escalate_human": "the margin is too thin to automate a swap"},
              "evolution/champion_registry.py", ADVISE, irreversible=True),
         leaf("D1.2.3", "generation_budget",
              "How should the generation budget in `state` be spent next?",
              {"continue": "keep the current allocation; progress is steady",
               "increase_mutation": "spend more on exploration; the search has plateaued",
               "increase_evaluation": "spend more on validation; results are unstable",
               "stop": "halt the run; further spend looks wasteful"},
              "evolution/genetics.py, evolution/evaluators.py", NOMONEY),
       ]},
     ]},
    {"id": "D2", "name": "validation_truth",
     "routes_on": "A result exists and the question is whether to believe it.",
     "sub_routers": [
       {"id": "D2.1", "name": "overfit_assessment", "leaves": [
         leaf("D2.1.1", "overfit_verdict",
              "Is the backtest result in `state` robust or curve-fit?",
              {"robust": "in-sample and out-of-sample agree; the result looks real",
               "curve_fit": "the edge disappears out of sample; it is fitted to noise",
               "inconclusive": "the evidence is too thin to call either way",
               "overfit_mild": "some degradation out of sample but the core survives"},
              "evolution/oos_metrics.py, oculus/fitness_calculator.py", NOMONEY),
         leaf("D2.1.2", "is_oos_divergence",
              "How far do the in-sample and out-of-sample results in `state` diverge?",
              {"consistent": "the two agree closely; no divergence to explain",
               "mild_decay": "out-of-sample is weaker but still positive",
               "severe_decay": "out-of-sample collapses; treat the result as unreliable",
               "inverted": "out-of-sample outperforms in-sample, which itself is suspicious"},
              "evolution/oos_metrics.py", NOMONEY),
         leaf("D2.1.3", "multiple_testing_burden",
              "Does the number of trials in `state` invalidate this result?",
              {"acceptable": "the trial count is low enough to trust the edge",
               "needs_deflation": "a deflated Sharpe adjustment is required before believing it",
               "too_many_trials": "too many trials to trust any single winner"},
              "evolution/oos_metrics.py, oculus/fitness_calculator.py", NOMONEY),
       ]},
       {"id": "D2.2", "name": "metric_interpretation", "leaves": [
         leaf("D2.2.1", "metric_trust",
              "Which metric in `state` should carry the most weight for this decision?",
              {"deflated_sharpe": "trust the deflated Sharpe; it already discounts trial count",
               "profit_factor": "trust profit factor; the shape of the curve is reliable",
               "ulcer_index": "trust the ulcer index; drawdown behaviour matters most here",
               "information_ratio": "trust the information ratio; consistency matters most here",
               "none_reliable": "no metric here is reliable enough to decide on"},
              "oculus/fitness_calculator.py", NOMONEY),
         leaf("D2.2.2", "backtest_artifact_check",
              "Does the result in `state` show a signature of a pipeline artifact rather than an edge?",
              {"clean": "no artifact signature; the result can be read at face value",
               "lookahead_suspect": "the timings look too good; suspect lookahead bias",
               "survivorship_suspect": "the asset set looks cherry-picked after the fact",
               "cost_model_suspect": "assumed costs look too optimistic to be real"},
              "evolution/oos_metrics.py, backtester/", NOMONEY),
         leaf("D2.2.3", "oos_review_routing",
              "Where should the result in `state` go next?",
              {"accept": "accept it and let it proceed to champion review",
               "rerun_validation": "re-run validation; the current evidence is incomplete",
               "reject": "reject it; it fails a required bar",
               "human_review": "send it to a human; the metrics conflict"},
              "evolution/oos_metrics.py, evolution/champion_registry.py", ADVISE),
       ]},
     ]},
    {"id": "D3", "name": "market_state",
     "routes_on": "The market or a bar stream needs a state label.",
     "sub_routers": [
       {"id": "D3.1", "name": "regime_reading", "leaves": [
         leaf("D3.1.1", "regime_label",
              "Which regime best describes the market state in `state`?",
              {"trend_up": "a sustained directional move upward",
               "trend_down": "a sustained directional move downward",
               "range": "price oscillates without a directional drift",
               "high_vol_chop": "large moves in both directions with no persistence",
               "transition": "the character of the market is changing right now"},
              "oculus/regime.py, oculus/regime_signal.py", NOMONEY),
         leaf("D3.1.2", "regime_quality",
              "How much should the regime reading in `state` be trusted?",
              {"high_confidence": "the clusters are well separated and stable",
               "low_confidence": "the reading sits between states and is unstable",
               "model_stale": "the fitted model is older than the data warrants",
               "unusable": "do not act on this regime reading at all"},
              "oculus/regime.py", NOMONEY),
         leaf("D3.1.3", "regime_shift_alert",
              "Has the market in `state` shifted out of its previous regime?",
              {"no_shift": "the state is unchanged; keep the current playbook",
               "early_shift": "early signs of a change; reduce conviction",
               "confirmed_shift": "the shift is established; the playbook must change"},
              "oculus/regime.py, oculus/regime_signal.py", NOMONEY),
       ]},
       {"id": "D3.2", "name": "asset_suitability", "leaves": [
         leaf("D3.2.1", "asset_tradeability",
              "Is the asset in `state` tradeable under the current conditions?",
              {"tradeable": "eligible now; data and liquidity support trading it",
               "supporting_only": "it is a supporting series, never traded directly",
               "not_tradeable": "it fails a requirement and must not be traded",
               "needs_review": "eligibility is ambiguous and a human should confirm"},
              "oculus/asset_universe.py, config/asset_universe.yaml", SAFE, irreversible=True),
         leaf("D3.2.2", "bar_type_for_asset",
              "Which bar type should be used for the asset in `state`?",
              {"temporal": "use a fixed-time bar; the series behaves evenly in time",
               "dollar_bars": "sample by traded value; volume is uneven in time",
               "volume_bars": "sample by volume; trade intensity is the driver",
               "range_bars": "sample by price movement; quiet periods are uninformative",
               "tick_bars": "sample by transaction count on the raw tick stream"},
              "oculus/asset_universe.py", NOMONEY),
         leaf("D3.2.3", "timeframe_fit",
              "Which timeframe best matches the intent in `state`?",
              {"fast": "a short horizon suits this decision", "medium": "an intraday-to-daily horizon suits this decision",
               "slow": "a multi-day horizon suits this decision", "no_preference": "the horizon does not change the answer"},
              "oculus/asset_universe.py", NOMONEY),
       ]},
     ]},
    {"id": "D4", "name": "risk_posture",
     "routes_on": "Exposure, sizing or a halt is being considered. Every leaf here is money.",
     "sub_routers": [
       {"id": "D4.1", "name": "exposure_posture", "leaves": [
         leaf("D4.1.1", "risk_posture",
              "What risk posture fits the conditions in `state`?",
              {"normal": "no change warranted; conditions are within the ordinary band",
               "cautious": "reduce conviction and size while conditions are unclear",
               "de_risk": "cut exposure now; the evidence points to elevated risk",
               "halt": "stop opening new risk and escalate to a human"},
              "oculus/regime.py, core/sot_guardrails.py", SAFE, irreversible=True),
         leaf("D4.1.2", "size_direction_check",
              "Does the request in `state` increase or reduce exposure?",
              {"reduce_exposure": "the request lowers or closes exposure, which may proceed",
               "hold_exposure": "the request changes nothing and may proceed",
               "increase_exposure": "the request raises exposure and must not be driven by a model score",
               "unclear": "the effect on exposure cannot be determined; treat as an increase"},
              "core/sot_guardrails.py", SAFE, irreversible=True),
         leaf("D4.1.3", "daily_loss_breach",
              "Has the loss in `state` breached a daily limit?",
              {"within_limit": "losses remain inside the daily limit",
               "approaching_limit": "losses are close to the limit; prepare to stop",
               "breached": "the daily limit is breached and trading must stop",
               "unknown_state": "the loss cannot be established; treat as breached"},
              "core/sot_guardrails.py, live/kill_switch.py", SAFE, irreversible=True),
       ]},
       {"id": "D4.2", "name": "halt_and_kill", "leaves": [
         leaf("D4.2.1", "kill_switch_trigger",
              "Should the kill switch fire for the conditions in `state`?",
              {"fire": "halt everything now; the condition is a halt condition",
               "arm_only": "arm the switch but do not fire; the condition is close",
               "do_not_fire": "the condition is not a halt condition; continue",
               "escalate_human": "a human must decide; the condition is outside the rules"},
              "live/kill_switch.py, live/kill_switch_guard.py", SAFE, irreversible=True),
         leaf("D4.2.2", "halt_scope",
              "How wide should the halt in `state` be?",
              {"all_strategies": "halt every strategy; the problem is systemic",
               "one_strategy": "halt the single strategy implicated",
               "one_asset": "halt the single asset implicated",
               "new_entries_only": "block new entries but let existing positions manage out"},
              "live/kill_switch.py, live/kill_switch_guard.py", SAFE, irreversible=True),
         leaf("D4.2.3", "resume_decision",
              "Should trading resume per the conditions in `state`?",
              {"resume": "the halt condition is cleared; resume normal operation",
               "resume_reduced": "resume at materially reduced size while it proves out",
               "stay_halted": "the condition has not cleared; stay halted",
               "human_only": "only a human may authorise resuming"},
              "live/kill_switch_guard.py, core/sot_guardrails.py", SAFE, irreversible=True),
       ]},
     ]},
    {"id": "D5", "name": "data_integrity",
     "routes_on": "Bars, ticks or a feed must be judged before anything is computed on them.",
     "sub_routers": [
       {"id": "D5.1", "name": "bar_quality", "leaves": [
         leaf("D5.1.1", "bar_trust",
              "Is the bar or series in `state` trustworthy?",
              {"trustworthy": "it passes the checks; compute on it",
               "stale": "it has not updated when it should have; do not compute on it",
               "corrupt": "it fails integrity checks; quarantine it",
               "repaired": "it had gaps that were repaired; use with reduced confidence"},
              "oculus/data_validation.py", NOMONEY),
         leaf("D5.1.2", "gap_classification",
              "What kind of gap does `state` show?",
              {"expected": "the gap matches a known market closure",
               "feed_dropout": "the gap looks like a data outage, not a real market pause",
               "genuine_move": "the gap is a real price move, not missing data",
               "unknown": "the gap cannot be classified; treat as missing data"},
              "oculus/data_validation.py", NOMONEY),
         leaf("D5.1.3", "outlier_verdict",
              "Is the extreme value in `state` a real print or an error?",
              {"real_move": "a genuine market move; keep it",
               "data_error": "a bad print; exclude it from computation",
               "needs_review": "ambiguous; a human should inspect the raw record"},
              "oculus/data_validation.py", NOMONEY),
       ]},
       {"id": "D5.2", "name": "feed_health", "leaves": [
         leaf("D5.2.1", "feed_health",
              "What is the state of the feed in `state`?",
              {"healthy": "messages are arriving on time and well formed",
               "degraded": "messages arrive late or out of order but are usable",
               "disconnected": "the feed is down; work from the last known good state",
               "unknown": "feed health cannot be determined; assume it is down"},
              "core/websocket_client.py", NOMONEY),
         leaf("D5.2.2", "anomaly_class",
              "What class of anomaly is the event in `state`?",
              {"market_event": "a genuine market event that must be handled as such",
               "system_fault": "an internal fault in our own pipeline",
               "upstream_issue": "a problem at the venue or data provider",
               "benign_noise": "ordinary variation; nothing to act on"},
              "core/websocket_client.py, monitoring/", NOMONEY),
         leaf("D5.2.3", "data_source_preference",
              "Which source should be trusted for `state`?",
              {"primary": "use the primary source; it is healthy",
               "secondary": "the primary is suspect; use the secondary",
               "reconcile": "the sources disagree; reconcile before computing",
               "none": "no source is trustworthy enough; do not compute"},
              "oculus/data_validation.py, data_pipeline/", NOMONEY),
       ]},
     ]},
    {"id": "D6", "name": "execution_ops",
     "routes_on": "An order, a failure or a change needs classifying. Nothing here places an order.",
     "sub_routers": [
       {"id": "D6.1", "name": "order_and_fill", "leaves": [
         leaf("D6.1.1", "execution_style",
              "Which execution style suits the intent in `state`? (Classify only — this leaf never executes.)",
              {"passive": "post and wait; the spread matters more than the fill certainty",
               "aggressive": "cross the spread; certainty of fill matters more",
               "wait": "do not send yet; conditions are unfavourable",
               "split": "work the order in slices to limit impact"},
              "live/", ADVISE),
         leaf("D6.1.2", "failure_triage",
              "What is the failure in `state` and how urgent is it?",
              {"rejected_retryable": "the venue rejected it and it can safely be retried",
               "rejected_fatal": "the venue rejected it and retrying would repeat the failure",
               "partial_fill": "the order filled only partly and the remainder needs a decision",
               "disconnect": "connectivity was lost; positions must be reconciled before anything else"},
              "live/", NOMONEY),
         leaf("D6.1.3", "reconciliation_verdict",
              "Do our records and the venue records in `state` agree?",
              {"matched": "local and venue positions agree",
               "mismatch": "they disagree and the difference must be resolved before trading",
               "stale_local": "our view is stale; refresh before deciding",
               "venue_unknown": "the venue cannot be queried; assume mismatch"},
              "live/", NOMONEY),
       ]},
       {"id": "D6.2", "name": "change_control", "leaves": [
         leaf("D6.2.1", "rollback_verdict",
              "Is the regression in `state` bad enough to roll back?",
              {"rollback": "revert now; the regression breaches a required bar",
               "forward_fix": "keep it and fix forward; the regression is tolerable",
               "hold_and_watch": "neither yet; gather more evidence first",
               "human_decision": "a human must decide; the trade-off is not automatic"},
              "rollback_validator.py, regression_risk_auditor.py", ADVISE, irreversible=True),
         leaf("D6.2.2", "change_risk",
              "How risky is the change in `state` to apply?",
              {"low_risk": "isolated and reversible; apply it",
               "moderate_risk": "touches shared behaviour; apply with a staged rollout",
               "high_risk": "touches money or state; require review before applying",
               "forbidden": "crosses an architecture boundary; do not apply"},
              "ocs/architecture_boundaries.md, .import-linter", NOMONEY),
         leaf("D6.2.3", "sot_conformance_verdict",
              "Does the change in `state` conform to the source of truth?",
              {"conforms": "it matches the documented contract",
               "violates": "it contradicts the source of truth and must not land",
               "unspecified": "the source of truth does not cover this case",
               "needs_sot_update": "the contract should be amended, which is a human decision"},
              "docs/spec/OCULUS_SOURCE_OF_TRUTH_7_23.md, core/sot_guardrails.py", NOMONEY),
       ]},
     ]},
  ],
}


def walk():
    out = []
    for d in TREE["domains"]:
        for sr in d["sub_routers"]:
            for lf in sr["leaves"]:
                out.append((d["id"], sr["id"], lf))
    return out


if __name__ == "__main__":
    leaves = walk()
    ids = [l["id"] for _, _, l in leaves]
    assert len(ids) == len(set(ids)), "duplicate leaf ids"
    for _, _, lf in leaves:
        ops = lf["options"]
        assert 2 <= len(ops) <= 9, f"{lf['id']}: {len(ops)} options"
        for k, v in ops.items():
            assert isinstance(k, str) and k.strip(), f"{lf['id']}: bad label"
            assert isinstance(v, str) and v.strip(), f"{lf['id']}:{k} bad description"
            # ~48-token cap in build_sequence; 4 chars/token is the safe headroom.
            assert len(v) <= 190, f"{lf['id']}:{k} description {len(v)} chars risks the 48-token cap"
    domains = len(TREE["domains"])
    subs = sum(len(d["sub_routers"]) for d in TREE["domains"])
    TREE["counts"] = {"domains": domains, "sub_routers": subs, "leaves": len(leaves)}
    TREE["money_leaves"] = [l["id"] for _, _, l in leaves if l["touches_money"]]
    TREE["irreversible_leaves"] = [l["id"] for _, _, l in leaves if l["irreversible"]]
    with open("taxonomy_map.json", "w") as fh:
        json.dump(TREE, fh, indent=2)
    print(f"  domains={domains} sub_routers={subs} leaves={len(leaves)}")
    print(f"  leaves touching money   : {len(TREE['money_leaves'])}")
    print(f"  irreversible leaves     : {len(TREE['irreversible_leaves'])}")
    print("  wrote taxonomy_map.json")
