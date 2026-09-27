"""Author hand-written seeds for the remaining leaves, in the same shape as the first four.

Bob: "generate the small batch urself, ur the llm" + "add more laya agents that are fine
tuned to answer more sub task answers". So each leaf gets its own deliberately written,
balanced seed set - one batch per option, never a majority-branch drift.
"""
from __future__ import annotations
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data_harvester as dh  # noqa: E402
import seed_orchestration as so  # noqa: E402
import seed_batch as sb  # noqa: E402

# Each entry: label -> list of states. 4 states per option keeps a leaf at 12-20 samples
# while covering every branch. Deliberately varied in asset, wording and magnitude.
LIB = {
"D1.1.3": {  # mutation_operator_choice
 "weight_perturb": [
  "genome g4471 gen 14: 1,204 trades, fitness 0.71, all 240 parameters within 12% of the population mean - the structure is already competitive.",
  "lineage 88 gen 12: fitness 0.68 on 1,880 trades, and ablating any single entry condition costs less than 0.02 fitness.",
  "genome b5510 gen 9: 902 trades, fitness 0.64, and the last four generations moved only numeric parameters, all upward.",
  "genome r7710 gen 16: 1,340 trades, fitness 0.70, every structural mutation in the last 6 generations scored below its parent."],
 "structural": [
  "genome n1110 gen 21: fitness 0.44 unchanged for 7 generations across 2,100 trades, and all numeric nudges return within noise.",
  "lineage 604 gen 18: 1,660 trades, fitness stuck at 0.47 for 9 generations while mutations only shift weights.",
  "genome c2211 gen 24: fitness 0.39 on 1,880 trades, converged, and the population contains no genome using the unexpressed regime slots.",
  "genome d1170 gen 19: 1,120 trades, fitness 0.41 flat for 8 generations, all 240 parameters present but none ever toggled."],
 "crossover": [
  "lineage 331 gen 15: fitness 0.58 on 2,010 trades, and two sibling lineages each carry 0.36 fitness with disjoint strong features.",
  "genome e5510 gen 17: 1,470 trades, fitness 0.55, and its weak regime block is strong in genome f9901 which sits beside it.",
  "genome a3340 gen 20: fitness 0.52 on 1,240 trades, stagnation for 5 generations, and the island holds complementary genomes above 0.50.",
  "lineage 915 gen 14: 1,660 trades, fitness 0.57, and the last crossover attempt produced 0.61 before being lost to the buffer limit."],
 "restart_lineage": [
  "lineage 771 gen 42: fitness 0.19 after 6,100 trades - the lineage is degenerate and its best member no longer reproduces.",
  "genome z7712 gen 38: fitness 0.12 on 4,880 trades, and 91% of the population is a clone of one genome with no variance left.",
  "genome q8890 gen 51: fitness 0.08 on 7,200 trades, all members fail the survival threshold and the island is effectively extinct.",
  "lineage 228 gen 47: fitness 0.15 on 5,400 trades with zero variance in the last 11 generations despite continued mutation."],
},
"D1.1.4": {  # diversity_triage
 "healthy": [
  "population of 200 genomes: fitness spread 0.12 to 0.71, 84 distinct parameter sets, mean pairwise distance 0.41.",
  "island 3 at generation 22: 200 members, 118 distinct, fitness standard deviation 0.14 across 2,010 trades evaluated.",
  "population at gen 15: 200 genomes, 96 unique, mean pairwise Hamming distance 0.38 on the 240 parameter slots.",
  "island 1 gen 30: 200 members, 132 distinct, fitness range 0.21 to 0.69, and 14 separate lineages above 0.50."],
 "converged": [
  "population of 200 genomes: fitness spread 0.63 to 0.71 but only 6 distinct parameter sets, mean pairwise distance 0.02.",
  "island 2 at generation 34: 189 of 200 members are clones of one genome, fitness standard deviation 0.008.",
  "population at gen 41: 200 genomes, 4 unique, mean pairwise distance 0.01 on the 240 slots - the search has collapsed.",
  "island 4 gen 28: 200 members, 7 distinct, and every genome shares the same 240-slot configuration with tiny weight offsets."],
 "fragmented": [
  "population of 200 genomes: only 23 clear the survival threshold, fitness spread 0.02 to 0.58 with a gap in the middle.",
  "island 1 at generation 19: 200 members but only 31 viable, and no genome above 0.60 survives to the next generation.",
  "population at gen 26: 41 viable of 200, fitness median 0.11, and the death rate exceeds the reproduction rate for 4 generations.",
  "island 2 gen 23: 200 members, 18 viable, and the survivors are unrelated singletons rather than a lineage."],
},
"D1.2.1": {  # champion_admit_verdict
 "admit": [
  "challenger g4471: OOS Sharpe 2.31 on 1,480 trades vs the incumbent's 1.84, deflated Sharpe 1.94, max drawdown 7.4% vs 11.2%.",
  "genome 88: beats the incumbent on all seven fitness metrics across two held-out windows, 2,204 trades, no metric regression.",
  "variant 1203: OOS return 38.9% against the incumbent's 22.1% on the same window, ulcer index lower at 0.041 vs 0.058.",
  "genome 77b: profit factor 1.92 vs 1.51, information ratio 0.62 vs 0.31, both measured on the identical held-out set of 388 trades."],
 "admit_provisional": [
  "challenger m2210: beats the incumbent on 5 of 7 metrics with 1,102 trades but the OOS window is only 6 weeks long.",
  "genome 9004: OOS Sharpe 2.02 vs 1.91 on 1,870 trades - a real lead but a margin inside the measurement error.",
  "variant 331: higher returns and lower drawdown on 640 trades, but the incumbent has 4,100 trades of live history behind it.",
  "genome 155: better on every OOS metric but only 402 trades, and the improvement concentrates in one month."],
 "reject": [
  "challenger b9902: OOS Sharpe 0.11 vs the incumbent's 1.84 on 1,102 trades - clearly worse in the held-out window.",
  "genome 61: OOS return -7.3% against the incumbent's 22.1% on 2,010 trades, and drawdown worse at 31.2%.",
  "variant 914: profit factor 0.79 vs 1.51 on 2,205 trades, fails on both profitability and risk metrics.",
  "genome 2201: IS Sharpe 2.62 but OOS 0.11 - it does not survive out of sample, so it cannot replace the incumbent."],
 "escalate_human": [
  "challenger h7730: better on OOS Sharpe and drawdown but with 3,011 trades in the incumbent and 402 in the challenger, no clean comparison exists.",
  "genome 5520: beats the incumbent on return, loses on drawdown, and the two windows disagree about which matters more.",
  "variant 3308: superior on deflated Sharpe but the challenger's data window is 2 days shorter than configured, so the comparison is unsound.",
  "genome 4410: the two candidates tie on fitness to 3 decimal places and the tie-break rule is not defined anywhere."],
},
"D2.1.1": {  # overfit_verdict
 "robust": [
  "IS Sharpe 2.31 over 1,480 trades, OOS Sharpe 2.18 over 620 trades, max drawdown 7.4% and 8.1% in the two windows.",
  "IS profit factor 1.92, OOS 1.87, 2,204 trades total, and both windows positive in each of the last 6 months.",
  "IS Calmar 1.81, OOS Calmar 1.76 on 388 trades, deflated Sharpe 1.94 after 240 trials, information ratio 0.62.",
  "IS return 27.4%, OOS 26.1% on 1,344 trades, ulcer index 0.036 and 0.038 - the degradation is inside noise."],
 "curve_fit": [
  "IS Sharpe 2.62, OOS Sharpe 0.11 on 1,204 trades - the edge does not survive the held-out window at all.",
  "IS profit factor 2.41, OOS 0.88 on 2,810 trades, and 71% of in-sample profit comes from 4 days.",
  "IS return 58.1%, OOS -7.3% on 1,102 trades, deflated Sharpe 0.31 after 420 trials.",
  "IS expectancy 0.0089, OOS -0.0014 on 1,660 trades - the sign flips between windows."],
 "inconclusive": [
  "IS Sharpe 1.91, OOS 1.44 but only 96 OOS trades - too few to separate a real decay from noise.",
  "IS and OOS both positive at 1.6 and 1.3 on 140 trades, and the OOS window is 3 weeks long.",
  "IS 0.71, OOS 0.48 with 210 trades, and the two windows cover materially different volatility regimes.",
  "IS profit factor 1.7, OOS 1.2 on 88 trades, and the deflated Sharpe was never computed."],
 "overfit_mild": [
  "IS Sharpe 1.88, OOS 1.41 on 1,102 trades - out of sample is weaker by 25% but stays clearly positive.",
  "IS profit factor 2.1, OOS 1.6 on 1,880 trades, with the core edge intact and only the tail trades missing.",
  "IS return 31%, OOS 19% on 2,010 trades, and every OOS month is positive except one.",
  "IS Calmar 1.6, OOS 1.1 on 640 trades - real degradation, but the strategy still earns out of sample."],
},
"D2.2.1": {  # metric_trust
 "deflated_sharpe": [
  "deflated Sharpe 1.94 computed across 240 trials, against an IS Sharpe of 2.31 - the trial count is known and high.",
  "deflated Sharpe 1.71 over 420 trials on 1,204 trades; the raw Sharpe of 2.44 is inflated by selection.",
  "deflated Sharpe 1.62 with 380 trials recorded, and the raw metrics agree with it in direction.",
  "deflated Sharpe 1.88 after 300 trials on 2,010 trades, all trial results logged and countable."],
 "profit_factor": [
  "profit factor 1.92 on 2,204 trades with a symmetric win/loss distribution and no single trade above 2.1% of profit.",
  "profit factor 2.11 on 1,880 trades, largest winner 1.4% of total profit, and the P&L is not fat-tailed.",
  "profit factor 1.78 on 3,100 trades where gross win and gross loss are both large and well sampled.",
  "profit factor 1.66 on 1,470 trades with the top 10 winners contributing under 18% of gross profit."],
 "ulcer_index": [
  "ulcer index 0.041 vs a peer median of 0.086 on 1,480 trades, and the equity curve recovers from every drawdown.",
  "ulcer index 0.036 with max drawdown 6.9% on 2,010 trades - drawdown behaviour is the binding constraint here.",
  "ulcer index 0.052 against a 10% max-drawdown limit, on 1,120 trades where time-under-water matters most.",
  "ulcer index 0.029 on 1,660 trades while Sharpe is only moderate - risk-adjusted pain is the strongest signal."],
 "information_ratio": [
  "information ratio 0.62 with tracking error 4.1% on 388 trades - consistency against the benchmark is the criterion.",
  "information ratio 0.58 on 2,204 trades, and the strategy beats the benchmark in 21 of 24 months.",
  "information ratio 0.71 with active return 3.2% on 1,340 trades, measured on the same window as the benchmark.",
  "information ratio 0.55 on 1,880 trades where the raw Sharpe overstates the edge against the benchmark."],
 "none_reliable": [
  "44 trials recorded but no deflated Sharpe computed; profit factor 1.3 on 61 trades; ulcer index undefined on 1 drawdown.",
  "all seven metrics present but every window is under 90 trades and the trial count is unknown.",
  "raw Sharpe 3.1 on 42 trades with no trial count, no OOS window and no cost model documented.",
  "metrics disagree in sign across windows and the trial count is not recorded anywhere in the run."],
},
"D3.1.1": {  # regime_label
 "trend_up": [
  "EURUSD 30m over 200 bars: ADX 34, price above a rising 50-period mean, higher highs on 78% of the last 40 bars.",
  "BTCUSDT 1h: 200 bars, slope of the 100-period regression +0.42% per 10 bars, drawdown from peak under 3%.",
  "SPY 1d over 180 bars: 21-day return +6.1%, price above the 200-day mean for 31 consecutive sessions.",
  "NQ 2h: ADX 31, 78% of bars closing above the prior close, and the 20-period mean above the 60-period mean."],
 "trend_down": [
  "GBPUSD 30m over 200 bars: ADX 36, price below a falling 50-period mean, lower lows on 74% of the last 40 bars.",
  "ETHUSDT 1h: 200 bars, regression slope -0.51% per 10 bars, and every rally in 30 bars failed to reclaim the mean.",
  "AAPL 1d over 180 bars: 21-day return -8.4%, price below the 200-day mean for 26 consecutive sessions.",
  "CL 1h: ADX 33, 71% of bars closing below the prior close, and the 20-period mean below the 60-period mean."],
 "range": [
  "EURUSD 15m over 200 bars: ADX 14, price oscillating within a 38-pip band, direction changes every 9 bars on average.",
  "AUDUSD 30m: 200 bars, no net move over 5 days, 60-period mean flat to within 0.02%.",
  "MSFT 1d over 150 bars: price between 398 and 412 for 6 weeks, ADX 11, mean reversion at both edges.",
  "USDJPY 1h: 200 bars, band width 44 pips, and 8 of the last 10 swings reversed inside the band."],
 "high_vol_chop": [
  "XRPUSDT 15m over 200 bars: realised volatility in the 96th percentile, 12 reversals in 40 bars, no persistence in either direction.",
  "BTCUSDT 5m: 200 bars, ATR at 4.2x its 30-day median, and the 20-bar return alternates sign 14 times.",
  "ES 5m: 200 bars, realised vol 3.1x the monthly average, ADX 9 with daily ranges above the 95th percentile.",
  "EURUSD 5m during an ECB release: 200 bars spanning 3 hours, 18 reversals, and no bar within 2 ATR of the previous close."],
 "transition": [
  "EURUSD 30m: ADX rising from 12 to 24 over 30 bars while band width expands 40% - the character is changing now.",
  "SPY 1d over 60 bars: 30-day realised volatility doubled in the last 8 sessions and the mean has just started to slope.",
  "BTCUSDT 1h: the regime model's top two clusters are 0.44 and 0.41, and the label changed twice in the last 10 bars.",
  "NQ 2h: range behaviour for 120 bars then two consecutive closes outside the band, with ADX turning up from 13."],
},
"D4.1.1": {  # risk_posture
 "normal": [
  "equity -0.4% today against a 3% daily limit, 3 open positions each under 0.5% risk, realised vol at the 52nd percentile.",
  "daily P&L +0.6%, drawdown from peak 1.1%, 4 open positions and all within their configured size bands.",
  "equity flat, open risk 1.2% of capital against a 4% cap, realised vol at the 48th percentile and no halt conditions active.",
  "daily loss 0.3%, max position 0.7% of capital, and the rolling 60-day drawdown sits at 2.1% against a 5% limit."],
 "cautious": [
  "equity -1.8% today against a 3% daily limit, realised vol at the 88th percentile, and 4 open positions in correlated assets.",
  "daily loss 2.1%, drawdown from peak 4.4% against a 5% rolling limit, and the regime label just flipped to transition.",
  "open risk 3.4% of capital against a 4% cap with two positions in the same sector, vol at the 91st percentile.",
  "equity -2.4% today, three of four open positions losing, and the data feed reported 0.4% late bars in the last hour."],
 "de_risk": [
  "equity -2.8% today against a 3% daily limit, drawdown 4.9% against a 5% rolling cap, 5 correlated positions open.",
  "open risk 3.9% of capital against a 4% cap, realised vol at the 97th percentile, and the feed is intermittent.",
  "daily loss 2.9%, the regime flipped to high-vol chop 2 bars ago, and four positions share the same underlying driver.",
  "drawdown 5.1% has breached the rolling limit and open exposure is still at 3.6% of capital."],
 "halt": [
  "daily loss reached 3.2%, breaching the 3% daily limit, with 6 positions still open and the feed down for 4 minutes.",
  "static drawdown stands at 10.4% against a 10% hard limit, and new entries were still attempted in the last bar.",
  "the venue is unreachable, local and venue positions disagree by 3 contracts, and open risk is 3.8% of capital.",
  "two kill conditions are simultaneously active: 5.2% rolling drawdown and a feed that has not ticked in 90 seconds."],
},
"D9.2.2": so.D9_2_2,
"D10.1.2": {  # reversibility
 "reversible": [
  "the change adds a new optional config key defaulting to the current behaviour; deleting the key restores the prior state exactly.",
  "the edit is confined to one file under version control and the previous revision is one git checkout away.",
  "the migration adds a nullable column; dropping it returns the schema and all rows to their prior state.",
  "the flag was flipped in a local config file; setting it back takes one line and affects nothing else."],
 "costly_to_undo": [
  "the change rewrote 40 files in one commit; reverting is possible but the merge conflicts across 12 of them are real work.",
  "the migration dropped and rebuilt a table; restoring needs a backup restore that takes 20 minutes of downtime.",
  "the branch was force-pushed, so the pre-change history survives only in the local reflog on this machine.",
  "the adapter was trained over 6 hours; reverting is trivial but regenerating it costs the same 6 hours."],
 "irreversible": [
  "the commit was pushed to main on a shared repository, so it cannot be unpublished and other agents may already have pulled it.",
  "messages were sent to the owner on Telegram; delivery cannot be recalled once the API returns ok true.",
  "the position was opened on the exchange; it can only be closed with a second trade at the prevailing price.",
  "the production database rows were deleted with no soft-delete column and no backup taken in the last 24 hours."],
 "external_side_effect": [
  "the package was published to a public registry; the version number is consumed and cannot be withdrawn from downstream installs.",
  "the PR was opened against a public repository and reviewers were notified by email.",
  "the webhook was deleted from the live bot, which affects every consumer of that endpoint outside this machine.",
  "the API key was rotated at the provider, so every external service holding the old credential breaks immediately."],
},
"D10.1.3": {  # authority_check
 "agent_may_proceed": [
  "the action is a read-only probe on a local service within the standing rules; no approval is required.",
  "the change is a fix to a failing test inside the repo, on a feature branch, with no push to main.",
  "the task is running the existing test suite and reading its output locally.",
  "the action writes only to a temporary directory inside this machine and touches nothing persisted."],
 "owner_must_decide": [
  "the action would widen a risk limit or increase position size - the model may advise, only the owner may authorise.",
  "the request is to spend money on a metered API beyond the ordinary cost of the work.",
  "the change alters which branch is the default on a shared repository.",
  "the decision is which of two conflicting sources of truth should win, where both have been edited."],
 "blocked_until_evidence": [
  "the action is permitted only after the failing case is reproduced and the failure output attached.",
  "the deploy is gated on a passing health probe and no probe has been run against the new revision.",
  "the migration may proceed only after a verified backup exists, and the backup has not been checked.",
  "the lane may be re-enabled only after a token send is observed to return content, which has not been done."],
 "forbidden": [
  "the action would edit the owner's source-of-truth document, which is under a standing byte-for-byte prohibition.",
  "the request is to push credentials or environment contents to a remote repository.",
  "the action would commit directly to main on a shared repository.",
  "the change would bypass an architecture boundary enforced by the import contract."],
},
}


def main() -> None:
    tax = json.load(open(os.path.join(HERE, "taxonomy_map.json")))
    by_id = {l["id"]: l for d in tax["domains"] for sr in d["sub_routers"] for l in sr["leaves"]}
    out_dir = os.path.join(HERE, "datasets"); os.makedirs(out_dir, exist_ok=True)
    added = 0
    for lid, data in LIB.items():
        leaf = by_id.get(lid)
        if leaf is None:
            print(f"  {lid}: NOT IN TAXONOMY, skipped"); continue
        opts = leaf["options"]
        rows = []
        for label, states in data.items():
            assert label in opts, f"{lid}: {label!r} is not a frozen option key"
            for st in states:
                rows.append({"state": st, "label": label, "option_text": opts[label]})
        counts = {k: sum(1 for r in rows if r["label"] == k) for k in opts}
        missing = [k for k in opts if counts[k] == 0]
        if missing:
            print(f"  {lid}: MISSING options {missing} - not written"); continue
        path = os.path.join(out_dir, f"{lid}.json")
        if os.path.exists(path):
            existing = len(json.load(open(path))["samples"])
            if existing >= len(rows):
                print(f"  {lid}: already has {existing} (>= {len(rows)}), skipped"); continue
        for b in range(0, len(rows), dh.BATCH_SIZE):
            dh.write_batch(lid, b // dh.BATCH_SIZE, rows[b:b + dh.BATCH_SIZE])
        json.dump({"id": lid, "kind": "leaf", "question": leaf["question"], "options": opts,
                   "label_counts": counts, "samples": rows}, open(path, "w"), indent=2)
        added += len(rows)
        print(f"  {lid:<10} {len(rows):4d} samples  balance={min(counts.values())}/{max(counts.values())}")
    print(f"  added: {added}")


if __name__ == "__main__":
    main()
