"""Hand-authored seed batch for the ORCHESTRATION tier.

Bob: "add more laya agents that are fine tuned to answer more sub task answers". These
are the sub-tasks his audit, cross-eval and execution agents ask, and I use them too.

Written directly, same as the trading seeds - no remote generator.
"""
from __future__ import annotations
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data_harvester as dh  # noqa: E402

D8_1_1 = {  # finding_actionable
 "actionable": [
  "audit finding: region.py reads a column by index 3 but the schema has no guarantee of that position; the writer is in the same file and can be fixed deterministically.",
  "audit finding: save_state uses json.dump without a temp file and os.replace, so a crash mid-write leaves a truncated state file. The fix is a two-line change at one call site.",
  "audit finding: three tests import a helper that was deleted; the helper has one remaining caller that was missed. Concrete, reproducible, single file.",
  "audit finding: the retry gate matches only the string 'Timed out' but the stall error says 'stalled: no new output', so the retry is unreachable. Provable by reading two lines.",
  "audit finding: a threshold is compared with > where the documented contract says >=, so the boundary case takes the wrong branch. One character.",
  "audit finding: the config loader reads a key that no other file defines and silently defaults it, so a typo in the config is never reported.",
  "audit finding: a lock file is written under /tmp with a fixed name shared by two accounts, so the second account can steal the first one's lock.",
  "audit finding: a metrics aggregator reads f['sharpe'] from a nested dict where the value lives under 'result', so it aggregates nothing and reports zero.",
 ],
 "noise": [
  "audit finding: line 412 exceeds 100 characters by 3 characters. No behaviour is affected.",
  "audit finding: a local variable is named data2. It is used twice within four lines and is perfectly clear in context.",
  "audit finding: the module imports datetime at the top and again inside one function. Both work; the inner import shadows nothing.",
  "audit finding: a docstring says 'returns a list' and the annotation is List[str]. They agree, so the finding is a false positive from a doc linter.",
  "audit finding: a test file lacks a module docstring. The test itself is correct and passing.",
  "audit finding: a comment is stale - it mentions a parameter that was removed. Nothing reads the comment.",
  "audit finding: whitespace-only change in a file the audit already scanned. No semantic difference.",
  "audit finding: a helper is defined but unused. It is public API re-exported for external callers and is referenced by docs.",
 ],
 "already_fixed": [
  "audit finding: security_utils.py is empty and atomic_json_write is missing. The file now exists with the function implemented and 4 tests pass against it.",
  "audit finding: the import chain fails on GENOME_SLOT_COUNT being undefined before use. The chain now imports cleanly at HEAD.",
  "audit finding: pytest dies during collection. Collection now completes with 7,298 tests and 4 errors on unrelated missing plugins.",
  "audit finding: git_status resolves repo paths one level too high. The path resolves correctly now and all three repos report their real branch.",
  "audit finding: the SoT integrity pin holds the sha1 of an empty string so it can never fire. The pin now validates with sha256sum -c returning OK.",
  "audit finding: a pre-commit hook is declared twice so it runs twice per commit. It is now declared once and the config parses.",
  "audit finding: README.md was truncated to 0 bytes by an auto-sweep. The file has content again at the restored revision.",
  "audit finding: a step retires landed work as yellow without reading the file. The retire path now reads the target first.",
 ],
 "needs_investigation": [
  "audit finding: a metric reads 0.0 on every run. It could be a genuine zero or a wiring bug; no caller can be found and no test covers it.",
  "audit finding: two modules define the same class name with different bodies. Which one is imported depends on a path choice that is not obvious.",
  "audit finding: a check has not fired in 30 days. It may be correct-and-quiet or dead code; there is no evidence either way yet.",
  "audit finding: the audit reports 39 blocked steps, all blocked by a solver that is disabled. Whether the blocker still applies is unverified.",
  "audit finding: a file appears in git as modified on every run. It may be a runtime artifact or a real edit; the diff is empty.",
  "audit finding: a lane is reported healthy while its gateway answers nothing. The health probe and the send path disagree.",
  "audit finding: a summary claims a test count that no command output supports. The claim needs checking against a real run before acting.",
  "audit finding: an untracked module is imported by tracked code, so a clone would break. Whether the module is truly needed is not yet established.",
 ],
}

D9_2_2 = {  # completion_truth
 "verified_done": [
  "Change driven end to end: the CLI launched an agent against a live webchat, the status bar read the webchat model, and the command output is attached to the report.",
  "The route was curled and returned 200 with the expected JSON body, and the same result is reproducible on a second call. Output pasted in the summary.",
  "Full suite run after the change: 348 of 348 passing, with the raw runner output attached and the failure count visible.",
  "The file the feature should have written exists, contains the expected rows, and was read back after the run completed.",
  "The page was rendered and the element asserted on by selector; a screenshot and the selector query result are both attached.",
  "The migration was applied to a copy of the database, the row counts match before and after, and both counts are in the report.",
  "The probe returned the exact token that was sent, on two separate calls, with the raw responses included.",
  "The adapter swap was exercised: three tiers gave three different answers and the output was captured from the live run.",
 ],
 "claimed_only": [
  "Agent reports: everything works now. No command output, no file read, no probe result is attached anywhere in the report.",
  "Agent reports: I have fixed the bug and verified it. The summary contains no command and no observed output.",
  "Agent reports: all tests pass. No runner output, no count, and no evidence a test command was executed.",
  "Agent reports: the feature is complete and working. Nothing in the transcript shows the feature being driven.",
  "Agent reports: I confirmed the fix. The only support is that the edit was applied, which is not a check of behaviour.",
  "Agent reports: deployed and healthy. No health response, no log line, and no request against the service is shown.",
  "Agent reports: the endpoint works. There is no request, response, or status code anywhere in the summary.",
  "Agent reports: verified end to end. The report names no command and shows no output from one.",
 ],
 "partially_done": [
  "The change was made and driven successfully for the primary path, but the error path and the empty-input path were never exercised.",
  "Two of the three promised files were updated; the third still holds the old implementation and was not mentioned.",
  "The feature works locally but the flag that enables it was never added to the config, so it cannot be turned on.",
  "The fix covers the main call site; two other sites still call the old function and were not updated.",
  "Tests were written for the happy path only; the regression the task was about has no test.",
  "The endpoint responds but the response omits a field the contract requires, so the caller cannot use it.",
  "The migration ran on the dev copy but the production-shaped config path was skipped.",
  "The change is committed but the branch was never pushed and the PR does not exist.",
 ],
 "skipped_verification": [
  "The change was applied and looks correct on reading. No test, probe or run was performed and the summary says so.",
  "The fix was made from reading the code alone; the failing case that motivated it was never reproduced.",
  "The suite was believed to be green because it was green before the change; it was not re-run afterwards.",
  "The feature was implemented and the tests were left for later. None have been run.",
  "A probe was attempted once and errored; the summary does not mention the attempt and reports the work as done.",
  "The change was verified by reading the diff rather than by running anything.",
  "The deploy was assumed healthy because the process started; no health check or request was made.",
  "The sample was generated and assumed balanced; the per-label counts were never printed.",
 ],
}

SEEDS = {
 "D8.1.1": D8_1_1,
 "D9.2.2": D9_2_2,
}


def main() -> None:
    tax = json.load(open(os.path.join(HERE, "taxonomy_map.json")))
    by_id = {l["id"]: l for d in tax["domains"] for sr in d["sub_routers"] for l in sr["leaves"]}
    out_dir = os.path.join(HERE, "datasets"); os.makedirs(out_dir, exist_ok=True)
    total = 0
    for lid, data in SEEDS.items():
        leaf = by_id[lid]; opts = leaf["options"]
        rows = []
        for label, states in data.items():
            assert label in opts, f"{lid}: {label!r} is not a frozen option key"
            for st in states:
                rows.append({"state": st, "label": label, "option_text": opts[label]})
        counts = {k: sum(1 for r in rows if r["label"] == k) for k in opts}
        assert all(counts[k] > 0 for k in opts), f"{lid}: an option got no examples: {counts}"
        for b in range(0, len(rows), dh.BATCH_SIZE):
            dh.write_batch(lid, b // dh.BATCH_SIZE, rows[b:b + dh.BATCH_SIZE])
        json.dump({"id": lid, "kind": "leaf", "question": leaf["question"], "options": opts,
                   "label_counts": counts, "samples": rows},
                  open(os.path.join(out_dir, f"{lid}.json"), "w"), indent=2)
        total += len(rows)
        print(f"  {lid}  {len(rows):4d} samples  balance={min(counts.values())}/{max(counts.values())}  {counts}")
    print(f"  orchestration seeds authored: {total}")


if __name__ == "__main__":
    main()
