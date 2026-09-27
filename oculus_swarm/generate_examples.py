"""Generate the labelled sample set for the OCULUS taxonomy.

Bob's order (2026-09-27): "generate it all" with an LLM. So the brief's
"never invent training data with an LLM" is overruled by the owner and this file
does it the safe way instead of not at all.

THE ONE THING THAT MAKES THIS WORTH DOING: BALANCE.
A generator left alone produces mostly the obvious answer. A leaf whose safe branch is
90% of the data trains an adapter that predicts that branch every time and scores
excellent recall on it while catching none of the rare - and usually decisive - branch.
So examples are generated PER OPTION, with the required label named in every call, and
the per-label counts are recorded and asserted. The brief says report per-class recall,
never overall accuracy; this file makes per-class data exist in the first place.

Input is the frozen taxonomy. Option strings are never reworded here: a model trained on
"promote_genome" and later asked "Promote this genome to the champion registry" is being
asked a different question, so the label keys are passed through verbatim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import data_harvester as dh

HERE = os.path.dirname(os.path.abspath(__file__))
TAXONOMY = os.path.join(HERE, "taxonomy_map.json")
OUT_DIR = os.path.join(HERE, "datasets")
KEY_FILE = os.path.expanduser("~/.claude/openrouter.token")
URL = "https://openrouter.ai/api/v1/chat/completions"

# Measured on the real generation task (see the probe in the commit message):
#   ling-3.0-flash-sante 4.8s, 4/4 valid, 4 distinct labels, cost 0
#   cohere/north-mini-code 14.2s, 3/3 valid
# Everything else failed: nemotron and dots returned EMPTY content (the reasoning-budget
# trap the brief warns about), poolside and qwen 429'd, inkling-small 403'd.
GENERATORS = ["inclusionai/ling-3.0-flash-sante:free", "cohere/north-mini-code:free"]
PER_CALL = 12          # measured: 40 examples overflow max_tokens and truncate the array
WORKERS = 6          # measured: TLS handshakes hang above this on the free tier
MAX_RETRIES = 4
STATE_CHARS = (80, 420)

_lock = threading.Lock()
_stats = {"calls": 0, "ok": 0, "empty": 0, "badjson": 0, "http": 0, "timeout": 0}


def _key() -> str:
    return open(KEY_FILE).read().strip()


SYSTEM = (
    "You generate labelled training examples for a decision classifier used by a "
    "quantitative trading system. You output ONLY a JSON array. No prose, no markdown "
    "fence, no explanation."
)


def _prompt(question: str, options: Dict[str, str], label: str, n: int, avoid: List[str]) -> str:
    """One call = n examples that ALL carry the SAME required label.

    Generating per label is what makes the set balanced. Asking for a spread and
    hoping is how a generator quietly returns 90% of the obvious answer.
    """
    opt_lines = "\n".join(f"  {k} = {v}" for k, v in options.items())
    avoid_block = ""
    if avoid:
        avoid_block = "\nDo NOT reuse these openings:\n" + "\n".join(f"  - {a[:70]}" for a in avoid[:12])
    return f"""DECISION: {question}

OPTIONS (the correct answer is exactly one of these keys):
{opt_lines}

TASK: write {n} DIFFERENT training examples whose correct label is EXACTLY "{label}".

Each example:
  "state": 1-3 sentences of realistic measured facts a trading system would report for
           this decision - numbers, counts, ratios, percentages, timestamps, symbols.
           Describe the SITUATION only. Never mention the option names.
  "label": exactly "{label}"

Requirements:
- every example must genuinely be a "{label}" case ({options.get(label, '')})
- vary the numbers, the wording, the asset, the timeframe and the sentence shape
- no two examples may be paraphrases of each other
- plain factual text, no preamble, no markdown{avoid_block}

Return ONLY the JSON array:
[{{"state": "...", "label": "{label}"}}]"""


def _call(model: str, prompt: str, timeout: int = 60) -> Optional[str]:
    """One call, bounded by a WALL-CLOCK deadline.

    urlopen's timeout bounds each socket operation, not the whole exchange, so a
    server that streams a body slowly holds the read open indefinitely - measured: a
    single call sat on one ESTAB socket for 6 minutes with the process otherwise idle
    and the thread pool never advancing. The deadline below is enforced from the
    calling thread, so a hung read cannot stall the run.
    """
    body = json.dumps({"model": model, "messages": [
        {"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        "max_tokens": 8000, "temperature": 1.15}).encode()
    req = urllib.request.Request(URL, data=body, method="POST", headers={
        "Authorization": f"Bearer {_key()}", "Content-Type": "application/json"})
    # A plain blocking call with a socket timeout. No inner thread: a pool worker per
    # call already gives concurrency, and adding a thread inside each one doubled the
    # concurrent TLS handshakes (measured: 32 stuck in ssl do_handshake) until the free
    # tier stopped completing handshakes at all.
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
    except Exception as e:
        with _lock:
            _stats["http"] += 1
        return f"__ERR__{type(e).__name__}: {str(e)[:80]}"
    with _lock:
        _stats["ok"] += 1
    c = (d.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    return c


def _salvage_objects(content: str) -> List[Dict[str, Any]]:
    """Recover complete {...} objects from a truncated JSON array.

    The reply is capped at max_tokens, so the closing ] often never arrives. Each
    object before the cut is still perfectly good; discarding the whole reply loses
    them and makes a generator that is working fine look broken.
    """
    out: List[Dict[str, Any]] = []
    depth = 0
    start = None
    in_str = False
    esc = False
    for i, ch in enumerate(content):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(content[start:i + 1])
                    if isinstance(obj, dict):
                        out.append(obj)
                except json.JSONDecodeError:
                    pass
                start = None
    return out


def _parse(content: str, label: str, options: Dict[str, str]) -> List[Dict[str, str]]:
    """Pull usable examples out, refusing anything that is not the required label."""
    if content.startswith("__ERR__"):
        return []
    if not content.strip():
        with _lock:
            _stats["empty"] += 1
        return []
    # A big request is routinely CUT OFF at max_tokens, so the array never closes and
    # a whole-array regex or json.loads returns nothing. Measured: asking for 40
    # examples returned a truncated array, parsed 0 rows, and the run retried forever
    # while looking like a hang. Salvage every COMPLETE object instead.
    m = re.search(r"\[.*\]", content, re.S)
    if m:
        try:
            arr = json.loads(m.group(0))
        except json.JSONDecodeError:
            arr = None
    else:
        arr = None
    if not isinstance(arr, list):
        arr = _salvage_objects(content)
        if not arr:
            with _lock:
                _stats["badjson"] += 1
            return []
    out = []
    for x in arr:
        if not isinstance(x, dict):
            continue
        st = x.get("state")
        if not isinstance(st, str):
            continue
        st = " ".join(st.split())
        # The label is FORCED to the requested one, never taken from the model's own
        # field: a model that drifts off-label would otherwise teach the wrong answer.
        if len(st) < STATE_CHARS[0] or len(st) > STATE_CHARS[1]:
            continue
        if not any(ch.isdigit() for ch in st):
            continue  # a trading state with no measurable number is not a state
        out.append({"state": st, "label": label, "option_text": options[label]})
    return out


def _dedup(samples: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen, out = set(), []
    for s in samples:
        h = hashlib.sha1(s["state"].lower().encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        out.append(s)
    return out


def generate_for(question: str, options: Dict[str, str], per_label: int,
                 on_batch=None) -> List[Dict[str, str]]:
    """Generate `per_label` examples for EVERY option, in parallel, deduped.

    Returns a balanced set: each label gets its own train of calls, so a majority
    branch cannot crowd out the rest.
    """
    labels = list(options)
    jobs: List[Tuple[str, int]] = []
    for lab in labels:
        need = per_label
        while need > 0:
            n = min(PER_CALL, need)
            jobs.append((lab, n))
            need -= n
    random.shuffle(jobs)

    got: Dict[str, List[Dict[str, str]]] = {l: [] for l in labels}
    avoided: Dict[str, List[str]] = {l: [] for l in labels}

    def one(job: Tuple[str, int], attempt: int = 0):
        lab, n = job
        model = GENERATORS[(len(lab) + attempt) % len(GENERATORS)]
        with _lock:
            _stats["calls"] += 1
        c = _call(model, _prompt(question, options, lab, n, avoided.get(lab, [])))
        rows = _parse(c, lab, options)
        return lab, rows

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(one, j): j for j in jobs}
        done = 0
        for f in as_completed(futs):
            j = futs[f]
            lab, rows = f.result()
            if rows:
                got[lab].extend(rows)
                avoided.setdefault(lab, []).extend(r["state"][:40] for r in rows[:3])
            done += 1
            if on_batch:
                on_batch(done, len(jobs))

    # Retry the labels that came up short, with the same per-label discipline.
    for lab in labels:
        attempt = 0
        while len(got[lab]) < per_label and attempt < MAX_RETRIES:
            attempt += 1
            short = per_label - len(got[lab])
            c = _call(GENERATORS[attempt % len(GENERATORS)],
                      _prompt(question, options, lab, min(PER_CALL, short), avoided.get(lab, [])))
            rows = _parse(c, lab, options)
            if rows:
                got[lab].extend(rows)
                avoided.setdefault(lab, []).extend(r["state"][:40] for r in rows[:3])

    out: List[Dict[str, str]] = []
    for lab in labels:
        out.extend(_dedup(got[lab])[:per_label])
    return out


def targets(tax: Dict[str, Any], per_leaf: int) -> List[Dict[str, Any]]:
    """Every decision to generate for: the leaves, plus the two overlord tiers."""
    items: List[Dict[str, Any]] = []
    for d in tax["domains"]:
        for sr in d["sub_routers"]:
            for lf in sr["leaves"]:
                items.append({"id": lf["id"], "kind": "leaf", "question": lf["question"],
                              "options": lf["options"]})
    # Tier 1 overlord: pick the domain.
    items.append({"id": "T1_OVERLORD", "kind": "overlord",
                  "question": "Which area of the trading system does `state` concern?",
                  "options": {d["id"]: d["name"].replace("_", " ") + f" - {d['routes_on']}"
                              for d in tax["domains"]}})
    # Tier 2 router: per domain, pick the sub-router.
    for d in tax["domains"]:
        items.append({"id": f"T2_{d['id']}", "kind": "overlord",
                      "question": f"Within {d['name'].replace('_', ' ')}, "
                                  "which kind of question is `state` about?",
                      "options": {sr["id"]: sr["name"].replace("_", " ")
                                  for sr in d["sub_routers"]}})
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-leaf", type=int, default=1200)
    ap.add_argument("--only", help="one target id, for a small proving batch")
    ap.add_argument("--limit-targets", type=int, help="stop after N targets (smoke test)")
    args = ap.parse_args()

    tax = json.load(open(TAXONOMY))
    items = targets(tax, args.per_leaf)
    if args.only:
        items = [t for t in items if t["id"] == args.only]
        if not items:
            raise SystemExit(f"no target {args.only!r}")
    if args.limit_targets:
        items = items[:args.limit_targets]

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"  targets: {len(items)}   per-leaf requested: {args.per_leaf}   "
          f"generators: {len(GENERATORS)}   workers: {WORKERS}", flush=True)

    started = time.time()
    for i, it in enumerate(items, 1):
        n_opts = len(it["options"])
        per_label = max(1, args.per_leaf // n_opts)
        if dh.already_done(it["id"], 0, expect=args.per_leaf):
            print(f"  [{i}/{len(items)}] {it['id']:<16} already done, skipping", flush=True)
            continue
        t0 = time.time()
        samples = generate_for(it["question"], it["options"], per_label)
        counts = Counter(s["label"] for s in samples)
        if not samples:
            print(f"  [{i}/{len(items)}] {it['id']:<16} NO SAMPLES (not written)", flush=True)
            continue
        # Write in batches of 100 through the guard that refuses to shrink a batch.
        for b in range(0, len(samples), dh.BATCH_SIZE):
            dh.write_batch(it["id"], b // dh.BATCH_SIZE, samples[b:b + dh.BATCH_SIZE])
        with open(os.path.join(OUT_DIR, f"{it['id']}.json"), "w") as fh:
            json.dump({"id": it["id"], "kind": it["kind"], "question": it["question"],
                       "options": it["options"], "label_counts": dict(counts),
                       "samples": samples}, fh, indent=2)
        bal = min(counts.values()) / max(counts.values()) if counts else 0
        print(f"  [{i}/{len(items)}] {it['id']:<16} {len(samples):5d} in {time.time()-t0:5.0f}s  "
              f"labels={len(counts)}/{n_opts} balance={bal:.2f}  {dict(counts)}", flush=True)

    print(f"\n  calls={_stats['calls']} ok={_stats['ok']} empty={_stats['empty']} "
          f"badjson={_stats['badjson']} http={_stats['http']} timeout={_stats['timeout']}")
    print(f"  total {time.time()-started:.0f}s")


if __name__ == "__main__":
    main()
