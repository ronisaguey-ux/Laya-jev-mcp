"""Independent oracle for every trained adapter, through laya's own public inference path.

run: python probe_adapters.py            (all adapters present on disk)
     python probe_adapters.py D4.2.1

For each adapter it asks: on a state the adapter has NEVER SEEN, does attaching it change
the base model's choice, and in the direction the leaf exists to enforce? Every number the
training run printed came from the same scoring code that trained the adapter, which is
self-confirming evidence; this goes through agent.predict instead.
"""
from __future__ import annotations

import json
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import laya
from peft import PeftModel

# Hand-written states, deliberately absent from the datasets, phrased in the leaf's own terms.
UNSEEN = {
    "D9.2.2": (
        "The engine reported the step green. No command output was attached and the claim "
        "rests on the summary text alone. A commit exists but its diff was not read."),
    "D4.2.1": (
        "Portfolio drawdown has crossed the hard threshold the rules define as a stop and the "
        "book is still carrying the losing strategy. No operator has acknowledged it."),
}


def probe(leaf_id, agent):
    ds = json.load(open(f"datasets/{leaf_id}.json"))
    q = {"type": "choice", "instructions": ds["question"], "criteria": dict(ds["options"])}
    state = UNSEEN.get(leaf_id)
    if state is None:
        print(f"  {leaf_id}: no unseen state authored; skipped")
        return None
    base = agent.predict(state, {"probe": q})["answers"]["probe"]
    adapter = PeftModel.from_pretrained(agent.model, f"adapters/t3_{leaf_id}")
    adapted = agent.predict(state, {"probe": q})["answers"]["probe"]
    moved = base["choice"] != adapted["choice"]
    print(f"  {leaf_id}: base={base['choice']:<20} adapter={adapted['choice']:<20} "
          f"{'CHANGED' if moved else 'unchanged'}")
    print(f"      base    p={ {k: round(v,3) for k,v in base['probabilities'].items()} }")
    print(f"      adapter p={ {k: round(v,3) for k,v in adapted['probabilities'].items()} }")
    return moved


def main():
    want = sys.argv[1:] or sorted(
        f[len("t3_"):] for f in os.listdir("adapters") if os.path.isdir(f"adapters/{f}"))
    agent = laya.load()
    print(f"  base loaded; probing {len(want)} adapter(s)")
    for leaf_id in want:
        probe(leaf_id, agent)


if __name__ == "__main__":
    main()
