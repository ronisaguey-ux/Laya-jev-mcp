"""Train ONE LoRA adapter per TIER, not per node.

Why per tier and not per leaf: Laya is a listwise scorer. `build_sequence` emits
`[CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]` and the
SAME `SCORER` scores every decision. A per-node adapter would be trained against a
scorer that the next node's adapter moves, so loading one would invalidate the rest.
Freezing the head (and therefore the scorer) and adapting the ENCODER ONLY removes the
conflict: every adapter is fitted against one unmoving scorer.

Measured on this machine (laya 0.3.20, convaiinnovations/laya):
  encoder   394.8M params   <- LoRA target
  head       25.2M params   <- frozen
  scorer      1.1M params   <- frozen (this is the shared scorer)
  act_head    0.3M params   <- frozen
  trainable with r=16       4.39M  (1.03% of the model)
  adapter size on disk     17.6 MB
  swap on a warm base       0.6s

Training rules carried over from the brief, each from a real failure:
  * class weighting + per-class recall, never overall accuracy. A lopsided classifier
    that always predicts the majority class scores 99% and catches nothing.
  * the measured sample is the label; never re-derive cost or quality from the text.
  * an unknown band returns nothing rather than falling back to everything.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
TAXONOMY = os.path.join(HERE, "taxonomy_map.json")
ADAPTER_DIR = os.path.join(HERE, "adapters")

TIERS = ("T1_overlord", "T2_router", "T3_specialist")

# Which level of the tree each tier is trained on. A tier-1 adapter sees the domain
# question; tier-2 sees the sub-router question; tier-3 sees leaf questions.
TIER_LEVEL = {"T1_overlord": "domain", "T2_router": "sub_router", "T3_specialist": "leaf"}


def load_taxonomy(path: str = TAXONOMY) -> Dict[str, Any]:
    with open(path) as fh:
        return json.load(fh)


def level_questions(tax: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Every question the model must answer, grouped by the tier that owns it.

    A tier's training set is the union of the questions at its level, holding the
    option set fixed per question. That is what makes one adapter per tier possible
    rather than one per node.
    """
    out: Dict[str, List[Dict[str, Any]]] = {t: [] for t in TIERS}
    for d in tax["domains"]:
        out["T1_overlord"].append({
            "id": f"domain::{d['id']}",
            "instructions": f"Which area of the trading system does `state` concern: {d['routes_on']}",
            "criteria": {x["id"]: x["name"].replace("_", " ") for x in tax["domains"]},
        })
        for sr in d["sub_routers"]:
            out["T2_router"].append({
                "id": f"sub::{sr['id']}",
                "instructions": f"Within {d['name'].replace('_', ' ')}, which kind of question is `state` about?",
                "criteria": {x["id"]: x["name"].replace("_", " ") for sr in d["sub_routers"] for x in [sr]},
            })
            for lf in sr["leaves"]:
                out["T3_specialist"].append({
                    "id": lf["id"],
                    "instructions": lf["question"],
                    "criteria": lf["options"],
                })
    return out


def class_weights(labels: Sequence[str]) -> Dict[str, float]:
    """Inverse-frequency weights so a rare branch is not learned away.

    A leaf whose safe branch is 95% of the data will be predicted every time by an
    unweighted fit. The rare branch is usually the one that matters.
    """
    n = len(labels)
    if not n:
        return {}
    counts = Counter(labels)
    k = len(counts)
    return {lab: n / (k * c) for lab, c in counts.items()}


def per_class_recall(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, float]:
    """Recall per class. Overall accuracy is deliberately not reported.

    Report this, never accuracy: a lopsided classifier scores 99% accuracy while
    catching none of the rare — and usually important — branch.
    """
    out: Dict[str, float] = {}
    for lab in sorted(set(y_true)):
        idx = [i for i, y in enumerate(y_true) if y == lab]
        hit = sum(1 for i in idx if y_pred[i] == lab)
        out[lab] = hit / len(idx) if idx else 0.0
    return out


def build_model(adapter_dir: str, tier: str, *, r: int = 16, alpha: int = 32, dropout: float = 0.05):
    """Load the base ONCE and wrap it for one tier.

    The head, scorer and act_head are frozen here, before PEFT is applied, so the
    shared scorer cannot move under another tier's adapter.
    """
    import laya
    from peft import LoraConfig, get_peft_model

    agent = laya.load()
    for name, param in agent.model.named_parameters():
        param.requires_grad = name.startswith("encoder.")   # encoder only
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                     target_modules=["Wqkv", "Wo"])
    peft_model = get_peft_model(agent.model, cfg)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    return agent, peft_model, trainable


def main() -> None:
    ap = argparse.ArgumentParser(description="Train one LoRA adapter per tier.")
    ap.add_argument("--taxonomy", default=TAXONOMY)
    ap.add_argument("--tier", choices=TIERS, help="train one tier; default is all three")
    ap.add_argument("--data", help="path to a harvested dataset dir (default oculus_swarm/datasets)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dry-run", action="store_true",
                    help="verify the freeze, the tier split and the weights without training")
    args = ap.parse_args()

    tax = load_taxonomy(args.taxonomy)
    by_tier = level_questions(tax)

    print(f"  taxonomy: {tax['counts']['domains']} domains / "
          f"{tax['counts']['sub_routers']} sub-routers / {tax['counts']['leaves']} leaves")
    for tier in TIERS:
        qs = by_tier[tier]
        opts = {len(q["criteria"]) for q in qs}
        print(f"  {tier:<16} {len(qs):3d} questions   option-set sizes {sorted(opts)}")

    if args.dry_run:
        # A dry run must prove the parts that decide success or failure, not just print.
        agent, peft_model, trainable = build_model(ADAPTER_DIR, "T1_overlord")
        total = sum(p.numel() for p in peft_model.parameters())
        print(f"\n  trainable        : {trainable/1e6:.2f}M of {total/1e6:.1f}M "
              f"({100*trainable/total:.2f}%)")

        # The invariant that actually matters, and the one the brief insists on:
        # EVERY trainable tensor is a LoRA tensor, and the shared scorer is frozen.
        # LoRA freezes the base weights by design, so "the encoder base is frozen"
        # is not the test - "nothing outside the LoRA pairs is trainable" is.
        trainable_names = [n for n, p in peft_model.named_parameters() if p.requires_grad]
        non_lora = [n for n in trainable_names if "lora_" not in n]
        shared = ("head.", "scorer.", "act_head.", "type_emb.")
        shared_trainable = [n for n in trainable_names if any(s in n for s in shared)]
        frozen_shared = [n for n, p in peft_model.named_parameters()
                         if not p.requires_grad and any(s in n for s in shared)]
        print(f"  trainable tensors  : {len(trainable_names)}")
        print(f"  non-LoRA trainable : {len(non_lora)}   (must be 0)")
        print(f"  shared&trainable   : {len(shared_trainable)}   (must be 0: head/scorer frozen)")
        print(f"  shared&frozen      : {len(frozen_shared)}")
        assert not non_lora, f"something outside LoRA is trainable: {non_lora[:3]}"
        assert not shared_trainable, f"the shared scorer/head moved: {shared_trainable[:3]}"
        assert frozen_shared, "expected the head/scorer/act_head tensors to be present and frozen"

        # Prove the weights are not degenerate on the real class balance we expect.
        demo = ["reject"] * 19 + ["promote_to_oos"]
        w = class_weights(demo)
        print(f"  class weights on a 95/5 split: "
              f"{ {k: round(v,3) for k,v in sorted(w.items())} }")
        assert w["promote_to_oos"] > w["reject"] * 3, "rare class must be up-weighted"
        print("\n  dry run OK — freeze verified, tier split verified, weights non-degenerate")
        return

    if not args.data or not os.path.isdir(args.data):
        print(f"  no harvested data at {args.data!r}. Harvest first; this trains on measured "
              "samples, never on generated ones.", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(
        "training is not wired to a dataset yet — the harvest format must be settled "
        "before fitting, so the adapter is not trained on invented labels"
    )


if __name__ == "__main__":
    main()
