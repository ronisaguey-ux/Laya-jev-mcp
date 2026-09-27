"""Train ONE adapter per TIER, which is what the brief actually specifies.

I had been training one adapter per LEAF - 54 runs at ~40 minutes each, which is not the
design and does not fit the machine. The taxonomy is 10 domains -> 19 sub-routers -> 54
leaves, so there are three tiers and therefore three adapters:

    T1_domain     10 questions   (which domain does this state belong to)
    T2_router     19 questions   (which sub-router inside the domain)
    T3_specialist 54 questions   (the leaf question itself, all leaves pooled)

Tier 3 is trained here. One adapter covers all 54 leaf questions because the decision head
scores every option at its own mask position with ONE shared scorer - that is exactly what
`laya.common.build_sequence` produces, and it is why per-tier works at all.

The split is stratified BY LEAF, not by label: every leaf must appear on both sides, or a
leaf's held-out recall is measured on questions the adapter never saw and cannot be
attributed to the leaf.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

DATA = "datasets"
ADAPTERS = "adapters"
BATCH = 16         # measured 2.44s/row at 16 vs 13.44s at 1 on this box
EVAL_BATCH = 16    # eval: no backward, so batch wide; 298 rows becomes 19 forwards, not 75
MAX_LEN = 448
HEAD_MAX_LEN = 192


def load_all_leaves():
    leaves = []
    for f in sorted(os.listdir(DATA)):
        if not f.endswith(".json"):
            continue
        d = json.load(open(os.path.join(DATA, f)))
        if d.get("samples"):
            leaves.append(d)
    return leaves


def split_by_leaf(leaves, frac, seed=1337):
    """Hold out `frac` of EVERY leaf, so each leaf's held-out recall is attributable."""
    rng = random.Random(seed)
    train, test = [], []
    for d in leaves:
        rows = [(d, s["state"], s["label"]) for s in d["samples"]]
        rng.shuffle(rows)
        k = max(1, int(round(len(rows) * frac)))
        k = min(k, len(rows) - 1) if len(rows) > 1 else 0
        test.extend(rows[:k])
        train.extend(rows[k:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def as_question(leaf):
    return {"t": "choice", "ins": leaf["question"], "crit": dict(leaf["options"])}


def make_batch(tok, rows, device):
    from laya.common import build_sequence
    seqs, marks, targs = [], [], []
    for leaf, state, label in rows:
        labels = list(leaf["options"].keys())
        if label not in labels:
            continue
        ids, mk = build_sequence(tok, state, as_question(leaf),
                                 max_len=MAX_LEN, head_max_len=HEAD_MAX_LEN)
        seqs.append(ids)
        marks.append(mk)
        targs.append(labels.index(label))
    if not seqs:
        return None
    maxn = max(len(m) for m in marks)
    L = max(max(len(s) for s in seqs), maxn + 1)
    ids = torch.zeros(len(seqs), L, dtype=torch.long)
    att = torch.zeros_like(ids)
    mp = torch.full((len(seqs), maxn), -1, dtype=torch.long)
    mm = torch.zeros((len(seqs), maxn), dtype=torch.bool)
    for i, (s, m) in enumerate(zip(seqs, marks)):
        ids[i, :len(s)] = torch.tensor(s)
        att[i, :len(s)] = 1
        mp[i, :len(m)] = torch.tensor(m)
        mm[i, :len(m)] = True
    return (ids.to(device), att.to(device), mp.to(device), mm.to(device),
            torch.tensor(targs, device=device), maxn)


def score(model, ids, att, mp, mm, device):
    out = model(input_ids=ids, attention_mask=att, marker_pos=mp, marker_mask=mm,
                qtype=torch.zeros(ids.shape[0], dtype=torch.long, device=device))
    return out[0] if isinstance(out, tuple) else out


def per_leaf_recall(model, tok, rows, device):
    """Held-out recall per leaf, and the leaf's overall - the number that means something."""
    model.eval()
    by_leaf = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    with torch.no_grad():
        for i in range(0, len(rows), EVAL_BATCH):
            chunk = rows[i:i + EVAL_BATCH]
            built = make_batch(tok, chunk, device)
            if built is None:
                continue
            ids, att, mp, mm, _t, _n = built
            lg = score(model, ids, att, mp, mm, device)
            for j, (leaf, _state, label) in enumerate(chunk):
                labels = list(leaf["options"].keys())
                pred = labels[int(torch.argmax(lg[j][:len(labels)]))]
                by_leaf[leaf["id"]][label][0] += 1
                if pred == label:
                    by_leaf[leaf["id"]][label][1] += 1
    model.train()
    return by_leaf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="T3_specialist")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--holdout", type=float, default=0.25)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--trace", action="store_true", help="print per-leaf held-out recall")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    import laya
    from peft import LoraConfig, get_peft_model

    leaves = load_all_leaves()
    tr, te = split_by_leaf(leaves, args.holdout)
    print(f"  tier {args.tier}: {len(leaves)} leaves, {len(tr)} train / {len(te)} held out")

    agent = laya.load()
    device = getattr(agent, "device", "cpu")
    tok = agent.tok
    base = agent.model

    before = per_leaf_recall(base, tok, te, device)
    b_hit = sum(v[1] for lf in before.values() for v in lf.values())
    b_n = sum(v[0] for lf in before.values() for v in lf.values())
    print(f"  base held-out overall: {b_hit}/{b_n} = {b_hit/max(1,b_n):.3f}")

    for n, p in base.named_parameters():
        p.requires_grad = n.startswith("encoder.")
    if hasattr(base.encoder, "gradient_checkpointing_enable"):
        base.encoder.gradient_checkpointing_enable()
    model = get_peft_model(base, LoraConfig(r=args.rank, lora_alpha=args.rank * 2,
                                            lora_dropout=0.0,
                                            target_modules=["Wqkv", "Wo"]))
    tr_params = [p for p in model.parameters() if p.requires_grad]
    print(f"  LoRA trainable: {sum(p.numel() for p in tr_params)/1e6:.2f}M")

    opt = torch.optim.AdamW(tr_params, lr=args.lr)
    for ep in range(args.epochs):
        random.shuffle(tr)
        tot, nb = 0.0, 0
        for i in range(0, len(tr), BATCH):
            built = make_batch(tok, tr[i:i + BATCH], device)
            if built is None:
                continue
            ids, att, mp, mm, targ, maxn = built
            lg = score(model, ids, att, mp, mm, device)[:, :maxn]
            loss = F.cross_entropy(lg, targ)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tr_params, 1.0)
            opt.step()
            tot += float(loss.detach()); nb += 1
        print(f"    epoch {ep+1}/{args.epochs}  loss={tot/max(1,nb):.4f}", flush=True)

    after = per_leaf_recall(model, tok, te, device)
    a_hit = sum(v[1] for lf in after.values() for v in lf.values())
    a_n = sum(v[0] for lf in after.values() for v in lf.values())
    print(f"  adapter held-out overall: {a_hit}/{a_n} = {a_hit/max(1,a_n):.3f}  "
          f"(base {b_hit/max(1,b_n):.3f})")
    imp, deg = [], []
    for lid in sorted(after):
        ah = sum(v[1] for v in after[lid].values()); an = sum(v[0] for v in after[lid].values())
        bh = sum(v[1] for v in before.get(lid, {}).values())
        bn = sum(v[0] for v in before.get(lid, {}).values())
        if an and bn:
            (imp if ah / an > bh / bn else deg).append((lid, bh / bn, ah / an))
    print(f"  leaves improved: {len(imp)}  degraded: {len(deg)}")
    if args.trace:
        for lid, b, a in deg[:15]:
            print(f"    DEGRADED {lid:<10} {b:.2f} -> {a:.2f}")

    os.makedirs(ADAPTERS, exist_ok=True)
    out = os.path.join(ADAPTERS, args.tier)
    model.save_pretrained(out)
    size = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out))
    print(f"  adapter saved: {out}  ({size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
