"""Train one LoRA adapter per tier on the authored seeds, and prove it moved the number.

Deliverable 4 from the brief. One adapter per TIER, LoRA on the ENCODER only, head and
scorer frozen - the shared scorer is what makes per-tier possible, and it is frozen
before PEFT is applied.

The measurement that matters is PER-CLASS RECALL, never overall accuracy. A lopsided
classifier scores well overall while catching none of the rare - and decisive - branch.
The base model catches 1/8 of `skipped_verification` on D9.2.2 (completion_truth), the
leaf that exists to catch a false "done" claim. That is the number to move.

SEQUENCES COME FROM laya.common.build_sequence, NOT FROM A LOCAL RE-IMPLEMENTATION.
A hand-rolled builder is a second, silently-diverging definition of the input format:
it omits the `head_max_len=192` option budget (so the options push the state out of the
window), tokenizes the instructions where laya tokenizes `"<type> question: <ins>"`,
and renders options without their label. Training on that format teaches the adapter a
prompt the model will never be given at inference.
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
TAXONOMY = os.path.join(HERE, "taxonomy_map.json")
DATA = os.environ.get("TRAIN_DATA_DIR") or os.path.join(HERE, "datasets")
ADAPTERS = os.path.join(HERE, "adapters")

BATCH = 16         # training: measured on this box, 2.44s/row at 16 vs 13.44s at 1
EVAL_BATCH = 16    # eval: no backward pass, so batch just as wide
MAX_LEN = 448
HEAD_MAX_LEN = 192

TIER_OF = {}


def load_taxonomy():
    tax = json.load(open(TAXONOMY))
    for d in tax["domains"]:
        for sr in d["sub_routers"]:
            for lf in sr["leaves"]:
                TIER_OF[lf["id"]] = "T3_specialist"
    return tax


def load_leaf(lid):
    p = os.path.join(DATA, f"{lid}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def available_leaves():
    out = []
    for f in sorted(os.listdir(DATA)):
        if not f.endswith(".json"):
            continue
        try:
            d = json.load(open(os.path.join(DATA, f)))
            if d.get("samples"):
                out.append(d)
        except Exception:
            pass
    return out


def as_question(leaf):
    """The dataset's frozen question + options in laya's own question-dict shape.

    `crit` is a dict keyed by OPTION LABEL; laya renders it as "<label>: <description>"
    and scores at the mask position of each, in that key order. The key order IS the
    label order, so it is never sorted here - the frozen taxonomy defines it.
    """
    return {"t": "choice", "ins": leaf["question"], "crit": dict(leaf["options"])}


def encode(tok, leaf, state):
    """One row via laya's own builder. Returns (ids, markers) with len(markers)==n options."""
    from laya.common import build_sequence
    return build_sequence(tok, state, as_question(leaf), max_len=MAX_LEN,
                          head_max_len=HEAD_MAX_LEN)


def make_batch(tok, rows, device):
    """rows: [(leaf, state, label)]. Pads to a rectangle and builds the marker tensors."""
    seqs, marks, targs = [], [], []
    for leaf, state, label in rows:
        ids, mk = encode(tok, leaf, state)
        labels = list(leaf["options"].keys())
        if label not in labels:
            continue
        seqs.append(ids)
        marks.append(mk)
        targs.append(labels.index(label))
    if not seqs:
        return None
    maxn = max(len(m) for m in marks)
    ids = torch.zeros(len(seqs), max(max(len(s) for s in seqs), maxn + 1), dtype=torch.long)
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


def score(model, ids, att, mp, mm, qtype, device):
    """(logits, act_logits) at the marker positions. laya masks pad slots to -1e4 itself."""
    out = model(input_ids=ids, attention_mask=att, marker_pos=mp, marker_mask=mm,
                qtype=torch.full((ids.shape[0],), qtype, dtype=torch.long, device=device))
    return out[0] if isinstance(out, tuple) else out


def recall(agent, model, tok, leaf, device, samples=None):
    """Per-class recall - the only metric worth reporting for a safety leaf."""
    labels = list(leaf["options"].keys())
    hits = defaultdict(lambda: [0, 0])
    if samples is None:
        samples = leaf["samples"]
    model.eval()
    with torch.no_grad():
        for i in range(0, len(samples), EVAL_BATCH):
            chunk = samples[i:i + EVAL_BATCH]
            built = make_batch(tok, [(leaf, s["state"], s["label"]) for s in chunk], device)
            if built is None:
                continue
            ids, att, mp, mm, _targ, maxn = built
            lg = score(model, ids, att, mp, mm, 0, device)
            for j, s in enumerate(chunk):
                pred = labels[int(torch.argmax(lg[j][:len(labels)]))]
                hits[s["label"]][0] += 1
                if pred == s["label"]:
                    hits[s["label"]][1] += 1
    model.train()
    return {k: (v[1], v[0], v[1] / v[0] if v[0] else 0.0) for k, v in hits.items()}


def fit(rows, tok, model, epochs, lr, device):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    for ep in range(epochs):
        random.shuffle(rows)
        tot, nb = 0.0, 0
        for i in range(0, len(rows), BATCH):
            built = make_batch(tok, rows[i:i + BATCH], device)
            if built is None:
                continue
            ids, att, mp, mm, targ, maxn = built
            lg = score(model, ids, att, mp, mm, 0, device)[:, :maxn]
            loss = F.cross_entropy(lg, targ)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += float(loss.detach()); nb += 1
        print(f"    epoch {ep+1}/{epochs}  loss={tot/max(1,nb):.4f}", flush=True)


def stratified_split(samples, frac, seed=1337):
    """Hold out `frac` of EVERY class. A split that empties a class cannot measure it."""
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for s in samples:
        by_label[s["label"]].append(s)
    train, test = [], []
    for label in sorted(by_label):
        rows = by_label[label][:]
        rng.shuffle(rows)
        k = max(1, int(round(len(rows) * frac)))
        k = min(k, len(rows) - 1) if len(rows) > 1 else 0
        test.extend(rows[:k])
        train.extend(rows[k:])
    rng.shuffle(train); rng.shuffle(test)
    return train, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leaf", default="D9.2.2")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--no-base", action="store_true",
                    help="skip the base-model eval; it is 2 passes over the held-out set")
    ap.add_argument("--data-dir", default=None,
                    help="dataset directory; defaults to datasets/, use datasets_tier1 or datasets_tier2 for a tier")
    ap.add_argument("--holdout", type=float, default=0.25,
                    help="fraction of EACH class held out; 0 trains on everything")
    args = ap.parse_args()
    if args.data_dir:
        globals()["DATA"] = args.data_dir

    torch.set_num_threads(args.threads)
    import laya
    from peft import LoraConfig, get_peft_model

    load_taxonomy()
    leaf = load_leaf(args.leaf)
    if leaf is None:
        print(f"  no dataset for {args.leaf}")
        return 1
    print(f"  leaf {leaf['id']}: {len(leaf['samples'])} samples, {leaf['label_counts']}")

    # Format is laya's, so prove it before spending an epoch: same builder as inference.
    from laya.common import build_sequence  # noqa: F401
    print(f"  sequence builder: laya.common.build_sequence (max_len={MAX_LEN}, head={HEAD_MAX_LEN})")

    agent = laya.load()
    device = getattr(agent, "device", "cpu")
    tok = agent.tok
    base = agent.model

    tr_rows, te_rows = stratified_split(leaf["samples"], args.holdout)
    if args.holdout <= 0:
        tr_rows, te_rows = leaf["samples"], []
        print("\n  NO HOLDOUT: every sample is trained on, so this measures memorisation only")
    else:
        print(f"\n  stratified split: {len(tr_rows)} train / {len(te_rows)} held out "
              f"({args.holdout:.0%} of every class)")

    # Captured BEFORE get_peft_model: it rewrites `base` in place, so a read taken after
    # the attach reports the adapter against itself.
    if args.no_base:
        base_tr, base_te = {}, {}
        print("\n  base eval skipped (--no-base)")
    else:
        base_tr = recall(agent, base, tok, leaf, device, samples=tr_rows)
        base_te = recall(agent, base, tok, leaf, device, samples=te_rows) if te_rows else {}
    print("\n  BEFORE (base model):")
    for name, got in (("train", base_tr), ("held-out", base_te)):
        if not got:
            continue
        hit = sum(v[0] for v in got.values()); n = sum(v[1] for v in got.values())
        print(f"    {name:<9} {hit}/{n} = {hit/max(1,n):.2f}")
        for k in sorted(got):
            c, nn, r = got[k]
            print(f"      {k:<22} {c}/{nn} = {r:.2f}")

    for n, p in base.named_parameters():
        p.requires_grad = n.startswith("encoder.")
    if hasattr(base.encoder, "gradient_checkpointing_enable"):
        base.encoder.gradient_checkpointing_enable()
        print("  encoder gradient checkpointing: on")

    peft_model = get_peft_model(base, LoraConfig(
        r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.0,
        target_modules=["Wqkv", "Wo"]))
    tr = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    print(f"  LoRA trainable: {tr/1e6:.2f}M")

    before = recall(agent, base, tok, leaf, device)
    print("\n  BEFORE (base model), per-class recall:")
    for k, (c, n, r) in before.items():
        print(f"    {k:<22} {c}/{n} = {r:.2f}")

    rows = [(leaf, s["state"], s["label"]) for s in tr_rows]
    print(f"  training on {len(rows)} rows, {args.epochs} epochs, batch {BATCH}")
    fit(rows, tok, peft_model, args.epochs, args.lr, device)

    # Save BEFORE the eval. T1 trained three epochs to convergence and then died in its
    # final scoring pass, losing the whole run because the save sat after it. The eval is
    # the expensive and least important half; the weights are the deliverable.
    os.makedirs(ADAPTERS, exist_ok=True)
    out = os.path.join(ADAPTERS, f"t3_{leaf['id']}")
    peft_model.save_pretrained(out)
    size = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out))
    print(f"  adapter saved BEFORE the eval: {out}  ({size/1e6:.1f} MB)")

    print("\n  AFTER (adapter) - per-class recall:")
    moved = 0
    for label_name, sample_set in (("TRAIN (seen)", tr_rows), ("HELD-OUT (never trained on)", te_rows)):
        if not sample_set:
            continue
        got = recall(agent, peft_model, tok, leaf, device, samples=sample_set)
        base_rows = base_tr if label_name.startswith("TRAIN") else base_te
        total_hit = sum(v[0] for v in got.values()); total_n = sum(v[1] for v in got.values())
        print(f"\n    {label_name}  ({total_hit}/{total_n} = "
              f"{total_hit/max(1,total_n):.2f} overall)")
        for k in sorted(got):
            c, n, r = got[k]
            b = base_rows.get(k, (0, n, 0.0))[2]
            if label_name.startswith("HELD") and r > b:
                moved += 1
            print(f"      {k:<22} {c}/{n} = {r:.2f}   base was {b:.2f}  "
                  f"{'UP' if r > b else ('same' if r == b else 'DOWN')}")
    print(f"\n  HELD-OUT classes improved: {moved}")

    os.makedirs(ADAPTERS, exist_ok=True)
    out = os.path.join(ADAPTERS, f"t3_{leaf['id']}")
    peft_model.save_pretrained(out)
    size = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out))
    print(f"  adapter saved: {out}  ({size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
