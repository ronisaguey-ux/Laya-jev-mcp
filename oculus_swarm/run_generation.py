"""Detached driver: generate every target, one at a time, logging progress."""
import json, os, sys, time
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_examples as g
import data_harvester as dh

PER_LEAF = int(os.environ.get("PER_LEAF", "1200"))
tax = json.load(open(g.TAXONOMY))
items = g.targets(tax, PER_LEAF)
os.makedirs(g.OUT_DIR, exist_ok=True)
print(f"targets={len(items)} per_leaf={PER_LEAF} workers={g.WORKERS}", flush=True)
total = 0
t_all = time.time()
for i, it in enumerate(items, 1):
    out = os.path.join(g.OUT_DIR, f"{it['id']}.json")
    if os.path.exists(out):
        try:
            if len(json.load(open(out)).get("samples", [])) >= PER_LEAF:
                print(f"[{i}/{len(items)}] {it['id']:<16} SKIP complete", flush=True)
                total += PER_LEAF
                continue
        except Exception:
            pass
    per_label = max(1, PER_LEAF // len(it["options"]))
    t0 = time.time()
    try:
        rows = g.generate_for(it["question"], it["options"], per_label)
    except Exception as e:
        print(f"[{i}/{len(items)}] {it['id']:<16} ERROR {type(e).__name__}: {str(e)[:70]}", flush=True)
        continue
    if not rows:
        print(f"[{i}/{len(items)}] {it['id']:<16} NO SAMPLES", flush=True)
        continue
    counts = Counter(r["label"] for r in rows)
    for b in range(0, len(rows), dh.BATCH_SIZE):
        dh.write_batch(it["id"], b // dh.BATCH_SIZE, rows[b:b + dh.BATCH_SIZE])
    json.dump({"id": it["id"], "kind": it["kind"], "question": it["question"],
               "options": it["options"], "label_counts": dict(counts), "samples": rows},
              open(out, "w"), indent=2)
    total += len(rows)
    eta = (len(items) - i) * (time.time() - t_all) / max(1, i)
    print(f"[{i}/{len(items)}] {it['id']:<16} {len(rows):5d} in {time.time()-t0:4.0f}s "
          f"bal={min(counts.values())/max(counts.values()):.2f} total={total} eta={eta/60:.0f}min {dict(counts)}", flush=True)
print(f"DONE {total} samples in {(time.time()-t_all)/60:.0f} min", flush=True)
