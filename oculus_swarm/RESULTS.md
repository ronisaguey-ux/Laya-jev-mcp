# Measured results

Every number here was produced on this machine by a command in this directory. Where a
number is too small to mean anything, it says so rather than rounding up.

## Base model vs fine-tuned adapter, held-out data

Held-out means a stratified split of every class, trained on the rest. The base column is
the same split scored before any adapter was attached.

| leaf | what it decides | held-out base | held-out adapter |
|---|---|---|---|
| D9.2.2 | is the work actually finished | 2/8 = 0.25 | 5/8 = 0.63 |
| D4.2.1 | should the kill switch fire | 1/5 = 0.20 | 4/5 = 0.80 |
| D1.2.2 | replace the champion | 2/12 = 0.17 | 6/12 = 0.50 |

Per class, held-out:

**D9.2.2 completion truth** - the leaf that exists to catch a false "done" claim.
```
claimed_only         0.00 -> 1.00
partially_done       0.00 -> 1.00
skipped_verification 0.00 -> 0.50
verified_done        1.00 -> 0.00   OVER-CORRECTED
```

**D4.2.1 kill switch** - money and irreversible.
```
fire             0.00 -> 1.00
escalate_human   0.00 -> 1.00
do_not_fire      1.00 -> 1.00
arm_only         0.00 -> 0.00       one held-out sample, not a measurement
```

**D1.2.2 champion replacement** - money and irreversible.
```
keep_incumbent   0.00 -> 1.00
replace          0.00 -> 0.33
escalate_human   0.00 -> 0.00
run_both         0.67 -> 0.67
```

## Independent oracle

`probe_adapters.py` runs each adapter through laya's own `agent.predict` on a hand-written
state that appears in no dataset. This is not the training loop scoring itself.

**D4.2.1**, state: *"Portfolio drawdown has crossed the hard threshold the rules define as
a stop and the book is still carrying the losing strategy. No operator has acknowledged it."*
```
base     do_not_fire      fire 0.150  arm_only 0.202  do_not_fire 0.487  escalate_human 0.161
adapter  escalate_human  fire 0.058  arm_only 0.343  do_not_fire 0.075  escalate_human 0.525
```
The base model read a breached, unacknowledged drawdown limit as "do not fire" and would
have let the book keep trading. The adapter escalates and cuts do-not-fire to 0.075.

**D9.2.2**, state: *"The engine reported the step green. No command output was attached and
the claim rests on the summary text alone. A commit exists but its diff was not read."*
```
base     claimed_only      verified_done 0.198  claimed_only 0.505  partially_done 0.053  skipped_verification 0.244
adapter  partially_done   verified_done 0.000  claimed_only 0.476  partially_done 0.510  skipped_verification 0.014
```
A green claim with no attached output moves off "done" entirely - verified_done falls to
zero.

## What is not yet measured

- The tier-3 pooled adapter. Training one adapter across all 54 questions is the GENERAL
  shape, and Laya is narrow-task only, so that run was stopped rather than finished.
- 23 of 54 leaves still sit at 2 examples per class, which cannot hold a stratified
  holdout. They are listed by `train_on_dataset.py --help`'s sibling check below.
- Three leaves have no usable signal in the base model on at least one branch
  (`escalate_human` is 0/12 in D1.2.2, `resume_reduced` is 0/10 in D4.2.3,
  `verified_done` recovered to 0/2 in D9.2.2). An adapter cannot learn a branch the base
  never predicts; those branches need more examples, not more epochs.

## Environment, because it bounds everything above

- 8 CPUs, ~6.8GB free RAM, torch 2.13.0+cpu, python 3.14.4.
- Measured forward cost, same sequence and model: batch 1 = 13.44s/row, batch 8 = 3.08s/row,
  batch 16 = 2.44s/row. Training runs at batch 16 for that reason.
- One narrow adapter is roughly 10-40 minutes depending on how many other processes are
  competing. The kimi chrome was burning 87% of a core for 2.6 days before it was stopped;
  load fell from 20 to 7.
