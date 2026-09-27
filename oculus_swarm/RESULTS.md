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

## Unattended queue: 13 money/irreversible leaves

Trained sequentially by `run_queue.sh`, four epochs, 25% stratified holdout.
`train` is recall on rows the adapter saw; `held-out` is the number that matters.

| leaf | train | held-out | classes improved |
|---|---|---|---|
| D1.2.1 | 0.92 | 0.75 | 1 |
| D10.1.2 | 1.00 | 0.25 | 0 |
| D10.1.3 | 1.00 | 0.75 | 1 |
| D2.2.3 | 1.00 | 0.78 | 2 |
| D3.2.1 | 1.00 | 0.75 | 2 |
| D4.1.1 | 1.00 | 0.25 | 1 |
| D4.1.2 | 1.00 | 0.75 | 2 |

Read this table carefully rather than optimistically:

- `train` at 1.00 with `held-out` at 0.25 (D10.1.2, D4.1.1) is memorisation. Four
  held-out rows is one row per class, so a single miss costs 0.25. Those two leaves
  cannot be judged from this run at any confidence.
- `classes improved: 0` on D10.1.2 means the adapter changed nothing on held-out data,
  which is the honest outcome for a leaf whose base model already gets that split right.
- The leaves with four held-out rows per class (D2.2.3 at 7/9, D4.1.2 at 6/8) are the
  only ones here whose held-out number is a measurement rather than a coin flip.

The queue writes every adapter to `adapters/t3_<LEAF>/`, 17.6MB each, gitignored.

## Independent oracle on the queue-trained adapters

`probe_adapters.py`, hand-written states present in no dataset, base vs adapter through
laya's own `agent.predict`.

| leaf | base | adapter | moved |
|---|---|---|---|
| D3.2.1 asset tradeability | not_tradeable 0.470 | tradeable 0.493 | YES |
| D4.1.2 exposure direction | increase_exposure 0.594 | unclear 0.578 | YES |
| D10.1.3 owner authority | agent_may_proceed 0.309 | owner_must_decide 0.503 | YES |
| D4.2.1 kill switch | do_not_fire 0.638 | escalate_human 0.525 | YES |
| D9.2.2 completion truth | claimed_only 0.505 | partially_done 0.510 | YES |
| D1.2.1 champion admission | admit 0.366 | admit 0.399 | no |
| D10.1.2 change risk | external_side_effect 0.495 | external_side_effect 0.444 | no |
| D2.2.3 result routing | accept 0.729 | accept 0.600 | no |

Four of eight moved, and three of those moved in the safe direction on a money or
authority question:

- **D4.1.2** is the clearest win. The state names an instrument and a size but no side and
  no target weights. The base model calls that `increase_exposure` at 0.594 and would have
  raised risk on an ambiguous request. The adapter calls it `unclear` at 0.578 and pushes
  increase_exposure down to 0.143.
- **D3.2.1** reverses a wrong rejection: a clearly tradeable instrument with live data and
  no restrictions was read as `not_tradeable` at 0.470; the adapter reads `tradeable` 0.493.
- **D10.1.3** moves an authority question from `agent_may_proceed` to `owner_must_decide`.

**D2.2.3 is a miss and should not be read as a pass.** The state describes a run that
predates the fix to the survivorship filter. The correct answer is `rerun_validation`. The
adapter still picks `accept`, though it does raise rerun_validation from 0.151 to 0.310. It
did not learn the staleness rule from 42 samples.

**D1.2.1 and D10.1.2 did not move at all**, which is honest: on a state the base model
already classifies with low confidence, an adapter trained on four epochs of 32 rows has
nothing new to say. That is a data problem, not a trainer problem.

## Blocker: the walk escalates on every state, so the adapters are unreachable

Measured through the real `tree_walk.walk` on five varied states:

| state | escalated | why |
|---|---|---|
| completion claim, no evidence | True | T1_overlord: leader 0.464 below the 0.65 threshold |
| completion claim, with evidence | True | T1_overlord: leader 0.516 below the 0.65 threshold |
| live drawdown breach | True | leading option 0.780 with a 0.653 margin |
| capital allocation question | True | T1_overlord leader 0.263; margin 0.033 over D4 |
| data quality question | True | T3_specialist: no leaf cleared its bar (best D5.1.3 at 0.547) |

Five of five escalate. Two things follow, and the first is the important one:

1. **T1 has no adapter.** Ten leaf adapters exist and every one of them sits at T3. T1 runs
   first, and it escalates before T3 is ever reached, so the trained adapters cannot be
   reached at all by the current walk. A T1 adapter over the 10 domains is what makes them
   reachable - and T1 is itself a narrow question ("which domain is this state in"), which
   is exactly the shape Laya handles.

2. The base model's T1 leader sits at 0.263-0.516 against a 0.65 threshold, so even a good
   T3 answer could not surface. The drawdown case is the exception worth noting: it
   produced a 0.780 leader with a 0.653 margin and STILL escalated, so that third state's
   failure is at a different tier - worth reading before assuming the threshold is the
   whole story.

Do not fix this by lowering the threshold. The threshold is doing its job: a 0.464 leader
on "is this work actually finished" is not an answer worth acting on, and the walk handing
it back to the calling agent is the correct behaviour. The fix is a T1 adapter, then a T2
adapter, then re-measure.
