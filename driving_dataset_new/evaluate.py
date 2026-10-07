"""
Score model predictions against the hand-labeled real test set.

    python evaluate.py --labels real_test_labeled.csv --preds preds.csv

preds.csv needs columns: id, action, category, speedLimit  (same ids as the labels file).
Rows with an empty 'action' label are skipped.
"""
import argparse, csv
from collections import Counter, defaultdict
from action_map import to_game_action

ap = argparse.ArgumentParser()
ap.add_argument("--labels", default="real_test_labeled.csv")
ap.add_argument("--preds", default="preds.csv")
args = ap.parse_args()

gold = {r["id"]: r for r in csv.DictReader(open(args.labels)) if r["action"].strip()}
pred = {r["id"]: r for r in csv.DictReader(open(args.preds))}
ids = [i for i in gold if i in pred]
missing = len(gold) - len(ids)
print(f"Scored rows: {len(ids)}" + (f"  ({missing} labeled rows have no prediction)" if missing else ""))
if not ids:
    raise SystemExit("Nothing to score.")

def g(r, k): return (r.get(k) or "none" if k == "speedLimit" else r.get(k, "")).strip()

def report(name, gv, pv):
    acc = sum(a == b for a, b in zip(gv, pv)) / len(gv)
    print(f"\n== {name}: accuracy {acc:.1%}")
    labels = sorted(set(gv) | set(pv))
    conf = defaultdict(Counter)
    for a, b in zip(gv, pv): conf[a][b] += 1
    width = max(len(l) for l in labels) + 2
    print("gold \\ pred".ljust(width) + "".join(l[:12].rjust(14) for l in labels))
    for a in labels:
        print(a.ljust(width) + "".join(str(conf[a][b]).rjust(14) for b in labels))
    for l in labels:
        tp = conf[l][l]; support = sum(conf[l].values())
        if support: print(f"  recall {l}: {tp}/{support} = {tp/support:.1%}")

for head in ["action", "category", "speedLimit"]:
    report(head, [g(gold[i], head) for i in ids], [g(pred[i], head) for i in ids])

ga = [to_game_action(g(gold[i], "action"), g(gold[i], "category"), g(gold[i], "speedLimit")) for i in ids]
pa = [to_game_action(g(pred[i], "action"), g(pred[i], "category"), g(pred[i], "speedLimit")) for i in ids]
report("game_action (after mapping)", ga, pa)

# the dangerous error: should stop, but the model said go
missed = [i for i in ids if g(gold[i], "action") == "stop" and g(pred[i], "action") == "go"]
print(f"\nDANGEROUS (gold stop -> pred go): {len(missed)}")
for i in missed[:20]:
    print(f"  {i}: {gold[i].get('text', '')}")
