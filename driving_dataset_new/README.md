# ob-road training data

Made-up captions in Florence-2's style, used to train the ob-road model.

## Input and labels

Input text: `"[zone] caption"`, e.g. `[own_lane] A dog sitting on the grass.`

- **zone** is where the photo was dropped: `own_lane` · `oncoming_lane` · `near_sidewalk` · `far_sidewalk`.
  Captions never say where the thing is on the road, so the model has to use the zone.
  Every caption appears once per zone (4 rows per caption).
- **3 labels** to predict (full lists in `labels.json`):
  - `action`: go · stop · slow_down
  - `category`: animal · person · vehicle · traffic_sign · traffic_light · obstacle · harmless · clear_road
  - `speedLimit`: none · 20 · 30 · 40 · 50 · 60 · 80

## What the data covers

- Animals, people, children, vehicles, obstacles, empty roads
- Harmless things the car can drive over: bags, leaves, paper, clothes
- Signs: stop, yield, speed limit, school zone, info, and warning signs (slow down, caution, crossings, road work…)
- Traffic lights: red, amber, green
- **OCR text:** about 30% of captions end like the app's captions do: `The text on it reads "..."`.
  Numbers count as a speed limit only on a traffic sign. On shirts, buses or boxes they are labeled `none`,
  so the model doesn't treat every "30" as a speed sign.

## Commands

```bash
python generate_dataset.py [n]      # n captions (default 40000) → 4n rows, split 80/10/10
python train_model.py               # train, then export to ../public/models/driving-clf/
python train_model.py --export      # only re-export the saved checkpoint
python predict.py --data X.csv --out preds.csv     # run the exported model on a CSV with a 'text' column
python evaluate.py --labels X.csv --preds preds.csv
```

The generator uses a fixed random seed, so it makes the same data every time.

## Files

| File | |
|---|---|
| `generate_dataset.py` | Builds the data from templates |
| `train_model.py` | Fine-tunes all-MiniLM-L6-v2 with 18 outputs (3 + 8 + 7), exports 8-bit ONNX |
| `predict.py`, `evaluate.py` | Score the exported model |
| `caption_real.py` | Captions your own photos for a real test set |
| `action_map.py` | Turns the 3 labels into a game action |
| `labels.json` | Label lists, in output order |

Not in git (rebuild them): `train/val/synth_test .jsonl/.csv`, `stats.json`, `checkpoint/`.

## Game action rules

Applied in order (`action_map.py`, and `src/shared/localDecide.ts` in the app):

1. red or amber traffic_light → `wait_for_green` (a green light: the app says `go`, `action_map.py` says `continue`)
2. traffic_sign + stop → `stop_then_go`
3. traffic_sign + a speed limit → `change_speed`
4. go → `continue`
5. otherwise → `stop` or `slow_down`

## Test on real photos

`synth_test` comes from the same templates as the training data, so ~100% there says little. For a real score:

1. `python caption_real.py --dirs ../public/samples --model microsoft/Florence-2-base-ft`
2. Fill in action / category / speedLimit in `real_test_to_label.csv`, save as `real_test_labeled.csv`
3. `python predict.py --data real_test_labeled.csv --out preds.csv`
4. `python evaluate.py --labels real_test_labeled.csv --preds preds.csv`

Where it's wrong, add templates for those cases to `generate_dataset.py` and retrain.
