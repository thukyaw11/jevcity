"""
Run the exported model (same q8 ONNX file the browser uses) on a CSV with a 'text' column.

    python predict.py --data real_test_labeled.csv --out preds.csv
    python evaluate.py --labels real_test_labeled.csv --preds preds.csv
"""
import argparse, csv, json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

MODEL = Path(__file__).parent.parent / "public" / "models" / "driving-clf"

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="real_test_labeled.csv")
ap.add_argument("--out", default="preds.csv")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(MODEL)
sess = ort.InferenceSession(str(MODEL / "onnx" / "model_quantized.onnx"))
heads = json.load(open(MODEL / "heads.json"))

rows = list(csv.DictReader(open(args.data)))
with open(args.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["id", "action", "category", "speedLimit"])
    w.writeheader()
    for r in rows:
        enc = tok(r["text"], truncation=True, max_length=64, return_tensors="np")
        logits = sess.run(None, {k: enc[k].astype(np.int64) for k in ("input_ids", "attention_mask", "token_type_ids")})[0][0]
        out = {"id": r["id"]}
        for k, (a, b) in heads["slices"].items():
            out[k] = heads["heads"][k][int(logits[a:b].argmax())]
        w.writerow(out)
print(f"wrote {len(rows)} predictions to {args.out}")
