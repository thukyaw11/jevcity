"""
Fine-tune MiniLM on the caption dataset and export it for the browser.

    python train_model.py            # train, test on synth_test, export to public/models/driving-clf
    python train_model.py --export   # only re-export the saved checkpoint

One classifier with 17 outputs = action (3) + category (7) + speedLimit (7), each slice trained
with its own loss. Keeps it a plain sequence classifier, so transformers.js can load it as is.
"""
import json, random, shutil, sys
from pathlib import Path

import torch
from torch.nn.functional import cross_entropy
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE = "sentence-transformers/all-MiniLM-L6-v2"
HERE = Path(__file__).parent
OUT = HERE.parent / "public" / "models" / "driving-clf"
CKPT = HERE / "checkpoint"
EPOCHS, LR, BATCH, MAX_LEN = 3, 5e-5, 64, 64

labels = json.load(open(HERE / "labels.json"))
HEADS = {"action": labels["action"], "category": labels["category"], "speedLimit": labels["speedLimit"]}
SLICES, start = {}, 0
for k, v in HEADS.items():
    SLICES[k] = (start, start + len(v))
    start += len(v)
NUM = start

def load(name):
    rows = [json.loads(l) for l in open(HERE / f"{name}.jsonl")]
    return [(r["text"], [HEADS[k].index(r[k]) for k in HEADS]) for r in rows]

EXPORT_ONLY = "--export" in sys.argv
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); random.seed(0)
tok = AutoTokenizer.from_pretrained(CKPT if EXPORT_ONLY else BASE)
model = AutoModelForSequenceClassification.from_pretrained(CKPT if EXPORT_ONLY else BASE, num_labels=NUM).to(device)

def batches(data, shuffle):
    idx = list(range(len(data)))
    if shuffle: random.shuffle(idx)
    for i in range(0, len(idx), BATCH):
        chunk = [data[j] for j in idx[i:i + BATCH]]
        enc = tok([t for t, _ in chunk], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        y = torch.tensor([ys for _, ys in chunk])
        yield {k: v.to(device) for k, v in enc.items()}, y.to(device)

def loss_fn(logits, y):
    return sum(cross_entropy(logits[:, a:b], y[:, i]) for i, (a, b) in enumerate(SLICES.values()))

@torch.no_grad()
def evaluate(data):
    model.eval()
    hits = {k: 0 for k in HEADS}
    for enc, y in batches(data, False):
        logits = model(**enc).logits
        for i, (k, (a, b)) in enumerate(SLICES.items()):
            hits[k] += (logits[:, a:b].argmax(-1) == y[:, i]).sum().item()
    model.train()
    return {k: round(v / len(data), 4) for k, v in hits.items()}

if not EXPORT_ONLY:
    train, val, test = load("train"), load("val"), load("synth_test")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * ((len(train) + BATCH - 1) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1, anneal_strategy="linear")

    model.train()
    for epoch in range(EPOCHS):
        for step, (enc, y) in enumerate(batches(train, True)):
            loss = loss_fn(model(**enc).logits, y)
            loss.backward(); opt.step(); sched.step(); opt.zero_grad()
            if step % 200 == 0: print(f"epoch {epoch} step {step} loss {loss.item():.4f}", flush=True)
        print(f"epoch {epoch} val", evaluate(val), flush=True)
    print("synth_test", evaluate(test))
    model.config.id2label = {i: f"{k}:{v}" for k, vals in HEADS.items() for i, v in enumerate(vals, SLICES[k][0])}
    model.config.label2id = {v: k for k, v in model.config.id2label.items()}
    model.save_pretrained(CKPT); tok.save_pretrained(CKPT)  # keep the weights even if export fails

# export: same file layout transformers.js expects (config + tokenizer + onnx/model*.onnx)
if OUT.exists(): shutil.rmtree(OUT)
(OUT / "onnx").mkdir(parents=True)
model.cpu().eval()
shutil.copy(CKPT / "config.json", OUT / "config.json")
tok.save_pretrained(OUT)
json.dump({"heads": HEADS, "slices": SLICES}, open(OUT / "heads.json", "w"), indent=2)

enc = tok(["[own_lane] A dog."], return_tensors="pt")
names = ["input_ids", "attention_mask", "token_type_ids"]
dyn = {n: {0: "batch", 1: "seq"} for n in names} | {"logits": {0: "batch"}}
torch.onnx.export(model, tuple(enc[n] for n in names), OUT / "onnx" / "model.onnx", input_names=names,
                  output_names=["logits"], dynamic_axes=dyn, opset_version=17, dynamo=False)

from onnxruntime.quantization import QuantType, quantize_dynamic
quantize_dynamic(OUT / "onnx" / "model.onnx", OUT / "onnx" / "model_quantized.onnx", weight_type=QuantType.QInt8)
(OUT / "onnx" / "model.onnx").unlink()  # browser only needs the q8 one
print("exported to", OUT, file=sys.stderr)
