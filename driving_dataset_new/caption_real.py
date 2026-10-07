"""
Build the REAL test set: caption your photos with Florence-2, one row per (photo, zone),
with empty label columns for you to fill in by hand.

    pip install transformers torch pillow einops timm
    python caption_real.py --dirs data-set samples

Then open real_test_to_label.csv, fill action / category / speedLimit for each row
(~100 photos is enough), and save it as real_test_labeled.csv.

IMPORTANT: use the SAME Florence model and task prompt your app uses in the browser
(e.g. base vs large, <CAPTION> vs <DETAILED_CAPTION>), or the test captions won't match
what the classifier sees in the game. If your app already logs its captions, you can
skip this script and label those instead.
"""
import argparse, csv
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

ZONES = ["own_lane", "oncoming_lane", "near_sidewalk", "far_sidewalk"]  # match the app
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

ap = argparse.ArgumentParser()
ap.add_argument("--dirs", nargs="+", default=["data-set", "samples"])
ap.add_argument("--model", default="microsoft/Florence-2-base")
ap.add_argument("--task", default="<CAPTION>")
ap.add_argument("--out", default="real_test_to_label.csv")
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, torch_dtype=dtype).to(device)
processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

images = sorted(p for d in args.dirs for p in Path(d).rglob("*") if p.suffix.lower() in EXTS)
print(f"Found {len(images)} images")

with open(args.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["id", "image", "zone", "caption", "text",
                                      "action", "category", "speedLimit", "notes"])
    w.writeheader()
    for i, path in enumerate(images):
        image = Image.open(path).convert("RGB")
        inputs = processor(text=args.task, images=image, return_tensors="pt").to(device, dtype)
        ids = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                             max_new_tokens=256, num_beams=3, do_sample=False)
        raw = processor.batch_decode(ids, skip_special_tokens=False)[0]
        caption = processor.post_process_generation(raw, task=args.task,
                                                    image_size=(image.width, image.height))[args.task].strip()
        for z in ZONES:
            w.writerow({"id": f"real-{i:04d}-{z}", "image": str(path), "zone": z,
                        "caption": caption, "text": f"[{z}] {caption}",
                        "action": "", "category": "", "speedLimit": "", "notes": ""})
        print(f"[{i+1}/{len(images)}] {path.name}: {caption}")
print(f"Wrote {args.out} -- now fill in the label columns by hand.")
