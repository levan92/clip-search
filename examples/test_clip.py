import time
import argparse
import operator
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import clip

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("text", help="path to text file containing query strings")
ap.add_argument("--model", choices=clip.available_models(), default="RN50")
ap.add_argument("--reps", help="num of reps for time benchmark", default=1, type=int)
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

img_path = Path(args.image)
pil_img = Image.open(args.image)
with open(args.text, "r") as f:
    queries = [l.strip() for l in f.readlines()]
print(queries)

model, preprocess = clip.load(args.model, device=device)
image = preprocess(pil_img).unsqueeze(0).to(device)

text = clip.tokenize(queries).to(device)

durations = [0 for _ in range(4)]
tics = [None for _ in range(5)]

with torch.no_grad():
    for _ in range(args.reps):
        tics[0] = time.perf_counter()
        image_features = model.encode_image(image)
        tics[1] = time.perf_counter()
        text_features = model.encode_text(text)
        tics[2] = time.perf_counter()
        logits_per_image, logits_per_text = model(image, text)
        tics[3] = time.perf_counter()
        probs = logits_per_image.softmax(dim=-1).cpu().numpy().flatten()
        tics[4] = time.perf_counter()

        this_durs = [next - curr for curr, next in zip(tics, tics[1:])]
        durations = list(map(operator.add, durations, this_durs))

dur_names = ["encode image", "encode text", "all", "softmax"]
avg_durs = np.array(durations) / args.reps
for name, dur in zip(dur_names, avg_durs):
    print(f"{name}:{dur:0.4f}s")

for label, prob in sorted(zip(queries, probs), key=lambda x: x[1], reverse=True):
    print(f"{prob*100:.0f}%: {label}")
