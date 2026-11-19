import json, sys
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm

MODEL = sys.argv[1] if len(sys.argv) > 1 else "vinid/plip"
name = MODEL.split("/")[-1]
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("model:", MODEL, "| device:", device)

model = CLIPModel.from_pretrained(MODEL).to(device).eval()
proc = CLIPProcessor.from_pretrained(MODEL)

recs = [json.loads(l) for l in open("data/records.jsonl")]
ids = [r["id"] for r in recs]

vecs = []
B = 32
with torch.no_grad():
    for i in tqdm(range(0, len(recs), B)):
        imgs = [Image.open(r["path"]).convert("RGB") for r in recs[i:i + B]]
        inp = proc(images=imgs, return_tensors="pt").to(device)
        v = model.get_image_features(**inp)
        if not torch.is_tensor(v):
            v = v.image_embeds if hasattr(v, "image_embeds") else v.pooler_output
        v = v / v.norm(dim=-1, keepdim=True)
        vecs.append(v.cpu().numpy().astype("float32"))

E = np.concatenate(vecs)
np.save(f"data/emb_{name}.npy", E)
json.dump(ids, open("data/emb_ids.json", "w"))
print("saved", E.shape, "->", f"data/emb_{name}.npy")

train_idx = [i for i, r in enumerate(recs) if r["split"] == "train"]
test_idx = [i for i, r in enumerate(recs) if r["split"] == "test" and r["near_dup_of"] is None][:3]
for t in test_idx:
    sims = E[train_idx] @ E[t]
    best = train_idx[int(sims.argmax())]
    print("\nTEST :", recs[t]["qa"][0])
    print("MATCH:", recs[best]["qa"][0], f"(cosine {sims.max():.3f})")