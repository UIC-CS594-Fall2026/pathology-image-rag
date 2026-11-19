import hashlib, io, json, os, glob
import numpy as np
import pandas as pd
import imagehash
from PIL import Image
from huggingface_hub import snapshot_download
from tqdm import tqdm

OUT = "data"
os.makedirs(f"{OUT}/images", exist_ok=True)

root = snapshot_download("flaviagiammarino/path-vqa", repo_type="dataset")

records = {}
for split in ["train", "validation", "test"]:
    for fp in sorted(glob.glob(f"{root}/data/{split}-*.parquet")):
        df = pd.read_parquet(fp)
        for _, row in tqdm(df.iterrows(), total=len(df), desc=os.path.basename(fp)[:20]):
            b = row["image"]["bytes"]
            key = hashlib.md5(b).hexdigest()[:16]
            if key not in records:
                img = Image.open(io.BytesIO(b))
                ext = (img.format or "jpeg").lower().replace("jpeg", "jpg")
                path = f"{OUT}/images/{key}.{ext}"
                with open(path, "wb") as f:
                    f.write(b)
                records[key] = {"id": key, "split": split, "splits_seen": [],
                                "path": path, "size": img.size,
                                "phash": str(imagehash.phash(img.convert("RGB"))), "qa": []}
            r = records[key]
            if split not in r["splits_seen"]:
                r["splits_seen"].append(split)
            r["qa"].append({"q": row["question"], "a": row["answer"]})
        del df

for r in records.values():
    r["doc"] = "\n".join(f"Q: {x['q']} A: {x['a']}" for x in r["qa"])
    r["n_yesno"] = sum(str(x["a"]).lower() in ("yes", "no") for x in r["qa"])

def bits(h):
    return imagehash.hex_to_hash(h).hash.flatten()

recs = list(records.values())
train = [r for r in recs if r["split"] == "train"]
T = np.stack([bits(r["phash"]) for r in train])
for r in recs:
    r["near_dup_of"] = None
    if r["split"] != "train":
        d = (T != bits(r["phash"])).sum(1)
        if d.min() <= 6:
            r["near_dup_of"] = train[int(d.argmin())]["id"]

with open(f"{OUT}/records.jsonl", "w") as f:
    for r in recs:
        f.write(json.dumps(r) + "\n")

for s in ["train", "validation", "test"]:
    group = [r for r in recs if r["split"] == s]
    dups = sum(r["near_dup_of"] is not None for r in group)
    qa = sum(len(r["qa"]) for r in group)
    yn = sum(r["n_yesno"] for r in group)
    print(f"{s}: {len(group)} images, {qa} Q&As, {yn/qa:.0%} yes/no, {dups} near-dups of train")
print("images in more than one split:", sum(len(r["splits_seen"]) > 1 for r in recs))
