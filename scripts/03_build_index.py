import json, sys
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (VectorParams, Distance, PointStruct,
                                  Filter, FieldCondition, MatchValue)

name = sys.argv[1] if len(sys.argv) > 1 else "plip"
collection = f"pathvqa_{name.split('-')[0]}"

E = np.load(f"data/emb_{name}.npy")
recs = [json.loads(l) for l in open("data/records.jsonl")]
assert len(E) == len(recs)

client = QdrantClient(path="data/qdrant")
if client.collection_exists(collection):
    client.delete_collection(collection)
client.create_collection(collection,
    vectors_config=VectorParams(size=E.shape[1], distance=Distance.COSINE))

points = [
    PointStruct(
        id=i,
        vector=E[i].tolist(),
        payload={
            "image_id": r["id"],
            "split": r["split"],
            "path": r["path"],
            "doc": r["doc"],
            "n_qa": len(r["qa"]),
            "near_dup_of": r["near_dup_of"],
            "multi_split": len(r["splits_seen"]) > 1,
        },
    )
    for i, r in enumerate(recs)
]
for i in range(0, len(points), 256):
    client.upsert(collection, points=points[i:i + 256])

print(collection, "->", client.count(collection).count, "points stored")

train_only = Filter(must=[FieldCondition(key="split", match=MatchValue(value="train"))])
t = next(i for i, r in enumerate(recs) if r["split"] == "test" and r["near_dup_of"] is None)
hits = client.query_points(collection, query=E[t].tolist(),
                           query_filter=train_only, limit=5).points

print("\nquery:", recs[t]["id"], "|", recs[t]["qa"][0]["q"])
for h in hits:
    print(f"  {h.score:.3f}  {h.payload['split']}  {h.payload['doc'].splitlines()[0][:90]}")