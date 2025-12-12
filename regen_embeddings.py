#!/usr/bin/env python3
import sqlite3, json, os
import numpy as np
from PIL import Image
import io

from insightface.app import FaceAnalysis

DB = "attendance_demo.db"
MODEL_NAME = "buffalo_s"

print("Loading InsightFace model:", MODEL_NAME)
app = FaceAnalysis(name=MODEL_NAME)
app.prepare(ctx_id=-1)
print("Model loaded successfully.")

def compute_emb(image_path):
    try:
        with open(image_path, "rb") as f:
            data = f.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        arr = np.asarray(img)
        faces = app.get(arr)
        if not faces:
            print("No face found:", image_path)
            return []
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        return face.embedding.tolist()
    except Exception as e:
        print("Failed on:", image_path, "error:", e)
        return []

def main():
    if not os.path.exists(DB):
        print("DB not found:", DB)
        return

    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute("SELECT id, image_url FROM user_images")
    rows = cur.fetchall()
    print("Found", len(rows), "images.")

    for ui_id, image_path in rows:
        if not os.path.exists(image_path):
            print("Missing image:", image_path)
            continue

        vec = compute_emb(image_path)
        print(f"UserImage {ui_id}: embedding length = {len(vec)}")

        cur.execute(
            "INSERT OR REPLACE INTO embeddings (user_image_id, model, vector_json) VALUES (?, ?, ?)",
            (ui_id, MODEL_NAME, json.dumps(vec)),
        )
        conn.commit()

    conn.close()
    print("DONE! Embeddings regenerated successfully.")

if __name__ == "__main__":
    main()

