import sqlite3
import json
from pathlib import Path
import numpy as np
import io
from PIL import Image

DB_PATH = "attendance_demo.db"
IMAGES_DIR = Path("images")

# --------------------------
# InsightFace embedding
# --------------------------
def compute_embedding_insightface(image_bytes: bytes):
    try:
        from insightface.app import FaceAnalysis
        import numpy as np

        global insight_app
        if "insight_app" not in globals():
            insight_app = FaceAnalysis(name="buffalo_l")
            insight_app.prepare(ctx_id=-1)

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.asarray(img)

        faces = insight_app.get(img_np)
        if not faces:
            print("No face found in:", img)
            return []

        return faces[0].embedding.tolist()
    except Exception as e:
        print("InsightFace error:", e)
        return []

# --------------------------
# Main batch process
# --------------------------
def regenerate_embeddings():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        SELECT ui.id AS uiid, ui.image_url
        FROM user_images ui
    """)

    rows = cur.fetchall()
    print("Found image rows:", len(rows))

    for r in rows:
        uiid = r["uiid"]
        path = Path(r["image_url"])
        if not path.exists():
            print("Missing file:", path)
            continue

        img_bytes = path.read_bytes()
        vec = compute_embedding_insightface(img_bytes)

        cur.execute("""
            INSERT OR REPLACE INTO embeddings (user_image_id, model, vector_json)
            VALUES (?, ?, ?)
        """, (uiid, "insightface", json.dumps(vec)))
        conn.commit()

        print("Updated embedding for:", path, "| length:", len(vec))

    conn.close()
    print("Batch embedding generation complete.")

if __name__ == "__main__":
    regenerate_embeddings()

