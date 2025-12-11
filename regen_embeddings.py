# regen_embeddings.py
# Regenerate embeddings safely: update if exists, else insert.

import os
import io
import json
import sqlite3
from pathlib import Path
from datetime import datetime

# Try to import compute_embedding from your server file; if not available, use a safe stub
try:
    from fastapi_attendance_app import compute_embedding
    print("Using compute_embedding from fastapi_attendance_app.py")
except Exception as e:
    print("Could not import compute_embedding from fastapi_attendance_app.py:", e)
    print("Using fallback stub embedding (PIL + numpy).")
    from PIL import Image
    import numpy as np
    def compute_embedding(image_bytes: bytes):
        try:
            im = Image.open(io.BytesIO(image_bytes)).convert("L").resize((64, 64))
            arr = np.asarray(im, dtype=np.float32) / 255.0
            return arr.flatten().tolist()
        except Exception:
            return []

# Paths
DB_PATH = "attendance_demo.db"
IMAGES_DIR = Path("images")

if not Path(DB_PATH).exists():
    raise SystemExit(f"ERROR: {DB_PATH} not found. Run script from project folder.")

# Create sqlite backup (proper way: copy DB using sqlite backup)
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
backup_path = f"attendance_demo_backup_{timestamp}.db"
print("Creating DB backup:", backup_path)
src_conn = sqlite3.connect(DB_PATH)
bak_conn = sqlite3.connect(backup_path)
src_conn.backup(bak_conn)
bak_conn.close()
print("Backup complete.")

# Re-open main connection for work
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

# Fetch all user_images
cur.execute("SELECT id, image_url FROM user_images ORDER BY id")
rows = cur.fetchall()
print(f"Found {len(rows)} user_images to process.")

updated = 0
inserted = 0
skipped = 0
errors = 0

for r in rows:
    ui_id = r["id"]
    image_url = r["image_url"]

    # Resolve candidate file paths
    candidates = [
        Path(image_url),                     # maybe stored absolute/relative
        IMAGES_DIR / Path(image_url).name,   # images/filename
        Path(image_url).name and Path(image_url).name or Path(image_url)
    ]
    img_file = None
    for c in candidates:
        if c and c.exists():
            img_file = c
            break

    if img_file is None:
        print(f"[SKIP] image not found for user_image_id={ui_id} (tried: {candidates})")
        skipped += 1
        continue

    try:
        with open(img_file, "rb") as fh:
            bytes_data = fh.read()
    except Exception as e:
        print(f"[SKIP] cannot read {img_file}: {e}")
        skipped += 1
        continue

    try:
        emb = compute_embedding(bytes_data)
    except Exception as e:
        print(f"[ERROR] compute_embedding failed for ui_id={ui_id} file={img_file}: {e}")
        errors += 1
        emb = []

    vec_json = json.dumps(emb)

    # Check if an embedding row already exists for this user_image_id
    cur.execute("SELECT id FROM embeddings WHERE user_image_id = ?", (ui_id,))
    exist = cur.fetchone()
    try:
        if exist:
            # Update existing row
            cur.execute("""
                UPDATE embeddings
                SET model = ?, vector_json = ?, created_at = CURRENT_TIMESTAMP
                WHERE user_image_id = ?
            """, ("insightface_or_stub", vec_json, ui_id))
            updated += 1
        else:
            # Insert new row
            cur.execute("""
                INSERT INTO embeddings (user_image_id, model, vector_json, created_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """, (ui_id, "insightface_or_stub", vec_json))
            inserted += 1

        # commit periodically
        if (updated + inserted) % 20 == 0:
            conn.commit()
            print(f"Committed progress: updated={updated} inserted={inserted}")

    except Exception as e:
        print(f"[ERROR] DB write failed for ui_id={ui_id}: {e}")
        errors += 1

# Final commit & close
conn.commit()
conn.close()

print("=== DONE ===")
print("Inserted:", inserted)
print("Updated:", updated)
print("Skipped (missing files):", skipped)
print("Errors:", errors)
print("Backup at:", backup_path)

