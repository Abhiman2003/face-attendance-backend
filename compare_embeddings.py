from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, List
from pathlib import Path
from datetime import date, datetime
import sqlite3
import json
import io
import numpy as np
from PIL import Image
import os

DB_PATH = "attendance_demo.db"
IMAGES_DIR = Path("images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Attendance Face Recognition Backend")

def get_db_conn():
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def stub_embedding(image_bytes: bytes) -> List[float]:
    try:
        im = Image.open(io.BytesIO(image_bytes)).convert("L").resize((64, 64))
        arr = np.asarray(im, dtype=np.float32) / 255.0
        return arr.flatten().tolist()
    except Exception:
        return []

def compute_embedding_face_recognition(image_bytes: bytes) -> List[float]:
    try:
        import face_recognition
    except Exception:
        raise RuntimeError("face_recognition not installed")
    try:
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.asarray(pil)
    except Exception:
        return []
    encs = face_recognition.face_encodings(arr)
    if not encs:
        return []
    return np.array(encs[0], dtype=float).tolist()

def compute_embedding(image_bytes: bytes) -> List[float]:
    try:
        vec = compute_embedding_face_recognition(image_bytes)
        if vec:
            return vec
    except Exception:
        pass
    return stub_embedding(image_bytes)

def cosine_similarity(a: List[float], b: List[float]) -> float:
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.get("/users")
def list_users():
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, code, name, email, is_active, created_at FROM users ORDER BY id")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return {"count": len(rows), "users": rows}

@app.post("/enroll")
async def enroll_user(
    code: str = Form(...),
    name: str = Form(...),
    email: Optional[str] = Form(None),
    file: UploadFile = File(...)
):
    ts = int(datetime.utcnow().timestamp())
    safe_name = os.path.basename(file.filename)
    filename = f"{code}_{ts}_{safe_name}"
    out_path = IMAGES_DIR / filename
    contents = await file.read()
    with open(out_path, "wb") as f:
        f.write(contents)

    vec = compute_embedding(contents)

    conn = get_db_conn()
    cur = conn.cursor()

    cur.execute("INSERT OR IGNORE INTO users (code, name, email) VALUES (?, ?, ?)", (code, name, email))
    conn.commit()

    cur.execute("SELECT id FROM users WHERE code = ?", (code,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=500, detail="Failed to create or find user")

    user_id = row["id"]

    cur.execute("INSERT OR IGNORE INTO user_images (user_id, image_url) VALUES (?, ?)", (user_id, str(out_path)))
    conn.commit()

    cur.execute("SELECT id FROM user_images WHERE user_id = ? AND image_url = ?", (user_id, str(out_path)))
    ui_row = cur.fetchone()
    user_image_id = ui_row["id"]

    cur.execute("INSERT INTO embeddings (user_image_id, model, vector_json) VALUES (?, ?, ?)",
                (user_image_id, "face_recognition_or_stub", json.dumps(vec)))
    conn.commit()
    conn.close()

    return {"status": "ok", "user_id": user_id, "image_path": str(out_path), "embedding_len": len(vec)}

@app.post("/recognize")
async def recognize(file: UploadFile = File(...), threshold: float = 0.6, top_k: int = 1):
    contents = await file.read()
    query_vec = compute_embedding(contents)
    if len(query_vec) == 0:
        raise HTTPException(status_code=400, detail="Could not compute embedding (no face?)")

    conn = get_db_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT e.id AS emb_id, e.vector_json, ui.user_id, ui.image_url, u.code, u.name
        FROM embeddings e
        JOIN user_images ui ON ui.id = e.user_image_id
        JOIN users u ON u.id = ui.user_id
    """)

    candidates = []
    for r in cur.fetchall():
        vec = json.loads(r["vector_json"]) if r["vector_json"] else []
        score = cosine_similarity(query_vec, vec)
        candidates.append({
            "user_id": r["user_id"],
            "code": r["code"],
            "name": r["name"],
            "image_url": r["image_url"],
            "score": score
        })

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    best = candidates[0] if candidates else None
    matched = best and best["score"] >= threshold

    conn.close()

    return {
        "matched": bool(matched),
        "best_score": best["score"] if best else None,
        "candidates": candidates[:top_k]
    }

@app.get("/attendance")
def view_attendance():
    today = date.today().isoformat()
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT a.id, a.user_id, u.code, u.name, a.recognized_at
        FROM attendance a
        JOIN users u ON u.id = a.user_id
        WHERE a.session_date = ?
    """, (today,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return {"date": today, "count": len(rows), "records": rows}
