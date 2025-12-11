# fastapi_attendance_app.py

from typing import Optional, List, Any
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
from datetime import datetime, date
import sqlite3
import json
import io
import os
from PIL import Image
import numpy as np

DB_PATH = "attendance_demo.db"
IMAGES_DIR = Path("images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Face Recognition Attendance Backend")

# Dev-time CORS: allow all origins (you can restrict later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

def get_db_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

# ---------------------------
# Embedding computation
# ---------------------------
def _stub_embedding(bytes_data: bytes) -> List[float]:
    """Very small deterministic stub (keeps server running if no ML lib)."""
    try:
        im = Image.open(io.BytesIO(bytes_data)).convert("L").resize((64, 64))
        arr = np.asarray(im, dtype=np.float32) / 255.0
        return arr.flatten().tolist()
    except Exception:
        return []

def compute_embedding(image_bytes: bytes) -> List[float]:
    """
    Try InsightFace (recommended). If not available or fails, fall back to stub.
    Returns a list of floats (embedding) or empty list if no face found.
    """
    # Lazy import & caching of InsightFace model instance
    try:
        # local import so missing package doesn't crash startup
        from insightface.app import FaceAnalysis  # type: ignore
        global _insight_app  # cached across calls
        if "_insight_app" not in globals():
            # model name buffalo_l is a good CPU model; ctx_id=-1 -> CPU
            _insight_app = FaceAnalysis(name="buffalo_l")
            _insight_app.prepare(ctx_id=-1)
        # convert bytes -> RGB numpy array
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.asarray(img)
        faces = _insight_app.get(img_np)
        if not faces:
            return []
        emb = faces[0].embedding
        # ensure Python list of floats
        return emb.tolist() if hasattr(emb, "tolist") else [float(x) for x in emb]
    except Exception as exc:
        # fallback to stub (prints minimal debug to server console)
        print("InsightFace error or not installed — using stub:", str(exc))
        return _stub_embedding(image_bytes)

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Safe cosine similarity:
    - Converts inputs to 1-D numpy arrays
    - Returns 0.0 for empty vectors, mismatched lengths, zero-norm, or on any error
    """
    try:
        a_arr = np.asarray(a, dtype=np.float32).reshape(-1)
        b_arr = np.asarray(b, dtype=np.float32).reshape(-1)
        # if either is empty or lengths differ, return 0.0 (no-match)
        if a_arr.size == 0 or b_arr.size == 0:
            return 0.0
        if a_arr.size != b_arr.size:
            # don't attempt dot on different sizes
            return 0.0
        denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
        if denom == 0.0:
            return 0.0
        return float(np.dot(a_arr, b_arr) / denom)
    except Exception as e:
        # log to server console for debugging, but don't crash
        print("cosine_similarity error:", repr(e))
        return 0.0

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
async def enroll(
    code: str = Form(...),
    name: str = Form(...),
    email: Optional[str] = Form(None),
    file: UploadFile = File(...)
):
    content = await file.read()
    ts = int(datetime.utcnow().timestamp())
    safe_name = os.path.basename(file.filename)
    filename = f"{code}_{ts}_{safe_name}"
    out_path = IMAGES_DIR / filename
    try:
        with open(out_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save image: {e}")

    # compute embedding (InsightFace preferred)
    vec = compute_embedding(content)

    conn = get_db_conn()
    cur = conn.cursor()
    # create or ignore user
    cur.execute("INSERT OR IGNORE INTO users (code, name, email) VALUES (?, ?, ?)", (code, name, email))
    conn.commit()

    cur.execute("SELECT id FROM users WHERE code = ?", (code,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=500, detail="Failed to create or find user")
    user_id = row["id"]

    # insert user_images
    cur.execute("INSERT INTO user_images (user_id, image_url) VALUES (?, ?)", (user_id, str(out_path)))
    conn.commit()
    cur.execute("SELECT id FROM user_images WHERE user_id = ? AND image_url = ?", (user_id, str(out_path)))
    ui_row = cur.fetchone()
    user_image_id = ui_row["id"]

    # save embedding (even if empty - useful to detect later)
    cur.execute(
        "INSERT OR REPLACE INTO embeddings (user_image_id, model, vector_json) VALUES (?, ?, ?)",
        (user_image_id, "insightface_or_stub", json.dumps(vec)),
    )
    conn.commit()
    conn.close()

    return {"status": "ok", "user_id": user_id, "image_path": str(out_path), "embedding_len": len(vec)}

@app.post("/recognize")
async def recognize(file: UploadFile = File(...), threshold: float = 0.6, top_k: int = 1):
    contents = await file.read()
    query_vec = compute_embedding(contents)
    if len(query_vec) == 0:
        raise HTTPException(status_code=400, detail="Could not compute embedding (no face detected?)")

    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT e.id AS emb_id, e.vector_json, ui.user_id, ui.image_url, u.code, u.name
        FROM embeddings e
        JOIN user_images ui ON ui.id = e.user_image_id
        JOIN users u ON u.id = ui.user_id
        """
    )
    rows = cur.fetchall()
    candidates = []
    for r in rows:
        try:
            stored_vec = json.loads(r["vector_json"]) if r["vector_json"] else []
        except Exception:
            stored_vec = []
        try:
            # stored_vec should be a flat list of floats; ensure it's a list
            if not stored_vec or not isinstance(stored_vec, list):
                score = 0.0
            else:
                score = cosine_similarity(query_vec, stored_vec)
        except Exception as e:
            try:
                uid = r["user_id"]
            except Exception:
                uid = None
            print("Error computing score for user", uid, ":", repr(e))
            score = 0.0
        candidates.append(
            {
                "user_id": r["user_id"],
                "code": r["code"],
                "name": r["name"],
                "image_url": r["image_url"],
                "score": score,
            }
        )

    # sort by score desc
    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    best = candidates[0] if candidates else None
    matched = bool(best and best["score"] >= threshold)

    # optionally mark attendance if matched
    if matched:
        try:
            conn2 = get_db_conn()
            cur2 = conn2.cursor()
            now = datetime.utcnow().isoformat()
            today = date.today().isoformat()
            cur2.execute(
                "INSERT INTO attendance (user_id, recognized_at, confidence, session_date) VALUES (?, ?, ?, ?)",
                (best["user_id"], now, float(best["score"]), today),
            )
            conn2.commit()
            conn2.close()
        except Exception as e:
            print("Warning: failed to write attendance:", e)

    conn.close()
    return JSONResponse({"matched": matched, "best_score": best["score"] if best else None, "candidates": candidates[:top_k], "marked_attendance": matched})

@app.get("/attendance")
def view_attendance():
    today = date.today().isoformat()
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT a.id, a.user_id, u.code, u.name, a.recognized_at, a.confidence, a.session_date
        FROM attendance a
        JOIN users u ON u.id = a.user_id
        WHERE a.session_date = ?
        ORDER BY a.id DESC
        """,
        (today,),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return {"date": today, "count": len(rows), "records": rows}

