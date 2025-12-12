# fastapi_attendance_app.py
from typing import Optional, List, Tuple
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pathlib import Path
from datetime import datetime, date
import sqlite3
import json
import io
import os
from PIL import Image
import numpy as np
import logging

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
DB_PATH = "attendance_demo.db"
IMAGES_DIR = Path("images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# InsightFace model name (not used when REQUIRE_INSIGHT=False)
INSIGHT_MODEL = "buffalo_s"

# ❗ Disable InsightFace for Render FREE tier (prevents memory crash)
REQUIRE_INSIGHT = False

DEFAULT_THRESHOLD = 0.70

# ------------------------------------------------------------
# App + logging
# ------------------------------------------------------------
app = FastAPI(title="Face Recognition Attendance Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

logger = logging.getLogger("uvicorn.error")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------------
# DB helper
# ------------------------------------------------------------
def get_db_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

# ------------------------------------------------------------
# Stub embedding (fallback)
# ------------------------------------------------------------
def _stub_embedding(bytes_data: bytes) -> List[float]:
    try:
        im = Image.open(io.BytesIO(bytes_data)).convert("L").resize((64, 64))
        arr = np.asarray(im, dtype=np.float32) / 255.0
        return arr.flatten().tolist()
    except Exception:
        return []

# ------------------------------------------------------------
# InsightFace initialization (disabled due to free tier limits)
# ------------------------------------------------------------
_insight_app = None
_insight_available = False
_insight_error = "InsightFace disabled (REQUIRE_INSIGHT=False)"

@app.get("/debug_model")
def debug_model():
    return {
        "insight_available": False,
        "insight_error": "InsightFace disabled (Render free tier memory limit)"
    }

# ------------------------------------------------------------
# Embedding computation
# ------------------------------------------------------------
def compute_embedding(image_bytes: bytes, require_insight: Optional[bool] = None) -> List[float]:
    if require_insight is None:
        require_insight = REQUIRE_INSIGHT

    if require_insight:
        raise HTTPException(status_code=500, detail="InsightFace disabled on Render free tier")

    # Always use stub embedding
    return _stub_embedding(image_bytes)

# ------------------------------------------------------------
# Similarity utils
# ------------------------------------------------------------
def cosine_similarity(a: List[float], b: List[float]) -> float:
    try:
        a_arr = np.asarray(a, dtype=np.float32).reshape(-1)
        b_arr = np.asarray(b, dtype=np.float32).reshape(-1)
        if a_arr.size == 0 or b_arr.size == 0:
            return 0.0
        if a_arr.size != b_arr.size:
            return 0.0
        denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
        if denom == 0.0:
            return 0.0
        return float(np.dot(a_arr, b_arr) / denom)
    except Exception:
        return 0.0

# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------
@app.get("/")
def root_redirect():
    return RedirectResponse(url="/docs")

@app.get("/users")
def list_users():
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, code, name, email, is_active, created_at FROM users ORDER BY id")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return {"count": len(rows), "users": rows}

@app.post("/enroll")
async def enroll(code: str = Form(...), name: str = Form(...), email: Optional[str] = Form(None), file: UploadFile = File(...)):
    content = await file.read()
    ts = int(datetime.utcnow().timestamp())
    filename = f"{code}_{ts}_{os.path.basename(file.filename)}"
    out_path = IMAGES_DIR / filename

    with open(out_path, "wb") as f:
        f.write(content)

    vec = compute_embedding(content)

    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO users (code, name, email) VALUES (?, ?, ?)", (code, name, email))
    conn.commit()

    cur.execute("SELECT id FROM users WHERE code = ?", (code,))
    row = cur.fetchone()
    user_id = row["id"]

    cur.execute("INSERT INTO user_images (user_id, image_url) VALUES (?, ?)", (user_id, str(out_path)))
    conn.commit()

    cur.execute("SELECT id FROM user_images WHERE image_url = ?", (str(out_path),))
    ui_row = cur.fetchone()
    user_image_id = ui_row["id"]

    cur.execute(
        "INSERT INTO embeddings (user_image_id, model, vector_json) VALUES (?, ?, ?)",
        (user_image_id, "stub", json.dumps(vec)),
    )
    conn.commit()
    conn.close()

    return {"status": "ok", "user_id": user_id, "embedding_len": len(vec)}

@app.post("/recognize")
async def recognize(file: UploadFile = File(...), threshold: float = DEFAULT_THRESHOLD, top_k: int = 1):
    contents = await file.read()
    query_vec = compute_embedding(contents)

    if len(query_vec) == 0:
        raise HTTPException(status_code=400, detail="Could not compute embedding")

    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT e.vector_json, ui.user_id, ui.image_url, u.code, u.name
        FROM embeddings e
        JOIN user_images ui ON ui.id = e.user_image_id
        JOIN users u ON ui.user_id = u.id
    """)

    rows = cur.fetchall()
    candidates = []

    for r in rows:
        stored_vec = json.loads(r["vector_json"])
        score = cosine_similarity(query_vec, stored_vec)
        candidates.append({
            "user_id": r["user_id"],
            "code": r["code"],
            "name": r["name"],
            "image_url": r["image_url"],
            "score": score,
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    best = candidates[0] if candidates else None
    matched = best["score"] >= threshold if best else False

    conn.close()
    return {"matched": matched, "best": best, "top_k": candidates[:top_k]}

@app.get("/attendance")
def view_attendance():
    today = date.today().isoformat()
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT a.id, u.code, u.name, a.recognized_at, a.confidence
        FROM attendance a
        JOIN users u ON a.user_id = u.id
        WHERE a.session_date = ?
        ORDER BY a.id DESC
    """, (today,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return {"date": today, "records": rows}

@app.get("/ping")
def ping():
    return {"status": "ok", "insight_available": False}
