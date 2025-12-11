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

# InsightFace model name to use (smaller models are kinder to limited RAM)
INSIGHT_MODEL = "buffalo_s"  # try "buffalo_s", "antelope"; "buffalo_l" is larger

# If True: require InsightFace to be available (raise 500 if not).
# If False: fall back to the deterministic stub embedding (recognition will be poor).
REQUIRE_INSIGHT = True

# Default matching threshold (you should compute a data-driven value using analyzer scripts)
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

# Use uvicorn logger for stdout capture on Render
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
    """Deterministic, low-quality embedding used as fallback so server doesn't crash."""
    try:
        im = Image.open(io.BytesIO(bytes_data)).convert("L").resize((64, 64))
        arr = np.asarray(im, dtype=np.float32) / 255.0
        return arr.flatten().tolist()
    except Exception:
        return []

# ------------------------------------------------------------
# InsightFace initialization (lazy + logged)
# ------------------------------------------------------------
_insight_app = None
_insight_available = False
_insight_error = None

def init_insightface(model_name: str = INSIGHT_MODEL):
    """
    Attempt to initialize InsightFace FaceAnalysis and log results.
    Call at import/startup and before compute_embedding.
    """
    global _insight_app, _insight_available, _insight_error
    if _insight_app is not None or _insight_available:
        return

    try:
        # lazy import
        from insightface.app import FaceAnalysis  # type: ignore
        logger.info("Attempting to initialize InsightFace model: %s", model_name)
        _insight_app = FaceAnalysis(name=model_name)
        # use CPU on Render / general hosts
        _insight_app.prepare(ctx_id=-1)
        _insight_available = True
        _insight_error = None
        logger.info("InsightFace initialized successfully: %s", model_name)
    except Exception as e:
        _insight_app = None
        _insight_available = False
        _insight_error = str(e)
        logger.error("InsightFace initialization failed: %s", _insight_error)

# init at import so render logs show startup behavior
init_insightface()

@app.get("/debug_model")
def debug_model():
    """Return model load status and error message (if any)."""
    return {
        "insight_available": bool(_insight_available),
        "insight_error": _insight_error,
        "model_obj": type(_insight_app).__name__ if _insight_app else None,
        "insight_model_name": INSIGHT_MODEL
    }

# ------------------------------------------------------------
# Embedding computation
# ------------------------------------------------------------
def compute_embedding(image_bytes: bytes, require_insight: Optional[bool] = None) -> List[float]:
    """
    Compute embedding for given image bytes.
    - If InsightFace is available, uses it.
    - If not available:
        - If require_insight True -> raises HTTPException(500)
        - If require_insight False -> returns stub embedding
    """
    if require_insight is None:
        require_insight = REQUIRE_INSIGHT

    # Ensure init attempted
    init_insightface()

    if _insight_available and _insight_app is not None:
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_np = np.asarray(img)
            faces = _insight_app.get(img_np)
            if not faces:
                logger.info("No faces detected by InsightFace.")
                return []
            # pick largest face for robustness
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            emb = face.embedding
            vec = emb.tolist() if hasattr(emb, "tolist") else [float(x) for x in emb]
            logger.info("Computed embedding (len=%d) with InsightFace.", len(vec))
            return vec
        except Exception as e:
            logger.error("InsightFace compute failed: %s", repr(e))
            if require_insight:
                raise HTTPException(status_code=500, detail="InsightFace embedding generation failed: " + str(e))
            else:
                logger.warning("Falling back to stub embedding due to compute error.")
                return _stub_embedding(image_bytes)
    else:
        msg = f"InsightFace not available: {_insight_error}"
        logger.warning(msg)
        if require_insight:
            raise HTTPException(status_code=500, detail=msg)
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
    except Exception as e:
        logger.error("cosine_similarity error: %s", repr(e))
        return 0.0

# ------------------------------------------------------------
# Endpoints: users / enroll / recognize / attendance
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
    vec = compute_embedding(content, require_insight=REQUIRE_INSIGHT)

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

    cur.execute("INSERT INTO user_images (user_id, image_url) VALUES (?, ?)", (user_id, str(out_path)))
    conn.commit()
    cur.execute("SELECT id FROM user_images WHERE user_id = ? AND image_url = ?", (user_id, str(out_path)))
    ui_row = cur.fetchone()
    user_image_id = ui_row["id"]

    cur.execute(
        "INSERT OR REPLACE INTO embeddings (user_image_id, model, vector_json) VALUES (?, ?, ?)",
        (user_image_id, INSIGHT_MODEL if _insight_available else "stub", json.dumps(vec)),
    )
    conn.commit()
    conn.close()

    return {"status": "ok", "user_id": user_id, "image_path": str(out_path), "embedding_len": len(vec)}

@app.post("/recognize")
async def recognize(file: UploadFile = File(...), threshold: float = DEFAULT_THRESHOLD, top_k: int = 1):
    contents = await file.read()
    # compute probe embedding (raises 500 if model required but not available)
    query_vec = compute_embedding(contents, require_insight=REQUIRE_INSIGHT)
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
        if not stored_vec or not isinstance(stored_vec, list):
            score = 0.0
        else:
            score = cosine_similarity(query_vec, stored_vec)
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

    # mark attendance if matched
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
            logger.warning("Warning: failed to write attendance: %s", e)

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

# ------------------------------------------------------------
# Simple health/ping endpoint
# ------------------------------------------------------------
@app.get("/ping")
def ping():
    return {"status": "ok", "insight_available": bool(_insight_available)}
