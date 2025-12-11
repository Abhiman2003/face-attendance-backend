#!/usr/bin/env python3
"""
diagnose_embeddings.py

Usage:
    python diagnose_embeddings.py            # runs summary for all users
    python diagnose_embeddings.py S003       # runs diagnostics for user code S003
"""

import sqlite3
import json
import sys
import math
from collections import defaultdict

DB = "attendance_demo.db"

def cosine(a, b):
    # safe cosine similarity
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x,y in zip(a,b):
        dot += x*y
        na += x*x
        nb += y*y
    if na == 0 or nb == 0:
        return None
    return dot / (math.sqrt(na) * math.sqrt(nb))

def load_embeddings(conn):
    cur = conn.cursor()
    cur.execute("SELECT e.id, e.user_image_id, ui.user_id, ui.image_url, e.vector_json FROM embeddings e JOIN user_images ui ON e.user_image_id = ui.id")
    rows = cur.fetchall()
    emb_by_user = defaultdict(list)
    all_vecs = []
    for id_, ui_id, user_id, img_url, vec_json in rows:
        try:
            vec = json.loads(vec_json)
            if not isinstance(vec, list):
                print(f"embedding {id_} vector_json not a list -> type {type(vec)}")
                continue
            emb_by_user[user_id].append({'emb_id': id_, 'user_image_id': ui_id, 'image_url': img_url, 'vector': vec})
            all_vecs.append(vec)
        except Exception as e:
            print(f"error parsing embedding id {id_}: {e}")
    return emb_by_user, all_vecs

def summary(conn):
    cur = conn.cursor()
    cur.execute("SELECT (SELECT COUNT(*) FROM users) AS users, (SELECT COUNT(*) FROM user_images) AS images, (SELECT COUNT(*) FROM embeddings) AS embeddings, (SELECT COUNT(*) FROM attendance) AS attendance;")
    print("DB counts:", cur.fetchone())

    emb_by_user, all_vecs = load_embeddings(conn)
    lens = [len(v) for vec in all_vecs for v in [vec]]  # flattened
    if lens:
        unique_lens = sorted(set(lens))
        print("Embedding vector lengths present:", unique_lens)
    else:
        print("No embeddings found.")

    # simple per-user summary
    print("\nPer-user embedding counts (user_id -> count):")
    for user_id, items in emb_by_user.items():
        lengths = sorted(set(len(it['vector']) for it in items))
        print(f"  user_id={user_id}: count={len(items)}, vector_lengths={lengths}")

def diagnose_user(conn, user_code):
    # map user_code -> user_id
    cur = conn.cursor()
    cur.execute("SELECT id, code, name FROM users WHERE code = ?", (user_code,))
    r = cur.fetchone()
    if not r:
        print("User code not found:", user_code)
        return
    user_id, code, name = r
    print(f"User: id={user_id}, code={code}, name={name}")

    emb_by_user, all_vecs = load_embeddings(conn)
    if user_id not in emb_by_user:
        print("No embeddings found for this user.")
        return

    user_embs = emb_by_user[user_id]
    print(f" Found {len(user_embs)} embedding(s) for user {code}.")
    for i,e in enumerate(user_embs):
        print(f"  [{i}] emb_id={e['emb_id']} user_image_id={e['user_image_id']} image_url={e['image_url']} length={len(e['vector'])}")

    # compare each user embedding to all others and show top matches
    all_vectors = []
    metadata = []
    for uid, items in emb_by_user.items():
        for it in items:
            all_vectors.append(it['vector'])
            metadata.append((uid, it['emb_id'], it['image_url']))

    # compute similarities: for each user_emb, find top 5 nearest
    for i,ue in enumerate(user_embs):
        sims = []
        for j,vec in enumerate(all_vectors):
            s = cosine(ue['vector'], vec)
            sims.append((s, metadata[j]))
        sims = [x for x in sims if x[0] is not None]
        sims.sort(key=lambda t: t[0], reverse=True)  # descending similarity
        print(f"\nTop matches for user_emb index {i} (emb_id={ue['emb_id']}):")
        for rank, (score, (uid, emb_id, img_url)) in enumerate(sims[:10], start=1):
            tag = "(SELF)" if uid == user_id else ""
            print(f" {rank:02d}. score={score:.4f} user_id={uid} emb_id={emb_id} {tag} img={img_url}")
        # show the best impostor (highest score from other users)
        best_impostor = next((s for s in sims if s[1][0] != user_id), None)
        if best_impostor:
            print(" Best impostor score:", best_impostor[0])
        else:
            print(" No impostor vectors present.")

def main():
    conn = sqlite3.connect(DB)
    if len(sys.argv) == 2:
        diagnose_user(conn, sys.argv[1])
    else:
        summary(conn)
    conn.close()

if __name__ == "__main__":
    main()

