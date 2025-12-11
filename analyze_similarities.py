#!/usr/bin/env python3
"""
analyze_similarities.py

Produces similarity distributions for genuine vs impostor pairs and
prints per-user problem cases and a simple threshold suggestion.

Usage:
    python3 analyze_similarities.py
"""
import sqlite3, json, math, itertools, statistics

DB = "attendance_demo.db"

def load_embeddings(conn):
    cur = conn.cursor()
    cur.execute("SELECT e.id, ui.user_id, ui.image_url, e.vector_json FROM embeddings e JOIN user_images ui ON e.user_image_id = ui.id")
    rows = cur.fetchall()
    emb = []
    for id_, user_id, image_url, vec_json in rows:
        try:
            vec = json.loads(vec_json)
            emb.append({"emb_id": id_, "user_id": user_id, "image_url": image_url, "vector": vec})
        except Exception as exc:
            print("Failed to parse embedding id", id_, ":", exc)
    return emb

def cosine(a,b):
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

def main():
    conn = sqlite3.connect(DB)
    emb = load_embeddings(conn)
    conn.close()

    if not emb:
        print("No embeddings found.")
        return

    # build per-user lists
    by_user = {}
    for e in emb:
        by_user.setdefault(e["user_id"], []).append(e)

    genuine_sims = []
    impostor_sims = []
    per_user_stats = {}

    # compute pairwise similarities
    for (i,a), (j,b) in itertools.combinations(enumerate(emb), 2):
        s = cosine(a["vector"], b["vector"])
        if s is None:
            continue
        if a["user_id"] == b["user_id"]:
            genuine_sims.append((s, a, b))
        else:
            impostor_sims.append((s, a, b))

    # global stats
    def stats_from_scores(scores):
        if not scores:
            return {}
        vals = [x for x in scores]
        return {
            "count": len(vals),
            "min": min(vals),
            "max": max(vals),
            "mean": statistics.mean(vals),
            "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "median": statistics.median(vals),
        }

    gen_scores = [s for s,_,_ in genuine_sims]
    imp_scores = [s for s,_,_ in impostor_sims]

    print("Global similarity stats:")
    print(" Genuine pairs:", stats_from_scores(gen_scores))
    print(" Impostor pairs:", stats_from_scores(imp_scores))

    # simple threshold suggestion:
    if imp_scores and gen_scores:
        max_imp = max(imp_scores)
        min_gen = min(gen_scores)
        mid = (max_imp + min_gen) / 2.0
        print("\nMax impostor score: {:.4f}".format(max_imp))
        print("Min genuine score: {:.4f}".format(min_gen))
        print("Simple suggested threshold (midpoint): {:.4f}".format(mid))
        if max_imp < min_gen:
            print(" -> Perfect separability between impostors and genuine (good).")
        else:
            print(" -> Overlap detected (impostor max >= genuine min). Consider:")
            print("    * more enrollment images per user")
            print("    * better alignment/detection")
            print("    * alternative thresholding strategy (per-user thresholds)")
    else:
        print("\nNot enough pairs to compute global threshold suggestion.")

    # per-user worst genuine (min), best impostor (max)
    print("\nPer-user worst genuine vs best impostor:")
    for user_id, items in by_user.items():
        user_gen = []
        user_imp = []
        # intra-user pairs
        for a, b in itertools.combinations(items, 2):
            s = cosine(a["vector"], b["vector"])
            if s is not None:
                user_gen.append(s)
        # impostors vs this user's embeddings
        for a in items:
            for b in emb:
                if b["user_id"] == user_id:
                    continue
                s = cosine(a["vector"], b["vector"])
                if s is not None:
                    user_imp.append(s)
        ug_min = min(user_gen) if user_gen else None
        ui_max = max(user_imp) if user_imp else None
        per_user_stats[user_id] = {"worst_genuine": ug_min, "best_impostor": ui_max, "enroll_count": len(items)}
        print(f" user_id={user_id} enroll_count={len(items)} worst_genuine={ug_min} best_impostor={ui_max}")

    # find the most problematic users (where best_impostor >= worst_genuine or close)
    print("\nPotential problem users (best_imp >= worst_gen OR gap < 0.05):")
    for uid, stats in per_user_stats.items():
        wg = stats["worst_genuine"]
        bi = stats["best_impostor"]
        if wg is None:
            print(f"  user {uid}: only one embedding (no genuine pairs) -> add more enroll images")
            continue
        if bi is None:
            continue
        gap = wg - bi
        if bi >= wg or gap < 0.05:
            print(f"  user {uid}: enrolls={stats['enroll_count']}  worst_genuine={wg:.4f} best_impostor={bi:.4f} gap={gap:.4f}")

    print("\nDone. Use the suggested threshold as a starting point, but validate on live data.")

if __name__ == "__main__":
    main()

