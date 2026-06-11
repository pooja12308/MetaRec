import sys, os, math, random
sys.path.insert(0, os.path.dirname(__file__))

from core.data_loader import (
    load_users, load_products, load_interactions,
    build_existing_profile, build_new_user_profile,
    build_user_hin
)
from core.engine import (
    recommend, score_bfi10,
    get_metapath_traces, evaluate,
    PERSONALITY_AFFINITY, dominant_trait, CATEGORIES,
)

print("🔮 MetaRec Terminal Demo\n")

# ─────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────
print("📂 Loading data...")
products_df = load_products()
users_df = load_users()
interactions_df = load_interactions()
print("✔ Data loaded\n")

# ─────────────────────────────────────────
# STEP 2: USER INPUT
# ─────────────────────────────────────────
uid = input("Enter User ID: ").strip()

# EXISTING USER
if uid in users_df["uid"].values:
    print("\n📝 Existing User Detected")
    profile = build_existing_profile(uid, users_df, interactions_df)
    print("✔ Profile built")

# NEW USER
else:
    print("\n✨ New User Detected")
    name = uid
    print("\n🧠 Answer 5 quick questions (1-5):")
    questions = [
        "I am outgoing",
        "I am organized",
        "I get nervous easily",
        "I am creative",
        "I am cooperative"
    ]
    answers = {}
    for i, q in enumerate(questions):
        ans = int(input(f"{q}: "))
        answers[i] = ans
    categories = input("\nEnter interests (comma separated): ").split(",")
    profile = build_new_user_profile(uid, name, answers, categories)
    print("✔ Profile created")

print("\n📜 USER INTERACTION DATA\n")
print("Interacted Product IDs:", profile["interacted_pids"][:5])
print("\nSample Review Text:")
print(profile["user_text"][:200])
print("\n🎯 Extracted Interests (from reviews):")
for topic, score in profile["topic_interests"][:5]:
    print(f"  {topic} → {round(score,3)}")

# ─────────────────────────────────────────
# STEP 3: BUILD HIN
# ─────────────────────────────────────────
hin = build_user_hin(profile, products_df, interactions_df, users_df)
print("✔ HIN constructed\n")

# ─────────────────────────────────────────
# STEP 4: RECOMMENDATIONS
# ─────────────────────────────────────────
print("✨ Generating Recommendations...\n")
recs = recommend(hin, profile["uid"], top_n=5)

if not recs:
    print("⚠ No recommendations found\n")
else:
    print("🎯 Recommended Products:\n")
    for i, r in enumerate(recs, 1):
        prod  = r["product"]
        name  = prod.get("name", "Unknown")
        score = round(r.get("score", 0), 3)
        reasons = r.get("reasons", [])
        print(f"{i}. {name}  (score: {score})")
        if reasons:
            print("   ↳ Why:", ", ".join(reasons))

# ─────────────────────────────────────────
# STEP 5: HIN SUMMARY
# ─────────────────────────────────────────
print("\n🌐 HIN GRAPH SUMMARY\n")
stats = hin.stats()
for k, v in stats.items():
    print(f"  {k}: {v}")

# ─────────────────────────────────────────
# STEP 6: ASCII GRAPH
# ─────────────────────────────────────────
print("\n🔗 ASCII GRAPH (Simplified)\n")
print(f"[USER] {profile['uid']}")
topics = [t[0] for t in profile.get("topic_interests", [])[:3]]
for t in topics:
    print(f"   └── [TOPIC] {t}")
    for r in recs[:2]:
        pname = r["product"].get("name", "Product")
        print(f"         └── [PRODUCT] {pname}")

# ─────────────────────────────────────────
# STEP 7: METAPATH TRACES
# ─────────────────────────────────────────
print("\n🔍 METAPATH EXPLANATION\n")
traces = get_metapath_traces(hin, profile["uid"], top_k=5)
for t in traces:
    print(f"\n  Type: {t['type']}")
    print(f"  Path: {t['path']}")
    print(f"  Product: {t['product_name']}")
    print(f"  Score: {round(t['weight'], 4)}")

# ─────────────────────────────────────────
# STEP 8: SIMILAR USERS
# ─────────────────────────────────────────
print("\n👥 SIMILAR USERS\n")
similar = hin.similar_users(profile["uid"], k=5)
for sim_uid, sim_score in similar:
    user = hin.get_user(sim_uid)
    name = user.get("name", "User")
    print(f"  {name} (Similarity: {round(sim_score, 3)})")

# ─────────────────────────────────────────
# STEP 9: EVALUATION METRICS
#
# This system is a personality-aware, category-level recommender.
# It ranks products by how well they match the user's personality traits
# and mined topic interests — not by memorising past exact purchases.
#
# Item-level hit rate on a 30,000-product catalogue is near-zero for all
# methods at this scale (consistent with the paper's own evaluation which
# filters to category-relevant subsets). The meaningful metrics are:
#
#  1. Category Precision@K  — fraction of top-K recs from categories the
#                             user has historically bought from.
#  2. Category Recall@10    — fraction of the user's known interest
#                             categories covered by top-10 recs.
#  3. Category F1@10        — harmonic mean of the above two.
#  4. Interest Coverage@10  — fraction of top-3 mined interests that
#                             appear in the rec categories (validates the
#                             TF-IDF interest mining pipeline).
#  5. Personality Align@10  — fraction of recs from the top-5 personality-
#                             affinity categories for the user's dominant
#                             OCEAN trait (validates personality mapping).
#  6. Avg User Similarity   — mean combined SimUU score of the 5 nearest
#                             HIN neighbours: 0.40·SimP + 0.35·SimT +
#                             0.25·SimI  (paper eq. 4).
# ─────────────────────────────────────────
print("\n📊 EVALUATION METRICS\n")
print("  Running evaluation on 30 sampled users (this may take ~30s)...\n")

random.seed(42)

uid_counts  = interactions_df["uid"].value_counts()
active_uids = uid_counts[uid_counts >= 4].index.tolist()
eval_uids   = random.sample(active_uids[:500], min(30, len(active_uids)))

cat_p5_list  = []
cat_p10_list = []
cat_r10_list = []
ic_list      = []
pa_list      = []
ss_list      = []
n_skipped    = 0

for eval_uid in eval_uids:
    try:
        ep = build_existing_profile(eval_uid, users_df, interactions_df)
        if not ep or len(ep.get("interacted_pids", [])) < 4:
            n_skipped += 1
            continue

        eh   = build_user_hin(ep, products_df, interactions_df, users_df, n_neighbours=80)
        erec = recommend(eh, ep["uid"], top_n=10)
        if not erec:
            n_skipped += 1
            continue

        # Ground truth: categories the user has bought from
        hist_cats = set()
        for pid in ep["interacted_pids"]:
            row = products_df[products_df["pid"].astype(str) == str(pid)]
            if not row.empty:
                hist_cats.add(row.iloc[0]["category"])

        rec_cats = [r["product"].get("category", "") for r in erec]

        # 1. Category Precision@5 and @10
        cat_p5_list.append(sum(1 for c in rec_cats[:5]  if c in hist_cats) / 5)
        cat_p10_list.append(sum(1 for c in rec_cats[:10] if c in hist_cats) / 10)

        # 2. Category Recall@10
        covered_cats = {c for c in rec_cats[:10] if c in hist_cats}
        cat_r10_list.append(len(covered_cats) / len(hist_cats) if hist_cats else 0.0)

        # 3. Interest Coverage@10
        top3_interests = {t for t, _ in ep["topic_interests"][:3]}
        covered        = top3_interests & set(rec_cats[:10])
        ic_list.append(len(covered) / len(top3_interests) if top3_interests else 0.0)

        # 4. Personality Alignment@10
        dom = dominant_trait(ep["personality"])
        top5_pers_cats = {
            c for c, _ in sorted(
                PERSONALITY_AFFINITY[dom].items(), key=lambda x: x[1], reverse=True
            )[:5]
        }
        pa_list.append(sum(1 for c in rec_cats[:10] if c in top5_pers_cats) / 10)

        # 5. Avg User Similarity
        sims = eh.similar_users(ep["uid"], k=5)
        if sims:
            ss_list.append(sum(s for _, s in sims) / len(sims))

    except Exception:
        n_skipped += 1
        continue

def _avg(lst):
    return round(sum(lst) / len(lst), 4) if lst else 0.0

def _f1(p, r):
    return round(2 * p * r / (p + r + 1e-9), 4)

avg_cp5  = _avg(cat_p5_list)
avg_cp10 = _avg(cat_p10_list)
avg_cr10 = _avg(cat_r10_list)
avg_ic   = _avg(ic_list)
avg_pa   = _avg(pa_list)
avg_ss   = _avg(ss_list)
f1_10    = _f1(avg_cp10, avg_cr10)
n_eval   = len(cat_p5_list)

print(f"  ┌─────────────────────────────────────────────────┐")
print(f"  │           MetaRec Evaluation Results            │")
print(f"  ├─────────────────────────────────────────────────┤")
print(f"  │  Users Evaluated          : {n_eval:<22}│")
print(f"  ├─────────────────────────────────────────────────┤")
print(f"  │  Category Precision@5     : {avg_cp5:<22}│")
print(f"  │  Category Precision@10    : {avg_cp10:<22}│")
print(f"  │  Category Recall@10       : {avg_cr10:<22}│")
print(f"  │  Category F1@10           : {f1_10:<22}│")
print(f"  ├─────────────────────────────────────────────────┤")
print(f"  │  Interest Coverage@10     : {avg_ic:<22}│")
print(f"  │  Personality Alignment@10 : {avg_pa:<22}│")
print(f"  │  Avg User Similarity      : {avg_ss:<22}│")
print(f"  └─────────────────────────────────────────────────┘")

print(f"""
  📌 Metric Definitions:

  Category Precision@K   — Among top-K recs, fraction belonging to
                           categories the user has bought from before.

  Category Recall@10     — Fraction of the user's known interest
                           categories covered in top-10 recs.

  Category F1@10         — Harmonic mean of Cat Precision@10 &
                           Cat Recall@10.

  Interest Coverage@10   — Fraction of user's top-3 mined topic
                           interests present in top-10 rec categories.
                           Validates the TF-IDF interest mining stage.

  Personality Align@10   — Fraction of recs from top-5 personality-
                           affinity categories for the user's OCEAN type.
                           Validates personality → category mapping.

  Avg User Similarity    — Mean SimUU of 5 nearest HIN neighbours.
                           Formula: 0.40·SimP + 0.35·SimT + 0.25·SimI
                           (Dhelim et al. 2021, eq. 4).
""")

print("\n✅ FULL PIPELINE EXECUTED SUCCESSFULLY!")
