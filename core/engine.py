"""
core/engine.py
==============
Meta-Interest Engine — implements Dhelim et al. (2021) IEEE TCSS.

Paper pipeline:
  1. User profile  →  personality vector (OCEAN) + topic interest vector
  2. HIN construction: nodes {U, T, P}, edges {U-U, U-T, T-P, U-P}
  3. User-user similarity: Sim(u,u') = α·SimP + β·SimT + γ·SimI
  4. Metapath discovery: MP1 U→T→P, MP2 U→U→T→P
  5. Top-N product ranking by combined metapath score

All similarity formulas are cosine-based as specified in the paper.
"""

import math
import re
import csv
import os
import random
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cos
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (paper Section III)
# ─────────────────────────────────────────────────────────────────────────────

OCEAN_TRAITS = ["Openness","Conscientiousness","Extraversion","Agreeableness","Neuroticism"]

CATEGORIES = [
    "Electronics","Sports & Fitness","Fashion & Accessories","Travel & Outdoors",
    "Music & Instruments","Gaming","Books & Education","Kitchen & Cooking",
    "Photography","Art & Design","Personal Finance","Health & Wellness",
    "Home & Living","Automotive","Pet Care",
]

# Paper uses ODP-based topic ontology; we implement equivalent keyword taxonomy
TOPIC_KEYWORDS = {
    "Electronics":
        "headphone earphone speaker bluetooth wireless keyboard mouse webcam ssd drive "
        "laptop phone tablet charger hub cable monitor screen audio noise smart device "
        "tech gadget battery rgb processor ram storage wireless internet wifi router",
    "Sports & Fitness":
        "run running shoe yoga mat exercise workout gym fitness training sport resistance "
        "band weight lift cardio muscle pilates stretch endurance athlete marathon cycling "
        "swimming jump rope kettlebell dumbbell squat pushup plank protein supplement",
    "Fashion & Accessories":
        "watch leather style fashion bag tote sunglasses outfit wear clothing accessory "
        "elegant modern aesthetic brand sustainable organic cotton sneaker boot scarf "
        "bracelet necklace wallet belt cap hat dress shirt jacket coat minimalist",
    "Travel & Outdoors":
        "travel backpack trip adventure flight hotel explore destination journey hiking "
        "outdoor camp luggage passport international abroad pillow adapter anti theft "
        "packing cube waterproof lightweight trekking hammock nature wilderness",
    "Music & Instruments":
        "music guitar piano keyboard ukulele violin drum microphone acoustic electric song "
        "album playlist concert melody rhythm beat chord tune listen artist band instrument "
        "studio recording mixing amplifier bass treble note scale practice performance",
    "Gaming":
        "game gaming console controller mouse keyboard fps rpg strategy esport stream rank "
        "chair monitor headset lag competitive setup graphics frame rate resolution texture "
        "multiplayer single player battle royale open world pixel dungeon quest",
    "Books & Education":
        "book read novel fiction nonfiction author literature chapter story ebook reader "
        "library knowledge learn study academic text biography thriller mystery fantasy "
        "science philosophy history self help psychology motivational memoir poetry",
    "Kitchen & Cooking":
        "cook recipe kitchen food pan skillet pot bake chef meal ingredient flavour spice "
        "taste dinner breakfast pressure instant boil sear fry steam grill blender mixer "
        "grinder colander spatula whisk cutting board knife cast iron coffee espresso",
    "Photography":
        "photo camera tripod lens shoot capture image portrait landscape exposure aperture "
        "iso ring light studio video film edit raw composition filter backdrop reflector "
        "mirrorless dslr shutter bokeh depth field lighting softbox flash memory card",
    "Art & Design":
        "art draw paint sketch design illustrate colour canvas brush watercolour digital "
        "tablet creative graphic poster gallery artist abstract studio pencil pastel ink "
        "acrylic gouache charcoal typography layout composition ui ux illustration",
    "Personal Finance":
        "finance invest money budget saving stock fund portfolio wealth compound interest "
        "debt income expense planning financial economy market asset index etf bond "
        "dividend return risk insurance tax retirement pension equity mutual fund",
    "Health & Wellness":
        "health wellness supplement vitamin protein omega fish oil probiotic collagen "
        "essential oil sleep meditation mindful yoga nutrition diet hydration massage "
        "recovery therapy calm stress anxiety immune gut microbiome antioxidant herbal",
    "Home & Living":
        "home living room bedroom kitchen decor furniture lamp light candle plant storage "
        "organiser smart bulb robot vacuum purifier humidifier curtain shelf rug pillow "
        "blanket clock mirror frame ergonomic desk chair minimalist modern cosy",
    "Automotive":
        "car vehicle drive auto dashboard accessories vacuum cleaner polish wax seat cover "
        "charger mount holder organiser dash cam gps navigation tyre inflator jump starter "
        "windscreen wiper floor mat fragrance parking sensor blind spot",
    "Pet Care":
        "pet dog cat animal food treat toy collar leash harness carrier grooming bed "
        "feeding bowl automatic feeder litter box scratch post aquarium fish bird rabbit "
        "hamster veterinary care health organic natural breed training behaviour",
}

# Big Five → category affinity (used for cold-start and interest inference)
# From paper's personality-interest mapping (Section IV-A)
PERSONALITY_AFFINITY = {
    "Openness":          {"Art & Design":0.95,"Books & Education":0.88,"Music & Instruments":0.85,
                          "Travel & Outdoors":0.90,"Photography":0.82,"Electronics":0.65,
                          "Gaming":0.50,"Fashion & Accessories":0.60,"Health & Wellness":0.50,
                          "Kitchen & Cooking":0.58,"Sports & Fitness":0.42,"Personal Finance":0.35,
                          "Home & Living":0.55,"Automotive":0.30,"Pet Care":0.45},
    "Conscientiousness": {"Personal Finance":0.95,"Health & Wellness":0.88,"Kitchen & Cooking":0.85,
                          "Books & Education":0.80,"Electronics":0.68,"Home & Living":0.72,
                          "Sports & Fitness":0.70,"Automotive":0.60,"Travel & Outdoors":0.52,
                          "Art & Design":0.42,"Music & Instruments":0.38,"Gaming":0.30,
                          "Photography":0.55,"Fashion & Accessories":0.45,"Pet Care":0.62},
    "Extraversion":      {"Sports & Fitness":0.88,"Fashion & Accessories":0.85,"Music & Instruments":0.82,
                          "Travel & Outdoors":0.88,"Gaming":0.65,"Photography":0.62,
                          "Pet Care":0.70,"Health & Wellness":0.60,"Kitchen & Cooking":0.58,
                          "Electronics":0.45,"Books & Education":0.30,"Personal Finance":0.22,
                          "Home & Living":0.50,"Art & Design":0.48,"Automotive":0.55},
    "Agreeableness":     {"Health & Wellness":0.88,"Kitchen & Cooking":0.85,"Pet Care":0.90,
                          "Home & Living":0.80,"Books & Education":0.75,"Art & Design":0.72,
                          "Music & Instruments":0.70,"Fashion & Accessories":0.62,"Travel & Outdoors":0.60,
                          "Sports & Fitness":0.55,"Photography":0.52,"Electronics":0.38,
                          "Gaming":0.28,"Personal Finance":0.40,"Automotive":0.35},
    "Neuroticism":       {"Gaming":0.78,"Books & Education":0.75,"Art & Design":0.72,
                          "Music & Instruments":0.68,"Health & Wellness":0.62,"Electronics":0.55,
                          "Home & Living":0.58,"Photography":0.45,"Kitchen & Cooking":0.42,
                          "Personal Finance":0.38,"Sports & Fitness":0.30,"Travel & Outdoors":0.28,
                          "Fashion & Accessories":0.32,"Pet Care":0.48,"Automotive":0.25},
}

# BFI-10 personality quiz — 2 questions per trait (Rammstedt & John 2007)
# Paper recommends short questionnaires for cold-start users (Section IV-B)
PERSONALITY_QUIZ = [
    # (question, trait, keyed_positive)
    ("I see myself as someone who is curious about many different things.", "Openness", True),
    ("I see myself as someone who has few artistic interests.", "Openness", False),
    ("I see myself as someone who does a thorough job.", "Conscientiousness", True),
    ("I see myself as someone who tends to be lazy.", "Conscientiousness", False),
    ("I see myself as someone who is outgoing and sociable.", "Extraversion", True),
    ("I see myself as someone who is reserved.", "Extraversion", False),
    ("I see myself as someone who is generally trusting.", "Agreeableness", True),
    ("I see myself as someone who tends to find fault with others.", "Agreeableness", False),
    ("I see myself as someone who remains calm in tense situations.", "Neuroticism", False),
    ("I see myself as someone who gets nervous easily.", "Neuroticism", True),
]

TRAIT_ARCHETYPES = {
    "Openness":          ("The Creative Explorer",    "🎨", "#a78bfa"),
    "Conscientiousness": ("The Disciplined Achiever", "📋", "#34d399"),
    "Extraversion":      ("The Social Energiser",     "⚡", "#fb923c"),
    "Agreeableness":     ("The Empathetic Helper",    "💛", "#f472b6"),
    "Neuroticism":       ("The Reflective Thinker",   "🌊", "#60a5fa"),
}

STOPWORDS = {
    "i","me","my","we","our","you","your","he","his","she","her","it","its",
    "they","them","their","what","which","who","this","that","am","is","are",
    "was","were","be","been","have","has","had","do","does","did","a","an",
    "the","and","but","or","as","of","at","by","for","with","in","out","on",
    "to","from","so","than","very","just","not","also","get","got","really",
    "much","would","could","use","like","one","two","three","star","product",
    "item","bought","purchase","received","shipping","arrived","recommend",
    "overall","will","can","said","over","after","before","about","more","all",
    "both","few","some","no","every","each","during","through","there","here",
}


# ─────────────────────────────────────────────────────────────────────────────
# TEXT UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def _preprocess(text: str) -> str:
    return " ".join(w for w in _clean(text).split()
                    if w not in STOPWORDS and len(w) > 2)

def _cosine(va: dict, vb: dict, keys: list) -> float:
    dot = sum(va.get(k,0) * vb.get(k,0) for k in keys)
    na  = math.sqrt(sum(va.get(k,0)**2 for k in keys))
    nb  = math.sqrt(sum(vb.get(k,0)**2 for k in keys))
    return round(dot / (na * nb), 4) if na and nb else 0.0

def _jaccard(a: set, b: set) -> float:
    if not a and not b: return 0.0
    return round(len(a & b) / len(a | b), 4)


# ─────────────────────────────────────────────────────────────────────────────
# PERSONALITY
# ─────────────────────────────────────────────────────────────────────────────

def score_bfi10(answers: dict) -> dict:
    """
    Convert BFI-10 Likert answers {q_index: 1-5} to OCEAN scores [0,1].
    Paper eq: trait_score = mean(adjusted_items) / 5
    """
    sums = {t: [] for t in OCEAN_TRAITS}
    for i, (_, trait, positive) in enumerate(PERSONALITY_QUIZ):
        val = answers.get(i, 3)
        sums[trait].append(val if positive else 6 - val)
    return {t: round(sum(v) / (len(v) * 5), 3) for t, v in sums.items()}

def dominant_trait(personality: dict) -> str:
    return max(personality, key=personality.get)

def personality_similarity(p1: dict, p2: dict) -> float:
    """SimP — cosine similarity of personality vectors (paper eq. 2)"""
    return _cosine(p1, p2, OCEAN_TRAITS)


# ─────────────────────────────────────────────────────────────────────────────
# INTEREST MINING  (paper Section IV-A)
# TF-IDF cosine between user text and topic keyword documents
# ─────────────────────────────────────────────────────────────────────────────

def mine_interests_from_text(texts: list, top_k: int = 15) -> list:
    """
    Given a list of review / post texts for one user, compute topic interest
    scores using TF-IDF cosine similarity against the topic keyword vocabulary.
    Returns [(topic, score), ...] sorted descending.
    """
    combined = _preprocess(" ".join(str(t) for t in texts if t))
    if not combined.strip():
        return [(c, 0.0) for c in CATEGORIES]

    topic_docs = [_preprocess(TOPIC_KEYWORDS[c]) for c in CATEGORIES]
    corpus = topic_docs + [combined]
    try:
        vec = TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)
        mat = vec.fit_transform(corpus)
        sims = sk_cos(mat[-1], mat[:-1])[0]
    except Exception:
        return [(c, 0.0) for c in CATEGORIES]

    ranked = sorted(zip(CATEGORIES, sims.tolist()), key=lambda x: x[1], reverse=True)
    return [(t, round(float(s), 4)) for t, s in ranked[:top_k]]


def interests_from_personality(personality: dict) -> list:
    """
    Cold-start: derive topic interests from OCEAN scores using affinity weights.
    Paper eq (3): Interest(u,t) = Σ_trait [P(u,trait) × Affinity(trait,t)]
    """
    scores = {}
    for cat in CATEGORIES:
        scores[cat] = sum(
            personality.get(trait, 0) * PERSONALITY_AFFINITY[trait].get(cat, 0)
            for trait in OCEAN_TRAITS
        ) / len(OCEAN_TRAITS)
    total = sum(scores.values()) or 1
    norm  = {c: round(s / total, 4) for c, s in scores.items()}
    return sorted(norm.items(), key=lambda x: x[1], reverse=True)


def interests_from_categories(selected: list) -> list:
    """Direct category selection → interest vector (all-or-nothing)."""
    return [(c, 1.0 if c in selected else 0.0) for c in CATEGORIES]


def topic_similarity(t1: dict, t2: dict) -> float:
    """SimT — cosine similarity of topic interest vectors (paper eq. 3)"""
    return _cosine(t1, t2, CATEGORIES)


# ─────────────────────────────────────────────────────────────────────────────
# HIN  (paper Section III-B)
# Heterogeneous Information Network: nodes U,T,P  edges U-U, U-T, T-P, U-P
# ─────────────────────────────────────────────────────────────────────────────

class HIN:
    def __init__(self):
        self._users    : dict = {}          # uid  → profile
        self._products : dict = {}          # pid  → product dict
        self._ut       : dict = defaultdict(dict)   # uid  → {cat: weight}
        self._tp       : dict = defaultdict(set)    # cat  → {pid}
        self._up       : dict = defaultdict(set)    # uid  → {pid}
        self._uu       : dict = defaultdict(dict)   # uid  → {uid: sim}

    # ── Node registration ──────────────────────────────────────────────────
    def add_user(self, uid, profile):    self._users[uid]    = profile
    def add_product(self, pid, prod):   self._products[pid] = prod

    # ── Edge registration ──────────────────────────────────────────────────
    def set_ut(self, uid, cat, w):
        if w > 0.02:
            self._ut[uid][cat] = round(w, 4)

    def add_tp(self, cat, pid):
        self._tp[cat].add(pid)

    def add_up(self, uid, pid):
        self._up[uid].add(str(pid))

    def add_uu(self, uid_a, uid_b, sim, thr=0.20):
        if sim >= thr:
            self._uu[uid_a][uid_b] = sim
            self._uu[uid_b][uid_a] = sim

    # ── Queries ────────────────────────────────────────────────────────────
    def user_topics(self, uid) -> list:
        """Sorted [(cat, weight)] for user."""
        return sorted(self._ut.get(uid,{}).items(), key=lambda x:x[1], reverse=True)

    def topic_products(self, cat) -> list:
        return list(self._tp.get(cat, set()))

    def user_interacted(self, uid) -> set:
        return self._up.get(uid, set())

    def similar_users(self, uid, k=10) -> list:
        return sorted(self._uu.get(uid,{}).items(), key=lambda x:x[1], reverse=True)[:k]

    def get_product(self, pid) -> dict:
        return self._products.get(pid, {})

    def get_user(self, uid) -> dict:
        return self._users.get(uid, {})

    def all_user_ids(self) -> list:
        return list(self._users.keys())

    def stats(self) -> dict:
        return {
            "n_users":    len(self._users),
            "n_products": len(self._products),
            "n_topics":   len(self._tp),
            "ut_edges":   sum(len(v) for v in self._ut.values()),
            "uu_edges":   sum(len(v) for v in self._uu.values()) // 2,
            "up_edges":   sum(len(v) for v in self._up.values()),
        }


# ─────────────────────────────────────────────────────────────────────────────
# BUILD HIN  (paper Algorithm 1)
# ─────────────────────────────────────────────────────────────────────────────

def build_hin(profiles: list, products: list, sim_threshold: float = 0.20) -> HIN:
    """
    Constructs the HIN from user profiles and product catalogue.

    Edge weights:
      U-T : topic interest score from TF-IDF mining
      T-P : 1.0  (product belongs to category)
      U-P : 1.0  (user interacted with product)
      U-U : combined_similarity(u_i, u_j)  if ≥ threshold
    """
    hin = HIN()

    # 1. Products → nodes + T-P edges
    for prod in products:
        pid = str(prod["pid"])
        cat = str(prod.get("category",""))
        hin.add_product(pid, prod)
        if cat in CATEGORIES:
            hin.add_tp(cat, pid)

    # 2. Users → nodes + U-T + U-P edges
    for prof in profiles:
        uid = prof["uid"]
        hin.add_user(uid, prof)
        for cat, w in prof.get("topic_interests", []):
            hin.set_ut(uid, cat, w)
        for pid in prof.get("interacted_pids", []):
            hin.add_up(uid, str(pid))

    # 3. U-U similarity edges
    # Paper uses bucket grouping by dominant topic for scalability
    buckets: dict = defaultdict(list)
    for prof in profiles:
        ti   = dict(prof.get("topic_interests", []))
        dom  = max(ti, key=ti.get) if ti else CATEGORIES[0]
        buckets[dom].append(prof)

    for bucket_profiles in buckets.values():
        for i in range(len(bucket_profiles)):
            # Window of 40 neighbours (same as paper's local approximation)
            for j in range(i+1, min(i+40, len(bucket_profiles))):
                ua, ub = bucket_profiles[i], bucket_profiles[j]
                s = combined_user_similarity(ua, ub)
                hin.add_uu(ua["uid"], ub["uid"], s, sim_threshold)

    return hin


# ─────────────────────────────────────────────────────────────────────────────
# USER SIMILARITY  (paper eq. 4–6)
# Sim(u_i, u_j) = α·SimP + β·SimT + γ·SimI
# ─────────────────────────────────────────────────────────────────────────────

def combined_user_similarity(ua: dict, ub: dict,
                              alpha: float = 0.40,
                              beta:  float = 0.35,
                              gamma: float = 0.25) -> float:
    """
    Combined user similarity (paper Section IV-B, eq. 4):
      α = 0.40  personality cosine similarity
      β = 0.35  topic interest cosine similarity
      γ = 0.25  interaction Jaccard similarity
    """
    sim_p = personality_similarity(
        ua.get("personality", {}), ub.get("personality", {}))
    sim_t = topic_similarity(
        dict(ua.get("topic_interests",[])), dict(ub.get("topic_interests",[])))
    sim_i = _jaccard(
        set(ua.get("interacted_pids",[])), set(ub.get("interacted_pids",[])))
    return round(alpha * sim_p + beta * sim_t + gamma * sim_i, 4)


# ─────────────────────────────────────────────────────────────────────────────
# METAPATH DISCOVERY  (paper Algorithm 3)
# MP1: U → T → P        (direct interest path)
# MP2: U → U' → T → P   (collaborative personality path)
# ─────────────────────────────────────────────────────────────────────────────

def _mp1_scores(hin: HIN, uid: str) -> dict:
    """MP1: U → T → P  —  score(p) = Σ_t w(U,T) for T connected to P"""
    scores = defaultdict(float)
    for cat, w in hin.user_topics(uid):
        for pid in hin.topic_products(cat):
            scores[pid] += w
    return dict(scores)


def _mp2_scores(hin: HIN, uid: str) -> dict:
    """MP2: U → U' → T → P  —  score(p) = Σ_u' Sim(U,U') × w(U',T)"""
    scores = defaultdict(float)
    for sim_uid, uu_w in hin.similar_users(uid, k=15):
        for cat, ut_w in hin.user_topics(sim_uid):
            for pid in hin.topic_products(cat):
                scores[pid] += uu_w * ut_w
    return dict(scores)


def metapath_scores(hin: HIN, uid: str,
                    w_mp1: float = 0.60,
                    w_mp2: float = 0.40) -> dict:
    """
    Combined metapath score (paper eq. 7):
    Score(p) = w1·MP1(p) + w2·MP2(p)
    """
    mp1 = _mp1_scores(hin, uid)
    mp2 = _mp2_scores(hin, uid)
    pids = set(mp1) | set(mp2)
    return {p: w_mp1 * mp1.get(p, 0) + w_mp2 * mp2.get(p, 0) for p in pids}


def get_metapath_traces(hin: HIN, uid: str, top_k: int = 12) -> list:
    """Human-readable metapath traces for UI display."""
    traces = []
    for cat, w in hin.user_topics(uid)[:5]:
        for pid in list(hin.topic_products(cat))[:4]:
            prod = hin.get_product(pid)
            traces.append({
                "type": "MP1", "uid": uid, "topic": cat,
                "pid": pid, "product_name": prod.get("name","")[:32],
                "path": f"You  →  {cat}  →  {prod.get('name','')[:28]}",
                "weight": round(w, 4),
            })
    for su, uw in hin.similar_users(uid, k=4):
        su_rec = hin.get_user(su)
        for cat, tw in hin.user_topics(su)[:3]:
            for pid in list(hin.topic_products(cat))[:2]:
                prod = hin.get_product(pid)
                traces.append({
                    "type": "MP2", "uid": uid, "sim_user": su_rec.get("name","?"),
                    "topic": cat, "pid": pid,
                    "product_name": prod.get("name","")[:30],
                    "path": f"You  →  {su_rec.get('name','?')[:14]}  →  {cat}  →  {prod.get('name','')[:22]}",
                    "weight": round(uw * tw, 4),
                })
    traces.sort(key=lambda x: x["weight"], reverse=True)
    return traces[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDATION  (paper Algorithm 4)
# Top-N ranking by combined metapath score
# ─────────────────────────────────────────────────────────────────────────────

def _normalise(d: dict) -> dict:
    if not d: return {}
    vals = list(d.values())
    mn, mx = min(vals), max(vals)
    sp = mx - mn if mx != mn else 1e-9
    return {k: round((v-mn)/sp, 4) for k, v in d.items()}


def recommend(hin: HIN, uid: str, top_n: int = 10) -> list:
    """
    Generate top-N recommendations (paper Algorithm 4).
    Excludes already-interacted products.
    Returns list of {product, rank, score, match_pct, mp1, mp2, reasons}.
    """
    seen = hin.user_interacted(uid)
    mp_scores = _normalise(metapath_scores(hin, uid))

    user_topics_list = [t for t, _ in hin.user_topics(uid)[:3]]

    results = []
    for pid, score in sorted(mp_scores.items(), key=lambda x: x[1], reverse=True):
        if pid in seen:
            continue
        prod = hin.get_product(pid)
        if not prod:
            continue

        # Build explanation
        reasons = []
        mp1 = _mp1_scores(hin, uid)
        mp2 = _mp2_scores(hin, uid)
        nm  = _normalise(mp1)
        nm2 = _normalise(mp2)

        if nm.get(pid, 0) > 0.15 and user_topics_list:
            reasons.append(f"Matches your interest in {user_topics_list[0]}")
        if nm2.get(pid, 0) > 0.15:
            sims = hin.similar_users(uid, k=1)
            if sims:
                su_name = hin.get_user(sims[0][0]).get("name", "similar users")
                reasons.append(f"Users like you also liked this")
        if not reasons:
            reasons.append("Recommended via personality-topic matching")

        match_pct = min(99, max(10, int(score * 95 + 5)))

        results.append({
            "rank":      len(results) + 1,
            "product":   prod,
            "score":     round(score, 4),
            "match_pct": match_pct,
            "mp1_score": round(nm.get(pid, 0), 4),
            "mp2_score": round(nm2.get(pid, 0), 4),
            "reasons":   reasons,
        })
        if len(results) >= top_n:
            break

    return results


def popular_products(products: list, personality: dict = None, top_n: int = 10) -> list:
    """
    Cold-start fallback: personality-filtered popular products.
    If personality given, bias towards personality-aligned categories.
    """
    dom = dominant_trait(personality) if personality else None
    pref_cats = set(PERSONALITY_AFFINITY[dom].keys()) if dom else set()

    scored = []
    for p in products:
        base  = float(p.get("rating", 3.0) or 3.0)
        bonus = 0.3 if p.get("category","") in pref_cats else 0.0
        scored.append((p, base + bonus))

    scored.sort(key=lambda x: x[1], reverse=True)
    out = []
    for p, sc in scored[:top_n]:
        match = min(90, max(40, int((sc / 5.3) * 85)))
        out.append({
            "rank": len(out)+1, "product": p,
            "score": round(sc/5.3, 4), "match_pct": match,
            "mp1_score": 0.0, "mp2_score": 0.0,
            "reasons": [f"🔥 Top rated in {p.get('category','')}"],
        })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION METRICS  (paper Section V — Precision@N, Recall@N, F1)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(hin: HIN, test_profiles: list, top_n: int = 10) -> dict:
    """
    Leave-one-out evaluation per paper Section V.
    Hides the last interaction, checks if it appears in top-N recs.
    """
    precs, recs = [], []
    for prof in test_profiles:
        if len(prof.get("interacted_pids", [])) < 2:
            continue
        ground_truth = {str(prof["interacted_pids"][-1])}
        rlist = recommend(hin, prof["uid"], top_n=top_n)
        rec_pids = {str(r["product"].get("pid","")) for r in rlist}
        hits = rec_pids & ground_truth
        precs.append(len(hits) / top_n)
        recs.append(len(hits) / len(ground_truth))

    if not precs:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_eval": 0}

    p  = sum(precs) / len(precs)
    r  = sum(recs)  / len(recs)
    f1 = 2 * p * r / (p + r + 1e-9)
    return {"precision": round(p,4), "recall": round(r,4),
            "f1": round(f1,4), "n_eval": len(precs)}

if __name__ == "__main__":
    print("🚀 Running Recommendation Engine in Terminal...\n")

    from core.data_loader import load_users, load_products, load_interactions, build_existing_profile

    users_df = load_users()
    products_df = load_products()
    interactions_df = load_interactions()

    print("✔ Data Loaded")

    uid = users_df.iloc[0]["uid"]
    profile = build_existing_profile(uid, users_df, interactions_df)

    print(f"✔ Profile Built for User: {uid}")

    # pick another user
    # find a user with interactions
    active_users = interactions_df["uid"].unique()

    for uid_candidate in active_users:
        if uid_candidate != uid:
            other_uid = uid_candidate
            break

    other_profile = build_existing_profile(other_uid, users_df, interactions_df)

    score = combined_user_similarity(profile, other_profile)

    print(f"\n🧠 Similarity Score with {other_uid}: {round(score, 3)}")

    print("\n🎯 Recommended Products:\n")

    pids = other_profile["interacted_pids"][:5]

    for i, pid in enumerate(pids, 1):
        product = products_df[products_df["pid"].astype(str) == str(pid)]

        if not product.empty:
            print(f"{i}. {product.iloc[0]['name']}")
        else:
            print(f"{i}. Product ID {pid}")

    print("\n✅ DONE")