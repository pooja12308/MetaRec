"""
core/data_loader.py
Efficient data loading with caching.
Builds user profiles for both existing and new users.
"""
import os, json, random
import pandas as pd
import streamlit as st

from core.engine import (
    mine_interests_from_text, interests_from_personality,
    interests_from_categories, score_bfi10, dominant_trait,
    CATEGORIES, OCEAN_TRAITS, PERSONALITY_AFFINITY, TRAIT_ARCHETYPES,
    build_hin, combined_user_similarity,
)

DATA_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
USERS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "users_db")
os.makedirs(USERS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_products() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA_DIR, "products.csv"), dtype=str)
    df["price"]  = pd.to_numeric(df["price"],  errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def load_users() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, "users.csv"), dtype=str)

@st.cache_data(show_spinner=False)
def load_interactions() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA_DIR, "interactions.csv"), dtype=str)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# USER PROFILE BUILDING
# ─────────────────────────────────────────────────────────────────────────────

def _archetype_info(trait: str) -> tuple:
    return TRAIT_ARCHETYPES.get(trait, ("Explorer","🔮","#a78bfa"))


def build_existing_profile(uid: str,
                            users_df: pd.DataFrame,
                            interactions_df: pd.DataFrame) -> dict:
    """
    Build profile for an existing user from interactions.
    Case 1 (paper): user has interaction history → mine interests from review text.
    """
    user_row = users_df[users_df["uid"] == uid]
    if user_row.empty:
        return None
    row  = user_row.iloc[0]
    name = row["name"]

    # Use pre-assigned personality from dataset
    pers_type = row.get("personality_type", "Openness")
    pers = {t: round(random.gauss(
                0.75 if t == pers_type else 0.40, 0.10), 3)
            for t in OCEAN_TRAITS}
    pers[pers_type] = max(0.65, min(0.95, pers[pers_type]))

    user_ints = interactions_df[interactions_df["uid"] == uid]
    pids = user_ints["pid"].astype(str).tolist()
    reviews = user_ints["review"].dropna().tolist()

    # Topic interests from reviews (TF-IDF mining)
    if reviews:
        interests = mine_interests_from_text(reviews)
    else:
        interests = interests_from_personality(pers)

    dom   = dominant_trait(pers)
    arch  = _archetype_info(dom)
    return {
        "uid": uid, "name": name,
        "source": "existing",
        "personality": pers,
        "dominant_trait": dom,
        "archetype": arch[0], "archetype_emoji": arch[1], "archetype_color": arch[2],
        "topic_interests": interests,
        "interacted_pids": pids,
        "user_text": " ".join(reviews[:20]),
        "n_reviews": len(reviews),
    }


def build_new_user_profile(uid: str, name: str,
                            bfi_answers: dict,
                            selected_categories: list) -> dict:
    """
    Build cold-start profile from BFI-10 quiz + category selection.
    Case 2 (paper): new user, no history → questionnaire-based.
    """
    pers = score_bfi10(bfi_answers)
    dom  = dominant_trait(pers)
    arch = _archetype_info(dom)

    # Combine category selection with personality-derived interests
    cat_ints  = dict(interests_from_categories(selected_categories))
    pers_ints = dict(interests_from_personality(pers))
    # Weighted blend: 60% explicit selection, 40% personality inference
    blended = {}
    for c in CATEGORIES:
        blended[c] = round(0.60 * cat_ints.get(c,0) + 0.40 * pers_ints.get(c,0), 4)
    interests = sorted(blended.items(), key=lambda x: x[1], reverse=True)

    return {
        "uid": uid, "name": name,
        "source": "new_user",
        "personality": pers,
        "dominant_trait": dom,
        "archetype": arch[0], "archetype_emoji": arch[1], "archetype_color": arch[2],
        "topic_interests": interests,
        "interacted_pids": [],
        "user_text": " ".join(selected_categories),
        "n_reviews": 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PERSISTENT PROFILE STORE
# ─────────────────────────────────────────────────────────────────────────────

def save_new_user(profile: dict):
    path = os.path.join(USERS_DIR, f"{profile['uid']}.json")
    with open(path, "w") as f:
        json.dump(profile, f, indent=2)


def load_new_user(uid: str) -> dict | None:
    path = os.path.join(USERS_DIR, f"{uid}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def is_existing_user(uid: str, users_df: pd.DataFrame) -> bool:
    return not users_df[users_df["uid"] == uid].empty


# ─────────────────────────────────────────────────────────────────────────────
# HIN BUILDER (loads a sample neighbourhood for speed)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_products_list(_products_df: pd.DataFrame) -> list:
    """Convert products dataframe to list of dicts (cached)."""
    return _products_df.to_dict("records")


def build_user_hin(profile: dict,
                   products_df: pd.DataFrame,
                   interactions_df: pd.DataFrame,
                   users_df: pd.DataFrame,
                   n_neighbours: int = 200) -> object:
    """
    Build HIN for a single user with a neighbourhood of similar users.
    For scalability, we sample n_neighbours users from the same personality bucket.
    """
    uid     = profile["uid"]
    dom     = profile["dominant_trait"]
    products = get_products_list(products_df)

    # Sample neighbourhood: same dominant personality type
    same_type = users_df[users_df["personality_type"] == dom]["uid"].tolist()
    if uid in same_type: same_type.remove(uid)
    sample_uids = random.sample(same_type, min(n_neighbours, len(same_type)))

    # Build lightweight profiles for neighbours
    neighbour_profiles = []
    uid_set = set(sample_uids)
    nb_ints = interactions_df[interactions_df["uid"].isin(uid_set)]

    for nuid in sample_uids:
        nrow = users_df[users_df["uid"] == nuid]
        if nrow.empty: continue
        nrow  = nrow.iloc[0]
        npers = {t: round(random.gauss(
                    0.70 if t == dom else 0.38, 0.12), 3)
                 for t in OCEAN_TRAITS}
        npers[dom] = max(0.60, min(0.95, npers[dom]))
        n_pids = nb_ints[nb_ints["uid"] == nuid]["pid"].astype(str).tolist()
        n_revs = nb_ints[nb_ints["uid"] == nuid]["review"].dropna().tolist()
        nints  = mine_interests_from_text(n_revs) if n_revs else interests_from_personality(npers)
        neighbour_profiles.append({
            "uid": nuid, "name": nrow["name"],
            "personality": npers, "dominant_trait": dom,
            "topic_interests": nints, "interacted_pids": n_pids,
        })

    all_profiles = [profile] + neighbour_profiles
    return build_hin(all_profiles, products)
