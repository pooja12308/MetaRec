"""
MetaRec — Personality-Aware Product Recommendation System
=========================================================
Implements: Dhelim et al. (2021) IEEE Trans. Computational Social Systems
            "Personality-Aware Product Recommendation System Based on
             User Interests Mining and Metapath Discovery"

Run:  streamlit run app.py
"""

import sys, os, random
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="MetaRec · Personality-Aware Recommendations",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# WARM THEME CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800;900&display=swap');

*, *::before, *::after { font-family: 'Plus Jakarta Sans', sans-serif !important; box-sizing: border-box; }

/* ── Base ──────────────────────────────────────────────────── */
.stApp                { background: #0e0a07; }
.block-container      { padding: 2rem 2.5rem 2rem !important; max-width: 1280px !important; }
#MainMenu, footer, header, .stDeployButton { visibility: hidden; display: none; }

/* Scrollbar */
::-webkit-scrollbar       { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #1a1208; }
::-webkit-scrollbar-thumb { background: #8b5e3c; border-radius: 99px; }

/* ── Sidebar ───────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #150f08 0%, #0e0a07 100%) !important;
    border-right: 1px solid #2a1f14 !important;
}

/* ── Typography ────────────────────────────────────────────── */
h1, h2, h3, h4, h5 { color: #f5ede3 !important; letter-spacing: -0.025em; line-height: 1.2; }
p, li, span         { color: #e6dfd5; }

/* ── Cards ─────────────────────────────────────────────────── */
.warm-card {
    background: linear-gradient(145deg, #1a1209 0%, #140d06 100%);
    border: 1px solid #2e1f12;
    border-radius: 18px;
    padding: 22px 24px;
    margin-bottom: 14px;
    transition: border-color .22s, transform .18s, box-shadow .22s;
    position: relative;
    overflow: hidden;
}
.warm-card::after {
    content: '';
    position: absolute; top: 0; left: 0;
    width: 100%; height: 2px;
    background: linear-gradient(90deg, transparent, #c2813a, transparent);
    opacity: 0;
    transition: opacity .22s;
}
.warm-card:hover {
    border-color: #8b5e3c;
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(139,94,60,0.18);
}
.warm-card:hover::after { opacity: 1; }

.stat-card {
    background: linear-gradient(145deg, #1a1209, #140d06);
    border: 1px solid #2e1f12;
    border-radius: 14px;
    padding: 20px 16px;
    text-align: center;
}
.stat-num {
    font-size: 2rem; font-weight: 800;
    background: linear-gradient(135deg, #e8a45a, #c2813a);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1;
}
.stat-lbl { color: #d6c7b5; font-size: .78rem; margin-top: 4px; }

/* ── Progress bars ─────────────────────────────────────────── */
.bar-track { background: #1e1409; border-radius: 99px; height: 9px; width: 100%; margin: 4px 0; }
.bar-amber  { background: linear-gradient(90deg,#c2813a,#e8a45a); border-radius:99px; height:9px; }
.bar-sienna { background: linear-gradient(90deg,#a0522d,#cd7f32); border-radius:99px; height:9px; }
.bar-copper { background: linear-gradient(90deg,#b87333,#daa520); border-radius:99px; height:9px; }
.bar-rust   { background: linear-gradient(90deg,#8b4513,#cd853f); border-radius:99px; height:9px; }
.bar-warm   { background: linear-gradient(90deg,#d2691e,#e9967a); border-radius:99px; height:9px; }
.bar-sage   { background: linear-gradient(90deg,#6b8e5e,#8fbc8f); border-radius:99px; height:9px; }

/* ── Badges ────────────────────────────────────────────────── */
.badge-amber {
    background: rgba(194,129,58,0.15);
    color: #e8a45a; border: 1px solid rgba(194,129,58,0.3);
    border-radius: 20px; padding: 3px 12px;
    font-size: .72rem; font-weight: 700; display: inline-block;
}
.badge-topic {
    background: rgba(160,82,45,0.12); color: #cd9b6a;
    border: 1px solid rgba(160,82,45,0.25);
    border-radius: 20px; padding: 3px 10px;
    font-size: .72rem; display: inline-block; margin: 2px;
}
.badge-reason {
    background: rgba(107,142,94,0.12); color: #8fbc8f;
    border: 1px solid rgba(107,142,94,0.22);
    border-radius: 10px; padding: 2px 9px;
    font-size: .70rem; display: inline-block; margin: 2px;
}
.badge-rank {
    background: linear-gradient(135deg,#2e1f12,#3d2a18);
    color: #e8a45a; border: 1px solid #4a3020;
    border-radius: 8px; padding: 4px 10px;
    font-size: .72rem; font-weight: 800;
    display: inline-block; letter-spacing: .03em;
}

/* ── Inputs ────────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea textarea,
.stNumberInput input {
    background: #1a1209 !important; border: 1px solid #2e1f12 !important;
    color: #f5ede3 !important; border-radius: 10px !important;
    font-size: .88rem !important;
}
.stTextInput > div > div > input:focus,
.stTextArea textarea:focus {
    border-color: #c2813a !important;
    box-shadow: 0 0 0 2px rgba(194,129,58,.18) !important;
}
.stSelectbox > div > div {
    background: #1a1209 !important; border-color: #2e1f12 !important;
    color: #f5ede3 !important; border-radius: 10px !important;
}

/* ── Buttons ───────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #c2813a, #e8a45a) !important;
    color: #0e0a07 !important; border: none !important;
    border-radius: 10px !important; font-weight: 700 !important;
    letter-spacing: .02em !important; padding: .5rem 1.5rem !important;
    transition: opacity .18s, transform .14s !important;
}
.stButton > button:hover { opacity: .88 !important; transform: translateY(-1px) !important; }

/* ── Slider ────────────────────────────────────────────────── */
.stSlider [data-baseweb="slider"] [role="slider"] { background: #c2813a !important; }
.stSlider [data-baseweb="slider"] > div > div:first-child { background: #2e1f12 !important; }

/* ── Tabs ──────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #150f08; border-radius: 12px; padding: 4px;
    border: 1px solid #2e1f12; gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent; color: ##d6c7b5;
    border-radius: 9px; font-size: .84rem; font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#c2813a,#e8a45a) !important;
    color: #0e0a07 !important;
}

/* ── Metrics ───────────────────────────────────────────────── */
div[data-testid="metric-container"] {
    background: #1a1209; border: 1px solid #2e1f12; border-radius: 12px;
    padding: 14px 18px;
}
div[data-testid="metric-container"] label { color: #d6c7b5 !important; }
div[data-testid="metric-container"] [data-testid="metric-value"] { color: #e8a45a !important; }

/* ── Radio ─────────────────────────────────────────────────── */
.stRadio label { color: #9d8872 !important; font-size: .86rem !important; }

/* ── Expander ──────────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: #1a1209 !important; border: 1px solid #2e1f12 !important;
    border-radius: 10px !important; color: #9d8872 !important;
}

/* ── Divider ───────────────────────────────────────────────── */
hr { border-color: #2e1f12 !important; margin: 1.2rem 0 !important; }

/* ── Landing page ──────────────────────────────────────────── */
.hero-title {
    font-size: 3.2rem; font-weight: 900; letter-spacing: -0.04em;
    background: linear-gradient(135deg, #f5ede3 30%, #e8a45a 70%, #c2813a);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.1; margin-bottom: 16px;
}
.hero-sub {
    color: #d6c7b5; font-size: 1.05rem; line-height: 1.7; max-width: 540px;
}
.uid-box {
    background: linear-gradient(145deg, #1a1209, #140d06);
    border: 1.5px solid #2e1f12; border-radius: 16px;
    padding: 28px 28px 24px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.4);
}

/* ── Onboarding steps ──────────────────────────────────────── */
.step-track { display: flex; gap: 8px; align-items: center; margin-bottom: 28px; }
.step-pill {
    height: 5px; border-radius: 99px; flex: 1;
    background: #2e1f12; transition: background .3s;
}
.step-pill.done   { background: #c2813a; }
.step-pill.active { background: linear-gradient(90deg,#c2813a,#e8a45a); }

/* ── Archetype card ────────────────────────────────────────── */
.arch-card {
    background: linear-gradient(145deg, rgba(194,129,58,.12), rgba(232,164,90,.06));
    border: 1.5px solid rgba(194,129,58,.25);
    border-radius: 20px; padding: 30px 24px; text-align: center;
    margin-bottom: 24px;
}
.arch-title {
    font-size: 1.6rem; font-weight: 800;
    background: linear-gradient(135deg,#f5ede3,#e8a45a);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-top: 8px;
}

/* ── Rec card ──────────────────────────────────────────────── */
.rec-card {
    background: linear-gradient(145deg, #1a1209, #140d06);
    border: 1px solid #2e1f12; border-radius: 16px;
    padding: 18px 20px; margin-bottom: 12px;
    transition: all .2s; position: relative; overflow: hidden;
}
.rec-card::before {
    content: '';
    position: absolute; top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, #c2813a, #e8a45a);
}
.rec-card:hover { border-color: #8b5e3c; transform: translateX(4px);
                  box-shadow: 0 8px 30px rgba(194,129,58,.12); }
.prod-name  { color: #f5ede3; font-weight: 700; font-size: .92rem; margin-bottom: 2px; }
.prod-brand { color: #c2813a; font-size: .72rem; font-weight: 600; }
.prod-meta  { color: #d6c7b5; font-size: .75rem; margin: 4px 0 8px; }
.rank-num   {
    font-size: 1.8rem; font-weight: 900; line-height: 1;
    background: linear-gradient(135deg,#c2813a,#e8a45a);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}

/* ── Match ring ────────────────────────────────────────────── */
.match-ring-outer {
    width: 54px; height: 54px; border-radius: 50%; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    position: relative;
}
.match-ring-inner {
    width: 40px; height: 40px; border-radius: 50%; background: #0e0a07;
    display: flex; align-items: center; justify-content: center;
    position: absolute;
    color: #f5ede3; font-size: .72rem; font-weight: 800;
}

/* ── Score breakdown mini bars ─────────────────────────────── */
.score-row { display:flex; align-items:center; gap:8px; margin:4px 0; }
.score-lbl { color:#3d2a18; font-size:.68rem; width:80px; flex-shrink:0; }
.score-trk { flex:1; background:#1e1409; border-radius:99px; height:5px; }
.score-fill{ border-radius:99px; height:5px;
             background: linear-gradient(90deg,#c2813a,#e8a45a); }

/* ── Sidebar nav ───────────────────────────────────────────── */
.snav-item {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 14px; border-radius: 10px; margin-bottom: 4px;
    color: #d6c7b5; font-size: .86rem; font-weight: 500;
    cursor: pointer; transition: background .15s, color .15s;
    border: none; width: 100%; background: transparent;
}
.snav-item:hover { background: #1e1409; color: #c9a880; }
.snav-item.active {
    background: rgba(194,129,58,.12); color: #e8a45a;
    border-left: 3px solid #c2813a; padding-left: 11px;
}

/* ── Network graph ─────────────────────────────────────────── */
.path-chip {
    background: #1e1409; border: 1px solid #2e1f12;
    border-radius: 12px; padding: 10px 14px;
    margin-bottom: 8px; font-size: .8rem; color: #9d8872;
    border-left: 3px solid #c2813a;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS (after page config)
# ─────────────────────────────────────────────────────────────────────────────
from core.engine import (
    OCEAN_TRAITS, CATEGORIES, PERSONALITY_QUIZ, TRAIT_ARCHETYPES,
    PERSONALITY_AFFINITY, recommend, popular_products,
    get_metapath_traces, evaluate, dominant_trait, score_bfi10,
)
from core.data_loader import (
    load_products, load_users, load_interactions,
    build_existing_profile, build_new_user_profile,
    save_new_user, load_new_user, is_existing_user,
    build_user_hin, get_products_list,
)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
def _init():
    defaults = dict(
        page="landing", uid=None, profile=None, hin=None,
        top_n=10, ob_step=0, quiz_ans={}, sel_cats=[], ob_name="",
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ─────────────────────────────────────────────────────────────────────────────
# DATA (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_all():
    return load_products(), load_users(), load_interactions()

with st.spinner("Loading MetaRec…"):
    products_df, users_df, interactions_df = _load_all()

PRODUCTS_LIST = get_products_list(products_df)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

CATEGORY_ICONS = {
    "Electronics":"💻","Sports & Fitness":"🏃","Fashion & Accessories":"👗",
    "Travel & Outdoors":"✈️","Music & Instruments":"🎵","Gaming":"🎮",
    "Books & Education":"📚","Kitchen & Cooking":"🍳","Photography":"📷",
    "Art & Design":"🎨","Personal Finance":"💰","Health & Wellness":"🌿",
    "Home & Living":"🏠","Automotive":"🚗","Pet Care":"🐾",
}
TRAIT_BARS = ["bar-amber","bar-sienna","bar-copper","bar-rust","bar-warm"]

def go(page):
    st.session_state.page = page
    st.rerun()

def _stat(val, label):
    return f'<div class="stat-card"><div class="stat-num">{val}</div><div class="stat-lbl">{label}</div></div>'

def _bar(cls, pct):
    return f'<div class="bar-track"><div class="{cls}" style="width:{pct}%"></div></div>'

def _steps_ui(step, total=4):
    pills = ""
    for i in range(total):
        cls = "done" if i < step else ("active" if i == step else "")
        pills += f'<div class="step-pill {cls}"></div>'
    labels = ["Welcome","Personality Quiz","Your Interests","Your Results"]
    return f"""
    <div class="step-track">{pills}</div>
    <p style="text-align:center;color:#5c4a38;font-size:.78rem;margin-bottom:20px">
        Step {step+1} of {total} — {labels[min(step,len(labels)-1)]}
    </p>"""

def _render_rec_card(rec):
    prod  = rec["product"]
    name  = str(prod.get("name",""))[:50]
    brand = str(prod.get("brand",""))
    cat   = str(prod.get("category",""))
    icon  = CATEGORY_ICONS.get(cat,"📦")
    try:    price = f"₹{float(prod.get('price',0)):,.0f}"
    except: price = str(prod.get("price","—"))
    rating    = prod.get("rating","—")
    rank      = rec["rank"]
    match_pct = rec["match_pct"]
    reasons   = rec.get("reasons",[])
    mp1       = int(rec.get("mp1_score",0)*100)
    mp2       = int(rec.get("mp2_score",0)*100)
    score_pct = int(rec.get("score",0)*100)

    reason_html = "".join(f'<span class="badge-reason">✓ {r}</span>' for r in reasons)
    stars = "⭐" * round(float(rating)) if rating != "—" else ""

    conic_deg = int(match_pct * 3.6)
    st.markdown(f"""
    <div class="rec-card">
        <div style="display:flex;align-items:flex-start;gap:14px">
            <div style="flex-shrink:0;text-align:center;min-width:32px;padding-top:2px">
                <div class="rank-num">#{rank}</div>
            </div>
            <div style="flex:1;min-width:0">
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:2px;flex-wrap:wrap">
                    <span style="font-size:1rem">{icon}</span>
                    <span class="prod-name">{name}</span>
                </div>
                <div class="prod-brand">{brand} &nbsp;·&nbsp;
                    <span style="color:#3d2a18;font-size:.7rem">{cat}</span>
                </div>
                <div class="prod-meta">{stars} {rating} &nbsp;·&nbsp; {price}</div>
                <div style="margin:6px 0">{reason_html}</div>
                <div style="margin-top:10px">
                    <div class="score-row">
                        <span class="score-lbl">MetaPath MP1</span>
                        <div class="score-trk"><div class="score-fill" style="width:{mp1}%"></div></div>
                        <span style="color:#3d2a18;font-size:.66rem;width:28px;text-align:right">{mp1}%</span>
                    </div>
                    <div class="score-row">
                        <span class="score-lbl">MetaPath MP2</span>
                        <div class="score-trk"><div class="score-fill" style="background:linear-gradient(90deg,#a0522d,#cd7f32);width:{mp2}%"></div></div>
                        <span style="color:#3d2a18;font-size:.66rem;width:28px;text-align:right">{mp2}%</span>
                    </div>
                </div>
            </div>
            <div style="flex-shrink:0;text-align:center">
                <div style="width:54px;height:54px;border-radius:50%;
                    background:conic-gradient(#c2813a {conic_deg}deg,#1e1409 0);
                    display:flex;align-items:center;justify-content:center;position:relative">
                    <div style="width:40px;height:40px;border-radius:50%;background:#0e0a07;
                        position:absolute;display:flex;align-items:center;justify-content:center;
                        color:#f5ede3;font-size:.72rem;font-weight:800">
                        {match_pct}%
                    </div>
                </div>
                <div style="color:#3d2a18;font-size:.62rem;margin-top:3px">match</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR (shown when profile loaded)
# ─────────────────────────────────────────────────────────────────────────────
def _sidebar():
    prof = st.session_state.profile
    if not prof: return
    with st.sidebar:
        st.markdown("""
        <div style="padding:14px 10px 12px;border-bottom:1px solid #2e1f12;margin-bottom:14px">
            <div style="font-size:1.5rem;font-weight:900;letter-spacing:-.03em;
                background:linear-gradient(135deg,#f5ede3,#e8a45a,#c2813a);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent">
                🔮 MetaRec
            </div>
            <div style="color:#3d2a18;font-size:.7rem;margin-top:1px">
                Personality-Aware Recommendations
            </div>
        </div>""", unsafe_allow_html=True)

        arch_emoji = prof.get("archetype_emoji","🎭")
        arch_name  = prof.get("archetype","Explorer")
        arch_color = prof.get("archetype_color","#e8a45a")
        src_label  = "📝 Existing User" if prof.get("source")=="existing" else "✨ New User"

        st.markdown(f"""
        <div style="background:rgba(194,129,58,.08);border:1px solid rgba(194,129,58,.18);
            border-radius:12px;padding:14px 12px;margin-bottom:16px">
            <div style="color:#f5ede3;font-weight:700;font-size:.9rem">
                {arch_emoji} {prof.get('name','User')}
            </div>
            <div style="color:{arch_color};font-size:.74rem;margin-top:3px">{arch_name}</div>
            <div style="margin-top:7px">
                <span style="background:rgba(194,129,58,.12);color:#c2813a;
                    border:1px solid rgba(194,129,58,.2);border-radius:20px;
                    padding:2px 10px;font-size:.7rem">{src_label}</span>
            </div>
        </div>""", unsafe_allow_html=True)

        cur = st.session_state.page
        nav_items = [
            ("recs",    "🏠", "Recommendations"),
            ("profile", "👤", "My Profile"),
            ("network", "🌐", "HIN Graph"),
            ("metrics", "📊", "Evaluation Metrics"),
            ("about",   "ℹ️",  "About"),
        ]
        for pid, icon, label in nav_items:
            active = "active" if cur == pid else ""
            if st.button(f"{icon}  {label}", key=f"nav_{pid}", use_container_width=True):
                go(pid)

        st.markdown("---")
        st.markdown('<div style="color:#5c4a38;font-size:.78rem;margin-bottom:6px">Top-N Results</div>',
                    unsafe_allow_html=True)
        st.session_state.top_n = st.slider(
            "", 4, 20, st.session_state.top_n,
            key="topn_sl", label_visibility="collapsed"
        )

        st.markdown("---")
        if st.button("🔄 Switch User", use_container_width=True, key="sw_user"):
            for k in ["uid","profile","hin","page","ob_step","quiz_ans","sel_cats","ob_name"]:
                if k in st.session_state: del st.session_state[k]
            st.rerun()

        st.markdown("""
        <div style="position:fixed;bottom:16px;left:0;width:220px;
            text-align:center;color:#2e1f12;font-size:.68rem">
            MetaRec v1.0 · B.Tech Mini Project 2025<br>
            IEEE TCSS · Dhelim et al. 2021
        </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: LANDING / UID ENTRY
# ═════════════════════════════════════════════════════════════════════════════
def page_landing():
    col_l, col_r = st.columns([1.1, 0.9], gap="large")

    with col_l:
        st.markdown("""
        <div style="padding-top:60px">
            <div class="hero-title">Discover products<br>made for <em>you.</em></div>
            <div class="hero-sub">
                MetaRec uses your <b style="color:#c2813a">Big Five personality traits</b>
                and topical interests to surface products that genuinely match who you are —
                not just what everyone else is clicking.
            </div>
            <div style="display:flex;gap:24px;margin-top:36px;flex-wrap:wrap">
                <div style="text-align:center">
                    <div style="font-size:1.8rem;font-weight:800;color:#e8a45a">100k+</div>
                    <div style="color:#3d2a18;font-size:.78rem">Users</div>
                </div>
                <div style="text-align:center">
                    <div style="font-size:1.8rem;font-weight:800;color:#e8a45a">30k+</div>
                    <div style="color:#3d2a18;font-size:.78rem">Products</div>
                </div>
                <div style="text-align:center">
                    <div style="font-size:1.8rem;font-weight:800;color:#e8a45a">15</div>
                    <div style="color:#3d2a18;font-size:.78rem">Categories</div>
                </div>
                <div style="text-align:center">
                    <div style="font-size:1.8rem;font-weight:800;color:#e8a45a">5</div>
                    <div style="color:#3d2a18;font-size:.78rem">Personality Types</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown('<div style="padding-top:40px">', unsafe_allow_html=True)
        st.markdown("""
        <div class="uid-box">
            <div style="font-size:1.15rem;font-weight:800;color:#f5ede3;margin-bottom:4px">
                🔮 Enter Your User ID
            </div>
            <div style="color:#5c4a38;font-size:.82rem;margin-bottom:20px">
                Existing users get instant recommendations.<br>
                New users take a quick personality quiz.
            </div>
        </div>""", unsafe_allow_html=True)

        uid_input = st.text_input(
            "User ID", placeholder="e.g. U000042  or  any name for new user",
            label_visibility="collapsed", key="uid_input_field"
        )

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("✦  Get Recommendations", use_container_width=True, key="btn_go"):
                uid = uid_input.strip()
                if not uid:
                    st.error("Please enter a User ID.")
                else:
                    st.session_state.uid = uid
                    # Check if existing
                    if is_existing_user(uid, users_df):
                        with st.spinner("Building your profile…"):
                            prof = build_existing_profile(uid, users_df, interactions_df)
                        st.session_state.profile = prof
                        # Build HIN
                        with st.spinner("Building recommendation graph…"):
                            hin = build_user_hin(prof, products_df, interactions_df, users_df)
                        st.session_state.hin = hin
                        go("recs")
                    else:
                        # New user → onboarding
                        st.session_state.ob_name = uid
                        st.session_state.ob_step  = 0
                        go("onboarding")

        with col_b:
            # Demo with a random existing UID
            if st.button("🎲  Try Demo User", use_container_width=True, key="btn_demo"):
                demo_uid = random.choice(users_df["uid"].tolist())
                with st.spinner("Loading demo…"):
                    prof = build_existing_profile(demo_uid, users_df, interactions_df)
                    hin  = build_user_hin(prof, products_df, interactions_df, users_df)
                st.session_state.uid     = demo_uid
                st.session_state.profile = prof
                st.session_state.hin     = hin
                go("recs")

        st.markdown("""
        <div style="margin-top:16px;padding:12px 14px;background:#150f08;
            border:1px solid #2e1f12;border-radius:10px;font-size:.76rem;color:#3d2a18">
            💡 Sample UIDs from the dataset: <b style="color:#8b5e3c">U000001</b> &nbsp;·&nbsp;
            <b style="color:#8b5e3c">U012345</b> &nbsp;·&nbsp;
            <b style="color:#8b5e3c">U099999</b><br>
            Or type any new name to create a fresh profile.
        </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── How it works ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### How MetaRec Works")
    c1,c2,c3,c4 = st.columns(4)
    steps = [
        ("🧠","Personality Profiling","Big-Five (OCEAN) traits extracted from reviews or quiz"),
        ("🎯","Interest Mining","TF-IDF maps your text to 15 topic categories"),
        ("🌐","HIN Graph","Users, Topics & Products linked in a heterogeneous network"),
        ("✦","Metapath Discovery","U→T→P and U→U'→T→P paths score every product"),
    ]
    for col, (icon,title,desc) in zip([c1,c2,c3,c4], steps):
        with col:
            st.markdown(f"""
            <div class="warm-card" style="text-align:center;padding:22px 16px">
                <div style="font-size:2rem;margin-bottom:10px">{icon}</div>
                <div style="color:#f5ede3;font-weight:700;font-size:.88rem;margin-bottom:6px">{title}</div>
                <div style="color:#4a3020;font-size:.76rem;line-height:1.6">{desc}</div>
            </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: ONBOARDING (new users)
# ═════════════════════════════════════════════════════════════════════════════
LIKERT_LABELS = ["Strongly Disagree","Disagree","Neutral","Agree","Strongly Agree"]

def page_onboarding():
    step = st.session_state.ob_step
    name = st.session_state.ob_name or "there"
    st.markdown(_steps_ui(step), unsafe_allow_html=True)

    # ── Step 0: Welcome ───────────────────────────────────────────────────────
    if step == 0:
        st.markdown(f"""
        <div style="text-align:center;padding:30px 0 24px">
            <div style="font-size:3.5rem;margin-bottom:14px">🔮</div>
            <div style="font-size:1.8rem;font-weight:900;color:#f5ede3;margin-bottom:10px">
                Welcome, {name.split()[0]}!
            </div>
            <div style="color:#5c4a38;max-width:480px;margin:0 auto;font-size:.92rem;line-height:1.8">
                We'll ask you <b style="color:#c2813a">10 quick personality questions</b>
                and let you pick your interests. Then MetaRec's AI will build a personalised
                recommendation profile just for you.
            </div>
        </div>""", unsafe_allow_html=True)

        c1,c2,c3 = st.columns(3)
        for col, icon, t, d in zip(
            [c1,c2,c3],
            ["🧠","🎯","✦"],
            ["Personality Analysis","Interest Mapping","Smart Recommendations"],
            ["10 BFI-10 questions","15 product categories","Powered by HIN + Metapaths"]
        ):
            with col:
                st.markdown(f"""
                <div class="warm-card" style="text-align:center;padding:20px 14px">
                    <div style="font-size:1.8rem;margin-bottom:8px">{icon}</div>
                    <div style="color:#f5ede3;font-weight:700;font-size:.84rem;margin-bottom:4px">{t}</div>
                    <div style="color:#3d2a18;font-size:.74rem">{d}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("")
        c_,cc,c__ = st.columns([1,1,1])
        with cc:
            if st.button("Begin Personality Quiz →", use_container_width=True, key="ob_start"):
                st.session_state.ob_step = 1
                st.rerun()

    # ── Step 1: BFI-10 Quiz ───────────────────────────────────────────────────
    elif step == 1:
        st.markdown("""
        <div style="text-align:center;margin-bottom:28px">
            <div style="font-size:1.3rem;font-weight:800;color:#f5ede3">🧠 Personality Quiz</div>
            <div style="color:#5c4a38;font-size:.84rem;margin-top:5px">
                Rate how much each statement describes you. Respond honestly — no right or wrong answers.
            </div>
        </div>""", unsafe_allow_html=True)

        answers = st.session_state.quiz_ans.copy()
        trait_colors = {"Openness":"#e8a45a","Conscientiousness":"#8fbc8f",
                        "Extraversion":"#f4a460","Agreeableness":"#dda0dd","Neuroticism":"#87ceeb"}

        # Group questions by trait
        from collections import OrderedDict
        trait_qs = OrderedDict()
        for i,(q,trait,_) in enumerate(PERSONALITY_QUIZ):
            trait_qs.setdefault(trait,[]).append((i,q))

        for trait, qlist in trait_qs.items():
            col_c = trait_colors.get(trait,"#e8a45a")
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:8px;margin:18px 0 10px">
                <div style="width:10px;height:10px;border-radius:50%;background:{col_c}"></div>
                <span style="color:{col_c};font-weight:700;font-size:.82rem">{trait}</span>
            </div>""", unsafe_allow_html=True)
            for qi, qtxt in qlist:
                st.markdown(f"""
                <div style="background:#150f08;border:1px solid #2e1f12;border-radius:12px;
                    padding:14px 18px;margin-bottom:8px;color:#c9a880;font-size:.88rem">
                    {qtxt}
                </div>""", unsafe_allow_html=True)
                val = st.select_slider(
                    " ", options=[1,2,3,4,5],
                    value=answers.get(qi,3),
                    format_func=lambda x: LIKERT_LABELS[x-1],
                    key=f"q_{qi}"
                )
                answers[qi] = val

        st.session_state.quiz_ans = answers
        st.markdown("")
        c1,c2 = st.columns(2)
        with c1:
            if st.button("← Back", key="ob1_bk", use_container_width=True):
                st.session_state.ob_step = 0; st.rerun()
        with c2:
            if st.button("Next: Choose Interests →", key="ob1_nx", use_container_width=True):
                st.session_state.ob_step = 2; st.rerun()

    # ── Step 2: Category Selection ────────────────────────────────────────────
    elif step == 2:
        st.markdown("""
        <div style="text-align:center;margin-bottom:24px">
            <div style="font-size:1.3rem;font-weight:800;color:#f5ede3">🛍️ What are you into?</div>
            <div style="color:#5c4a38;font-size:.84rem;margin-top:5px">
                Select at least 3 categories that interest you.
            </div>
        </div>""", unsafe_allow_html=True)

        sel = st.session_state.sel_cats.copy()
        cols = st.columns(4)
        for idx, cat in enumerate(CATEGORIES):
            icon   = CATEGORY_ICONS.get(cat,"📦")
            is_sel = cat in sel
            check  = "✓ " if is_sel else ""
            bg     = "rgba(194,129,58,0.12)" if is_sel else "transparent"
            border = "#c2813a" if is_sel else "#2e1f12"
            col_c  = "#e8a45a" if is_sel else "#5c4a38"
            with cols[idx % 4]:
                if st.button(f"{icon} {check}{cat}", key=f"cat_{idx}",
                             use_container_width=True):
                    if cat in sel: sel.remove(cat)
                    else: sel.append(cat)
                    st.session_state.sel_cats = sel
                    st.rerun()

        st.markdown(f"""
        <div style="text-align:center;margin:18px 0;color:#5c4a38;font-size:.8rem">
            {len(sel)} selected — need at least 3
        </div>""", unsafe_allow_html=True)

        c1,c2 = st.columns(2)
        with c1:
            if st.button("← Back", key="ob2_bk", use_container_width=True):
                st.session_state.ob_step = 1; st.rerun()
        with c2:
            disabled = len(sel) < 3
            if st.button("Analyse My Personality →", key="ob2_nx",
                         use_container_width=True, disabled=disabled):
                st.session_state.ob_step = 3; st.rerun()

    # ── Step 3: Results ───────────────────────────────────────────────────────
    elif step == 3:
        answers = st.session_state.quiz_ans
        cats    = st.session_state.sel_cats
        pers    = score_bfi10(answers)
        dom     = dominant_trait(pers)
        arch    = TRAIT_ARCHETYPES[dom]
        arch_name, arch_emoji, arch_color = arch

        st.markdown(f"""
        <div class="arch-card">
            <div style="font-size:3rem">{arch_emoji}</div>
            <div class="arch-title">{arch_name}</div>
            <div style="color:#9d8872;font-size:.86rem;margin-top:10px;
                max-width:420px;margin-left:auto;margin-right:auto;line-height:1.7">
                Your dominant Big-Five trait is <b style="color:{arch_color}">{dom}</b>.
                Your recommendations are personalised around this personality profile.
            </div>
        </div>""", unsafe_allow_html=True)

        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("#### Your Big Five Scores")
            for i,(trait,score) in enumerate(pers.items()):
                pct = int(score*100)
                st.markdown(f"""
                <div style="margin-bottom:12px">
                    <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                        <span style="color:#c9a880;font-size:.84rem;font-weight:600">{trait}</span>
                        <span style="color:#3d2a18;font-size:.78rem">{pct}%</span>
                    </div>
                    {_bar(TRAIT_BARS[i], pct)}
                </div>""", unsafe_allow_html=True)

        with c_right:
            st.markdown("#### Selected Interests")
            chips = "".join(
                f'<span class="badge-topic">{CATEGORY_ICONS.get(c,"📦")} {c}</span>'
                for c in cats
            )
            st.markdown(chips, unsafe_allow_html=True)
            st.markdown("")
            st.markdown("""
            <div style="background:#150f08;border:1px solid #2e1f12;border-radius:12px;
                padding:14px 16px;margin-top:12px;font-size:.8rem;color:#5c4a38;line-height:1.7">
                <b style="color:#c2813a">How this works:</b><br>
                Your OCEAN scores are used to compute user-user similarity in the HIN.
                Metapaths connect you through shared topics to products you haven't seen yet.
            </div>""", unsafe_allow_html=True)

        st.markdown("")
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            if st.button("← Redo Quiz", key="ob3_bk", use_container_width=True):
                st.session_state.ob_step = 1; st.rerun()
        with c3:
            if st.button("🚀 See My Recommendations!", key="ob3_done",
                         use_container_width=True):
                uid  = f"NEW_{st.session_state.ob_name.replace(' ','_')[:20]}"
                prof = build_new_user_profile(uid, st.session_state.ob_name, answers, cats)
                save_new_user(prof)
                with st.spinner("Building your recommendation graph…"):
                    hin = build_user_hin(prof, products_df, interactions_df, users_df)
                st.session_state.uid     = uid
                st.session_state.profile = prof
                st.session_state.hin     = hin
                st.session_state.ob_step = 0
                go("recs")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: RECOMMENDATIONS
# ═════════════════════════════════════════════════════════════════════════════
def page_recs():
    prof  = st.session_state.profile
    hin   = st.session_state.hin
    top_n = st.session_state.top_n

    name      = prof.get("name","")
    interests = prof.get("topic_interests",[])
    pers      = prof.get("personality",{})
    dom       = prof.get("dominant_trait","")
    arch_name = prof.get("archetype","")
    arch_emoji= prof.get("archetype_emoji","🎭")
    source    = prof.get("source","")

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="margin-bottom:22px">
        <div style="font-size:1.7rem;font-weight:900;color:#f5ede3;letter-spacing:-.025em">
            {arch_emoji} Hey, {name.split()[0]}!
        </div>
        <div style="color:#5c4a38;font-size:.88rem;margin-top:4px">
            {"Personalised picks from your interaction history" if source=="existing"
             else "Personalised picks based on your personality quiz"} ·
            <b style="color:#c2813a">HIN Metapath Analysis</b>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Stats row ─────────────────────────────────────────────────────────────
    top_interest = interests[0][0] if interests else "—"
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(_stat(top_n, "Recommendations"), unsafe_allow_html=True)
    with c2: st.markdown(_stat(dom[:12] if dom else "—", "Dominant Trait"), unsafe_allow_html=True)
    with c3: st.markdown(_stat(f"{CATEGORY_ICONS.get(top_interest,'🎯')} {top_interest[:10]}", "Top Interest"), unsafe_allow_html=True)
    with c4: st.markdown(_stat(prof.get("n_reviews",0), "Reviews Analysed"), unsafe_allow_html=True)

    st.markdown("---")

    # ── Interest chips ────────────────────────────────────────────────────────
    visible_interests = [(t,s) for t,s in interests[:8] if s > 0.01]
    if visible_interests:
        st.markdown('<div style="color:#5c4a38;font-size:.78rem;margin-bottom:6px">Your mined interests</div>', unsafe_allow_html=True)
        chips = "".join(
            f'<span class="badge-topic">{CATEGORY_ICONS.get(t,"📦")} {t}'
            f'<span style="color:#3d2a18"> · {int(s*100)}%</span></span>'
            for t,s in visible_interests
        )
        st.markdown(chips, unsafe_allow_html=True)
        st.markdown("")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["  ✦  For You  ", "  🔥  Trending  "])

    with tab1:
        if hin:
            recs = recommend(hin, prof["uid"], top_n=top_n)
        else:
            recs = []

        if not recs:
            recs = popular_products(PRODUCTS_LIST, pers, top_n)
            st.info("💡 Showing personality-matched popular products while your graph is warming up.")

        st.markdown(f"#### Top {len(recs)} Picks for You")
        cl, cr = st.columns(2)
        for i, rec in enumerate(recs):
            with cl if i%2==0 else cr:
                _render_rec_card(rec)

    with tab2:
        pop = popular_products(PRODUCTS_LIST, None, top_n)
        st.markdown("#### 🔥 Trending Across MetaRec")
        cl, cr = st.columns(2)
        for i, rec in enumerate(pop):
            with cl if i%2==0 else cr:
                _render_rec_card(rec)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: PROFILE
# ═════════════════════════════════════════════════════════════════════════════
def page_profile():
    prof = st.session_state.profile
    hin  = st.session_state.hin
    if not prof: st.warning("No profile loaded."); return

    name      = prof.get("name","")
    pers      = prof.get("personality",{})
    interests = prof.get("topic_interests",[])
    pids      = prof.get("interacted_pids",[])
    arch_name = prof.get("archetype","")
    arch_emoji= prof.get("archetype_emoji","🎭")
    dom       = prof.get("dominant_trait","")
    arch_color= prof.get("archetype_color","#e8a45a")

    # ── Avatar + header ───────────────────────────────────────────────────────
    c_av, c_info = st.columns([0.12, 0.88])
    with c_av:
        st.markdown(f"""
        <div style="width:64px;height:64px;border-radius:50%;margin-top:4px;
            background:linear-gradient(135deg,#c2813a,#e8a45a);
            display:flex;align-items:center;justify-content:center;
            font-size:1.6rem;font-weight:800;color:#0e0a07">
            {name[0].upper() if name else "U"}
        </div>""", unsafe_allow_html=True)
    with c_info:
        st.markdown(f"""
        <div>
            <div style="font-size:1.4rem;font-weight:800;color:#f5ede3">{name}</div>
            <div style="color:{arch_color};font-size:.82rem;margin-top:2px">
                {arch_emoji} {arch_name}
            </div>
            <div style="margin-top:6px">
                <span class="badge-amber">UID: {prof.get('uid','—')}</span>
                &nbsp;
                <span class="badge-topic">
                    {"📝 Existing User" if prof.get('source')=='existing' else "✨ New User"}
                </span>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["  🧠  Personality  ","  🎯  Interests  ","  🛒  History  "])

    # ── Personality ───────────────────────────────────────────────────────────
    with tab1:
        st.markdown(f"""
        <div class="arch-card">
            <div style="font-size:2.2rem">{arch_emoji}</div>
            <div class="arch-title">{arch_name}</div>
            <div style="color:#9d8872;font-size:.84rem;margin-top:8px;
                max-width:400px;margin-left:auto;margin-right:auto;line-height:1.7">
                Dominant trait: <b style="color:{arch_color}">{dom}</b>.
                Your recommendations are tuned to this personality profile using
                the Big-Five affinity model from Dhelim et al. (2021).
            </div>
        </div>""", unsafe_allow_html=True)

        for i,(trait,score) in enumerate(pers.items()):
            pct = int(score*100)
            st.markdown(f"""
            <div style="margin-bottom:14px">
                <div style="display:flex;justify-content:space-between;margin-bottom:5px">
                    <span style="color:#c9a880;font-size:.86rem;font-weight:600">{trait}</span>
                    <span style="color:#3d2a18;font-size:.8rem">{pct}/100</span>
                </div>
                {_bar(TRAIT_BARS[i], pct)}
            </div>""", unsafe_allow_html=True)

    # ── Interests ─────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("#### Topic Interest Scores (TF-IDF Mined)")
        for cat, score in sorted(interests, key=lambda x: x[1], reverse=True):
            pct  = int(score * 100)
            icon = CATEGORY_ICONS.get(cat,"📦")
            st.markdown(f"""
            <div style="margin-bottom:10px">
                <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                    <span style="color:#c9a880;font-size:.82rem">{icon} {cat}</span>
                    <span style="color:#3d2a18;font-size:.76rem">{pct}%</span>
                </div>
                {_bar("bar-sienna", pct)}
            </div>""", unsafe_allow_html=True)

    # ── History ───────────────────────────────────────────────────────────────
    with tab3:
        if not pids:
            st.info("No interaction history — recommendations are based on your personality profile.")
        else:
            st.markdown(f"#### {len(pids)} Past Interactions")
            prod_idx = {str(r["pid"]): r for r in PRODUCTS_LIST}
            cols = st.columns(2)
            for i, pid in enumerate(pids[:24]):
                p = prod_idx.get(str(pid), {})
                if not p: continue
                cat   = str(p.get("category",""))
                icon  = CATEGORY_ICONS.get(cat,"📦")
                pname = str(p.get("name",""))[:40]
                try: price = f"₹{float(p.get('price',0)):,.0f}"
                except: price = "—"
                with cols[i%2]:
                    st.markdown(f"""
                    <div class="warm-card" style="padding:12px 14px 12px 17px">
                        <div style="font-size:.72rem;color:#c2813a;margin-bottom:2px">{icon} {cat}</div>
                        <div style="color:#f5ede3;font-weight:600;font-size:.84rem">{pname}</div>
                        <div style="color:#3d2a18;font-size:.72rem;margin-top:3px">⭐ {p.get('rating','—')} · {price}</div>
                    </div>""", unsafe_allow_html=True)

    # ── Similar users ─────────────────────────────────────────────────────────
    if hin:
        sims = hin.similar_users(prof["uid"], k=5)
        if sims:
            st.markdown("---")
            st.markdown("#### 👥 Your Nearest Neighbours in the HIN")
            for su, sw in sims:
                u     = hin.get_user(su)
                uname = u.get("name","User")
                uarch = u.get("archetype","—")
                pct   = int(sw*100)
                st.markdown(f"""
                <div class="warm-card" style="display:flex;align-items:center;gap:14px;padding:12px 16px">
                    <div style="width:36px;height:36px;border-radius:50%;flex-shrink:0;
                        background:linear-gradient(135deg,#c2813a,#e8a45a);
                        display:flex;align-items:center;justify-content:center;
                        color:#0e0a07;font-weight:800;font-size:.9rem">
                        {uname[0].upper()}
                    </div>
                    <div style="flex:1">
                        <div style="color:#f5ede3;font-weight:600;font-size:.86rem">{uname}</div>
                        <div style="color:#5c4a38;font-size:.74rem">{uarch}</div>
                    </div>
                    <span class="badge-amber">Sim: {pct}%</span>
                </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: HIN GRAPH
# ═════════════════════════════════════════════════════════════════════════════
def page_network():
    prof = st.session_state.profile
    hin  = st.session_state.hin
    if not prof or not hin:
        st.warning("Please load a profile first.")
        return

    st.markdown("""
    <div style="margin-bottom:20px">
        <div style="font-size:1.4rem;font-weight:800;color:#f5ede3">🌐 Heterogeneous Information Network</div>
        <div style="color:#5c4a38;font-size:.84rem;margin-top:4px">
            HIN with node types U (Users), T (Topics), P (Products) and
            metapaths MP1: U→T→P and MP2: U→U'→T→P
        </div>
    </div>""", unsafe_allow_html=True)

    stats = hin.stats()
    c1,c2,c3,c4 = st.columns(4)
    for col, lbl, val in zip([c1,c2,c3,c4],
        ["Users in Graph","Products","Topics","U-T Edges"],
        [stats["n_users"],stats["n_products"],stats["n_topics"],stats["ut_edges"]]):
        with col:
            st.markdown(_stat(f"{val:,}", lbl), unsafe_allow_html=True)

    st.markdown("---")
    uid    = prof["uid"]
    traces = get_metapath_traces(hin, uid, top_k=14)

    col_l, col_r = st.columns([1, 1])

    with col_l:
        mp1_t = [t for t in traces if t["type"]=="MP1"]
        mp2_t = [t for t in traces if t["type"]=="MP2"]

        if mp1_t:
            st.markdown('<span class="badge-topic">MP1: You → Topic → Product</span>', unsafe_allow_html=True)
            st.markdown("")
            for t in mp1_t[:7]:
                icon  = CATEGORY_ICONS.get(t["topic"],"📦")
                st.markdown(f"""
                <div class="path-chip">
                    <div style="display:flex;align-items:center;gap:8px">
                        <span style="font-size:.9rem">{icon}</span>
                        <div>
                            <div style="color:#9d8872;font-size:.72rem">{t['path']}</div>
                            <div style="display:flex;align-items:center;gap:8px;margin-top:4px">
                                <span style="color:#f5ede3;font-size:.8rem;font-weight:600">
                                    {t['product_name']}
                                </span>
                                <span class="badge-amber" style="font-size:.64rem;padding:1px 7px">
                                    w={t['weight']:.3f}
                                </span>
                            </div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

        if mp2_t:
            st.markdown("")
            st.markdown('<span class="badge-reason">MP2: You → Similar User → Topic → Product</span>',
                        unsafe_allow_html=True)
            st.markdown("")
            for t in mp2_t[:5]:
                st.markdown(f"""
                <div style="background:#150f08;border:1px solid #2e1f12;
                    border-radius:10px;padding:10px 14px;margin-bottom:6px;
                    font-size:.76rem;color:#5c4a38;border-left:3px solid #8b5e3c">
                    {t['path']}<br>
                    <span style="color:#3d2a18">score: {t['weight']:.4f}</span>
                </div>""", unsafe_allow_html=True)

    with col_r:
        try:
            import plotly.graph_objects as go
            import networkx as nx

            G = nx.DiGraph()
            nc, ns, nl = {}, {}, {}

            G.add_node(uid)
            nc[uid] = "#c2813a"; ns[uid] = 24; nl[uid] = prof.get("name","You")[:14]

            for tr in traces[:12]:
                tid  = f"T:{tr['topic']}"
                pid  = f"P:{tr['pid']}"
                pname= tr["product_name"][:18]

                if tid not in G:
                    G.add_node(tid); nc[tid]="#e8a45a"; ns[tid]=16; nl[tid]=f"{CATEGORY_ICONS.get(tr['topic'],'')} {tr['topic'][:10]}"
                if pid not in G:
                    G.add_node(pid); nc[pid]="#8fbc8f"; ns[pid]=10; nl[pid]=pname

                G.add_edge(uid, tid); G.add_edge(tid, pid)
                if tr["type"]=="MP2":
                    for su,_ in hin.similar_users(uid, k=2):
                        suid = f"U:{su}"; u=hin.get_user(su)
                        if suid not in G:
                            G.add_node(suid); nc[suid]="#cd9b6a"; ns[suid]=13
                            nl[suid]=u.get("name","?")[:10]
                        G.add_edge(uid,suid); G.add_edge(suid,tid)

            pos = nx.spring_layout(G, seed=42, k=2.5)
            ex,ey=[],[]
            for u_n,v_n in G.edges():
                x0,y0=pos[u_n]; x1,y1=pos[v_n]
                ex+=[x0,x1,None]; ey+=[y0,y1,None]

            fig = go.Figure(
                data=[
                    go.Scatter(x=ex,y=ey,mode="lines",
                               line=dict(width=1,color="rgba(194,129,58,0.25)"),
                               hoverinfo="none"),
                    go.Scatter(
                        x=[pos[n][0] for n in G.nodes()],
                        y=[pos[n][1] for n in G.nodes()],
                        mode="markers+text",
                        hoverinfo="text",
                        text=[nl.get(n,n[:10]) for n in G.nodes()],
                        textposition="top center",
                        textfont=dict(color="#9d8872",size=9),
                        marker=dict(
                            color=[nc.get(n,"#c2813a") for n in G.nodes()],
                            size=[ns.get(n,10) for n in G.nodes()],
                            line=dict(width=1.5,color="#0e0a07")
                        )
                    )
                ],
                layout=go.Layout(
                    paper_bgcolor="#0e0a07", plot_bgcolor="#0e0a07",
                    showlegend=False, height=440,
                    xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                    yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                    margin=dict(l=10,r=10,t=10,b=30),
                    annotations=[dict(
                        x=0.5,y=-0.06,xref="paper",yref="paper",showarrow=False,
                        text="🟠 You  &nbsp;🟡 Topic  &nbsp;🟢 Product  &nbsp;🟤 Similar User",
                        font=dict(color="#3d2a18",size=10)
                    )]
                )
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.info("Install `plotly` and `networkx` for the interactive graph visualisation.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: EVALUATION METRICS
# ═════════════════════════════════════════════════════════════════════════════
def page_metrics():
    prof = st.session_state.profile
    hin  = st.session_state.hin
    if not prof or not hin:
        st.warning("Please load a profile first.")
        return

    st.markdown("""
    <div style="margin-bottom:20px">
        <div style="font-size:1.4rem;font-weight:800;color:#f5ede3">📊 Evaluation Metrics</div>
        <div style="color:#5c4a38;font-size:.84rem;margin-top:4px">
            Leave-one-out evaluation following paper Section V —
            Precision@N, Recall@N, and F1 score.
        </div>
    </div>""", unsafe_allow_html=True)

    top_n = st.session_state.top_n

    # Run evaluation on neighbourhood profiles
    nb_profiles = [hin.get_user(uid) for uid in hin.all_user_ids()
                   if uid != prof["uid"] and hin.get_user(uid).get("interacted_pids")]

    if len(nb_profiles) >= 3:
        test_sample = nb_profiles[:min(50, len(nb_profiles))]
        with st.spinner("Running leave-one-out evaluation…"):
            metrics = evaluate(hin, test_sample, top_n=top_n)

        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric("Precision@N",  f"{metrics['precision']:.4f}")
        with c2: st.metric("Recall@N",     f"{metrics['recall']:.4f}")
        with c3: st.metric("F1 Score",     f"{metrics['f1']:.4f}")
        with c4: st.metric("Users Tested", metrics["n_eval"])
    else:
        st.info("Need more neighbourhood data for evaluation. Try an existing user UID.")

    st.markdown("---")

    # User similarity breakdown
    st.markdown("#### User Similarity Decomposition")
    st.markdown("""
    <div style="background:#150f08;border:1px solid #2e1f12;border-radius:12px;
        padding:16px 18px;font-size:.82rem;color:#5c4a38;line-height:2">
        <b style="color:#c2813a">Paper eq. (4):</b>
        Sim(u<sub>i</sub>, u<sub>j</sub>) = α·Sim<sub>P</sub> + β·Sim<sub>T</sub> + γ·Sim<sub>I</sub><br>
        where α=0.40 (personality cosine), β=0.35 (topic interest cosine), γ=0.25 (interaction Jaccard)<br>
        <b style="color:#c2813a">Paper eq. (7):</b>
        Score(p) = w₁·MP1(p) + w₂·MP2(p) &nbsp; (w₁=0.60, w₂=0.40)
    </div>""", unsafe_allow_html=True)
    st.markdown("")

    # Per-neighbour similarity table
    sims = hin.similar_users(prof["uid"], k=10)
    if sims:
        st.markdown("#### Your Top Neighbours' Similarity Scores")
        rows = []
        for su, sw in sims:
            u = hin.get_user(su)
            if not u: continue
            from core.engine import personality_similarity, topic_similarity, _jaccard
            sp = personality_similarity(prof.get("personality",{}), u.get("personality",{}))
            st_val = topic_similarity(dict(prof.get("topic_interests",[])), dict(u.get("topic_interests",[])))
            si = _jaccard(set(prof.get("interacted_pids",[])), set(u.get("interacted_pids",[])))
            rows.append({"User": u.get("name","?"), "SimP": sp, "SimT": st_val, "SimI": si, "Combined": sw})

        import pandas as pd
        df = pd.DataFrame(rows).sort_values("Combined", ascending=False)
        st.dataframe(
            df.style.background_gradient(cmap="YlOrBr", subset=["Combined"]),
            use_container_width=True
        )

    # HIN stats
    st.markdown("---")
    st.markdown("#### HIN Graph Statistics")
    stats = hin.stats()
    c1,c2,c3 = st.columns(3)
    with c1: st.metric("U-U Edges (sim)", stats["uu_edges"])
    with c2: st.metric("U-T Edges (interest)", stats["ut_edges"])
    with c3: st.metric("U-P Edges (interaction)", stats["up_edges"])


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ═════════════════════════════════════════════════════════════════════════════
def page_about():
    st.markdown("""
    <div style="max-width:820px">
        <div style="font-size:1.6rem;font-weight:900;color:#f5ede3;margin-bottom:6px">
            🔮 About MetaRec
        </div>
        <div style="color:#5c4a38;font-size:.88rem">
            B.Tech III Year Industrial Mini Project · 2025–2026
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    c1, c2 = st.columns([1.2,0.8])
    with c1:
        st.markdown("""
        ### 📖 Reference Paper
        > **Dhelim, S., Ning, H., Aung, N., Huang, R., & Ma, J. (2021).**
        > *Personality-Aware Product Recommendation System Based on User Interests
        > Mining and Metapath Discovery.*
        > **IEEE Transactions on Computational Social Systems**, 8(1), 86–98.
        > DOI: 10.1109/TCSS.2020.3037040

        ---

        ### 🏗️ System Architecture (Paper Fig. 1)
        """)
        st.markdown("""
        <div style="background:#150f08;border:1px solid #2e1f12;border-radius:14px;
            padding:20px;font-family:monospace;font-size:.78rem;color:#9d8872;line-height:1.9">
        User Data (reviews / quiz)<br>
        &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
        <span style="color:#c2813a">① Personality Inference</span>  (BFI-10 / LIWC lexicon)<br>
        &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
        <span style="color:#e8a45a">② Interest Mining</span>  (TF-IDF cosine → 15 topics)<br>
        &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
        <span style="color:#cd9b6a">③ HIN Construction</span>  (U–U, U–T, T–P, U–P edges)<br>
        &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
        <span style="color:#c2813a">④ Metapath Discovery</span>  (MP1: U→T→P, MP2: U→U'→T→P)<br>
        &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
        <span style="color:#e8a45a">⑤ Top-N Ranking</span>  (normalised metapath score)<br>
        &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
        Product Recommendations
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        ---
        ### ⚙️ Key Equations

        **User Similarity (eq. 4):**
        """)
        st.latex(r"\text{Sim}(u_i, u_j) = \alpha \cdot \text{Sim}_P + \beta \cdot \text{Sim}_T + \gamma \cdot \text{Sim}_I")
        st.markdown("α=0.40 · β=0.35 · γ=0.25")

        st.markdown("**Interest Mining (eq. 3):**")
        st.latex(r"\text{Interest}(u,t) = \frac{1}{|\mathcal{F}|}\sum_{f \in \mathcal{F}} P(u,f) \cdot A(f,t)")

        st.markdown("**Metapath Score (eq. 7):**")
        st.latex(r"\text{Score}(p) = w_1 \cdot \text{MP1}(p) + w_2 \cdot \text{MP2}(p)")

    with c2:
        st.markdown("""
        ### 👥 Project Team
        """)
        team = [
            ("23B81A67E3","A. Keerthi"),
            ("23B81A67E7","D. Lekhana"),
            ("23B81A67F8","G. Poojitha Sindhu"),
        ]
        for roll, tname in team:
            st.markdown(f"""
            <div class="warm-card" style="padding:14px 16px;margin-bottom:10px">
                <div style="color:#f5ede3;font-weight:700;font-size:.9rem">{tname}</div>
                <div style="color:#5c4a38;font-size:.74rem;margin-top:2px">{roll}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("""
        ---
        ### 📊 Dataset Info
        """)
        ds_info = [
            ("users.csv",        "100,000 users"),
            ("products.csv",     "30,000 products"),
            ("interactions.csv", "200,000 reviews"),
        ]
        for fname, desc in ds_info:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;
                padding:10px 14px;background:#150f08;border:1px solid #2e1f12;
                border-radius:8px;margin-bottom:6px">
                <span style="color:#c9a880;font-size:.8rem;font-weight:600">{fname}</span>
                <span class="badge-topic">{desc}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("""
        ---
        ### 🛠️ Stack
        """)
        tools = ["Python 3.10+","Streamlit","scikit-learn","NetworkX","Plotly","Pandas","NumPy"]
        st.markdown(" ".join(f'<span class="badge-reason">{t}</span>' for t in tools),
                    unsafe_allow_html=True)

        st.markdown("""
        ---
        ### 🔗 Dataset Sources
        """)
        sources = [
            ("Amazon Reviews (McAuley UCSD)",  "https://jmcauley.ucsd.edu/data/amazon"),
            ("Kaggle Amazon Product Reviews",   "https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews"),
            ("Amazon Metadata (McAuley)",       "https://jmcauley.ucsd.edu/data/amazon/links.html"),
        ]
        for label, url in sources:
            st.markdown(f'<div style="margin-bottom:5px"><a href="{url}" style="color:#c2813a;font-size:.78rem">↗ {label}</a></div>',
                        unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# ROUTER
# ═════════════════════════════════════════════════════════════════════════════
def _has_profile():
    return st.session_state.profile is not None

page = st.session_state.page

if page == "landing":
    page_landing()
elif page == "onboarding":
    page_onboarding()
else:
    if not _has_profile():
        go("landing")
    else:
        _sidebar()
        if page == "recs":    page_recs()
        elif page == "profile":  page_profile()
        elif page == "network":  page_network()
        elif page == "metrics":  page_metrics()
        elif page == "about":    page_about()
        else: go("recs")
