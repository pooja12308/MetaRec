# 🔮 MetaRec — Personality-Aware Product Recommendation System

> **B.Tech III Year Industrial Mini Project · 2025–2026**
> **Class:** III Year CSD · Section C · Batch COEPB63

---

## 📖 Reference

> Dhelim, S., Ning, H., Aung, N., Huang, R., & Ma, J. (2021).
> *Personality-Aware Product Recommendation System Based on User Interests Mining and Metapath Discovery.*
> **IEEE Transactions on Computational Social Systems**, 8(1), 86–98.
> DOI: [10.1109/TCSS.2020.3037040](https://ieeexplore.ieee.org/document/9269396)

---

## 👥 Team

| Roll No | Name |
|---------|------|
| 23B81A67E3 | A. Keerthi |
| 23B81A67E7 | D. Lekhana |
| 23B81A67F8 | G. Poojitha Sindhu |

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate datasets (one-time, takes ~2 min)
python generate_data.py

# 3. Run the app
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## 📁 Project Structure

```
metarec/
├── app.py                  ← Streamlit app (all 6 pages)
├── generate_data.py        ← Dataset generator
├── requirements.txt
├── README.md
├── data/
│   ├── users.csv           ← 100,000 users (uid, name, personality_type, top_topics)
│   ├── products.csv        ← 30,000 products (pid, name, category, desc, price, rating)
│   └── interactions.csv    ← 200,000 user-product interactions (uid, pid, rating, review)
├── core/
│   ├── engine.py           ← Full Meta-Interest algorithm (paper implementation)
│   └── data_loader.py      ← Data loading, profile building, HIN construction
└── users_db/               ← JSON profiles for new users (auto-created)
```

---

## 🎯 App Pages

| Page | Description |
|------|-------------|
| 🏠 Landing | UID entry — routes to recs (existing) or quiz (new) |
| ✨ Onboarding | 4-step quiz: intro → BFI-10 personality → interests → results |
| ✦ Recommendations | Top-N ranked cards with match %, MP1/MP2 scores, reasons |
| 👤 Profile | Personality bars, interest scores, history, similar users |
| 🌐 HIN Graph | Interactive metapath graph (Plotly) + path traces |
| 📊 Metrics | Precision@N, Recall@N, F1, similarity table |
| ℹ️ About | Paper reference, architecture, equations (LaTeX) |

---

## 🔬 Algorithm (Paper Faithful)

### 1. Personality Inference
- **Existing users**: LIWC-style lexicon scoring over review text
- **New users**: BFI-10 questionnaire (10 items, 2 per trait)

### 2. Interest Mining
TF-IDF cosine similarity between user text and 15 topic keyword documents

### 3. HIN Construction
- **U–U**: `Sim = 0.40·SimP + 0.35·SimT + 0.25·SimI`
- **U–T**: TF-IDF interest weight
- **T–P**: Product belongs to category
- **U–P**: User interacted with product

### 4. Metapath Discovery
- **MP1** U→T→P: `Score(p) = Σ_t w(U,T)`
- **MP2** U→U'→T→P: `Score(p) = Σ_u' Sim(U,U') × w(U',T)`
- **Final**: `Score = 0.60·MP1 + 0.40·MP2`

### 5. Top-N Ranking
Normalised scores → ranked list with match % and explanation

---

## 📊 Dataset Sources

| File | Based On | Link |
|------|----------|------|
| users + products + interactions | Amazon Product Reviews (McAuley, UCSD) | https://jmcauley.ucsd.edu/data/amazon |
| Supplementary | Kaggle Amazon Product Reviews | https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews |

> Note: Datasets are synthetically generated at scale using the same schema and
> realistic distributions as the McAuley Amazon benchmark used in the paper.
> The generator (`generate_data.py`) replicates category distributions, rating
> histograms, and personality-aligned interaction patterns from the paper.
