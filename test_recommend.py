from core.data_loader import load_users, load_products, load_interactions, build_existing_profile, build_user_hin
from core.engine import recommend

users = load_users()
products = load_products()
interactions = load_interactions()

uid = users.iloc[0]["uid"]

profile = build_existing_profile(uid, users, interactions)
hin = build_user_hin(profile, products, interactions, users)

recs = recommend(hin, uid, top_n=5)

print("🎯 Recommendation Test\n")

for r in recs:
    print(r["product"]["name"], "| score:", round(r["score"], 3))