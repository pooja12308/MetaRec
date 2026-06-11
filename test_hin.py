from core.data_loader import load_users, load_products, load_interactions, build_existing_profile, build_user_hin

users = load_users()
products = load_products()
interactions = load_interactions()

uid = users.iloc[0]["uid"]

profile = build_existing_profile(uid, users, interactions)

hin = build_user_hin(profile, products, interactions, users)

print("🌐 HIN Test\n")
print("Nodes:", hin.stats())