from core.data_loader import load_users, load_interactions, build_existing_profile

users = load_users()
interactions = load_interactions()

uid = users.iloc[0]["uid"]

profile = build_existing_profile(uid, users, interactions)

print("🧠 Profile Test\n")
print("User:", uid)
print("Dominant Trait:", profile["dominant_trait"])
print("Top Interests:", profile["topic_interests"][:3])