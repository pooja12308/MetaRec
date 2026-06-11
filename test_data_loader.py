from core.data_loader import load_users, load_products, load_interactions

print("📂 Testing Data Loader...\n")

users = load_users()
products = load_products()
interactions = load_interactions()

print("Users:", users.shape)
print("Products:", products.shape)
print("Interactions:", interactions.shape)

print("\nSample User:")
print(users.head(1))

print("\n✅ Data Loader Working!")