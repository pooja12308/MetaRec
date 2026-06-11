"""
generate_data.py
================
Generates the full dataset used by MetaRec.

Sizes:
  users.csv      →  100,000 users  (uid, name, personality_type, top_topics)
  products.csv   →   30,000 products  (pid, name, category, description, price, rating, brand)
  interactions.csv → 200,000 user-product interactions  (uid, pid, rating, review, ts)

NOTE: "300k products" with rich unique descriptions = ~600 MB disk + unusable TF-IDF.
Industry benchmarks (Amazon-5core, MovieLens-20M) use 10k-100k items.
We generate 30,000 richly described products — same density used in the paper's
Amazon dataset experiments — and report "30,000 products across 15 categories"
accurately in the UI.

Run once:  python generate_data.py
"""

import csv, random, os
from datetime import datetime, timedelta

random.seed(2024)
OUT = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT, exist_ok=True)

# ── Big Five types (paper uses OCEAN) ─────────────────────────────────────────
OCEAN = ["Openness","Conscientiousness","Extraversion","Agreeableness","Neuroticism"]

# ── 15 product categories ─────────────────────────────────────────────────────
CATEGORIES = [
    "Electronics","Sports & Fitness","Fashion & Accessories","Travel & Outdoors",
    "Music & Instruments","Gaming","Books & Education","Kitchen & Cooking",
    "Photography","Art & Design","Personal Finance","Health & Wellness",
    "Home & Living","Automotive","Pet Care",
]

# Personality → category affinity (from paper's ODP-based interest model)
PERSONALITY_CATEGORY = {
    "Openness":          ["Art & Design","Books & Education","Music & Instruments",
                          "Travel & Outdoors","Photography","Electronics"],
    "Conscientiousness": ["Personal Finance","Health & Wellness","Kitchen & Cooking",
                          "Books & Education","Electronics","Home & Living"],
    "Extraversion":      ["Sports & Fitness","Fashion & Accessories","Music & Instruments",
                          "Travel & Outdoors","Gaming","Pet Care"],
    "Agreeableness":     ["Health & Wellness","Kitchen & Cooking","Pet Care",
                          "Home & Living","Books & Education","Art & Design"],
    "Neuroticism":       ["Gaming","Books & Education","Art & Design","Music & Instruments",
                          "Health & Wellness","Electronics"],
}

# Category → brands & adjectives for realistic product name generation
CAT_META = {
    "Electronics":          (["Sony","Samsung","Apple","Logitech","Anker","Bose","JBL","Corsair","LG","Belkin"],
                             ["Wireless","Smart","Ultra","Pro","Nano","Slim","Turbo","4K"],
                             ["Headphones","Speaker","Keyboard","Webcam","SSD","Charger","Monitor","Earbuds","Router","Tablet"]),
    "Sports & Fitness":     (["Nike","Adidas","Puma","Under Armour","Reebok","Asics","Manduka","Fitbit","Garmin","TheraBand"],
                             ["Pro","Elite","Flex","Power","Swift","Max","Active","Core"],
                             ["Running Shoes","Yoga Mat","Resistance Bands","Dumbbell Set","Gym Bag","Fitness Tracker","Jump Rope","Foam Roller"]),
    "Fashion & Accessories":(["Zara","Ray-Ban","Fossil","Levi's","Titan","H&M","Coach","Guess","Casio","Tommy"],
                             ["Minimalist","Classic","Premium","Vintage","Slim","Bold","Luxe","Casual"],
                             ["Watch","Sunglasses","Wallet","Leather Belt","Tote Bag","Cap","Bracelet","Sneakers","Scarf","Loafers"]),
    "Travel & Outdoors":    (["Pacsafe","Samsonite","Osprey","Victorinox","Eagle Creek","Away","Tumi","Deuter","Gregory","Thule"],
                             ["Anti-Theft","Waterproof","Lightweight","Foldable","Compact","Durable","Smart"],
                             ["Backpack","Neck Pillow","Travel Adapter","Packing Cubes","Luggage","Passport Holder","Hammock","Trekking Poles"]),
    "Music & Instruments":  (["Yamaha","Fender","Roland","Casio","Shure","Audio-Technica","Korg","Boss","AKG","Blue"],
                             ["Acoustic","Digital","Portable","Professional","Studio","Wireless","Electric"],
                             ["Guitar","Keyboard","Microphone","Ukulele","Drum Pad","Violin","Capo","Metronome","Harmonica"]),
    "Gaming":               (["Razer","Corsair","SteelSeries","Logitech","HyperX","ASUS ROG","Secretlab","MSI","Elgato","BenQ"],
                             ["Mechanical","Wireless","RGB","Ultra","Pro","Hyper","Tactile","Precision"],
                             ["Gaming Mouse","Keyboard","Headset","Chair","Monitor","Controller","Mousepad","Capture Card","LED Strip"]),
    "Books & Education":    (["Penguin","HarperCollins","Oxford","McGraw-Hill","Scholastic","Wiley","Pearson","Bloomsbury","MIT Press","Routledge"],
                             ["Bestselling","Illustrated","Complete","Essential","Award-Winning","Revised","Annotated"],
                             ["Novel","Biography","Self-Help Guide","Textbook","Thriller","Science Book","History","Psychology","Business Guide","Memoir"]),
    "Kitchen & Cooking":    (["Lodge","Instant Pot","KitchenAid","Ninja","Cuisinart","Tefal","Philips","Prestige","Hawkins","Wonderchef"],
                             ["Non-Stick","Cast Iron","Stainless","Professional","Smart","Ceramic","Multi-Function"],
                             ["Skillet","Pressure Cooker","Air Fryer","Blender","Coffee Maker","Knife Set","Stand Mixer","Dutch Oven","Wok","Toaster"]),
    "Photography":          (["Joby","Manfrotto","Neewer","Godox","Lowepro","Peak Design","Hoya","SanDisk","Benro","K&F Concept"],
                             ["Professional","Carbon Fibre","Waterproof","Foldable","360°","Wireless","Portable"],
                             ["Tripod","Ring Light","Camera Bag","Lens Filter","Memory Card","Camera Strap","Backdrop","Reflector","Light Box"]),
    "Art & Design":         (["Wacom","Winsor & Newton","Faber-Castell","Sakura","Prismacolor","Staedtler","Moleskine","Arteza","Mont Marte","Liquitex"],
                             ["Professional","Student","Premium","Portable","Digital","Eco-Friendly","Artist-Grade"],
                             ["Drawing Tablet","Watercolour Set","Sketchbook","Acrylic Paints","Colour Pencils","Brush Set","Canvas","Easel","Pastel Set"]),
    "Personal Finance":     (["Clever Fox","Panda Planner","Leuchtturm","Wiley","McGraw-Hill","Penguin","Moleskine","Full Focus","Routledge","HarperCollins"],
                             ["Personal","Guided","Annual","Weekly","Undated","Smart","Comprehensive","Structured"],
                             ["Budget Planner","Finance Journal","Investment Guide","Tax Organiser","Expense Tracker","Savings Log","Goal Planner","Portfolio Book"]),
    "Health & Wellness":    (["Optimum Nutrition","GNC","Now Foods","Garden of Life","Himalaya","Dabur","Theragun","Hyperice","doTERRA","Calm"],
                             ["Organic","Natural","Advanced","Clinical","Essential","Daily","Premium","Pure"],
                             ["Protein Powder","Vitamin Supplement","Essential Oil","Massage Gun","Probiotic","Omega-3","Collagen Powder","Diffuser","Sleep Aid"]),
    "Home & Living":        (["IKEA","Dyson","Philips","Nest","Xiaomi","Lifx","Muji","Umbra","Joseph Joseph","OXO"],
                             ["Smart","Ergonomic","Minimalist","Compact","Modern","Eco","Adjustable"],
                             ["Desk Lamp","Air Purifier","Robot Vacuum","Smart Bulb","Storage Organiser","Wall Clock","Throw Blanket","Candle Set","Planter"]),
    "Automotive":           (["3M","Meguiar's","Armor All","WeatherTech","Garmin","Thule","Black+Decker","Chemical Guys","Turtle Wax","Bosch"],
                             ["Heavy-Duty","Premium","Portable","Wireless","Universal","Professional","Waterproof"],
                             ["Car Vacuum","Dash Cam","Car Charger","Seat Cover","Tyre Inflator","Car Mount","Polish Kit","Jump Starter","Parking Sensor"]),
    "Pet Care":             (["Kong","Purina","Royal Canin","Furminator","Ruffwear","PetSafe","Bark","Chewy","Blue Buffalo","Hill's"],
                             ["Interactive","Organic","Premium","Durable","Automatic","Ergonomic","Natural"],
                             ["Dog Toy","Cat Tree","Pet Carrier","Grooming Kit","Automatic Feeder","Pet Bed","Training Treats","Collar","Leash","Harness"]),
}

DESC_TMPL = [
    "{adj} {noun} by {brand}. Features {f1} and {f2}. Ideal for {u1} and {u2}. {benefit}",
    "The {brand} {adj} {noun} delivers {f1} with {f2}. Perfect for {u1}. {benefit}",
    "Experience {f1} with the {brand} {adj} {noun}. Designed for {u1} and {u2}. {benefit}",
]
FEATURES = ["superior build quality","long-lasting durability","ergonomic design",
            "ultra-lightweight construction","waterproof protection","premium materials",
            "fast performance","advanced technology","easy setup","compact form factor",
            "multi-device compatibility","precision engineering","eco-friendly materials",
            "adjustable fit","360-degree coverage","whisper-quiet operation","smart connectivity"]
USE_CASES = ["everyday use","professional work","outdoor adventures","home workouts",
             "travel and commuting","creative projects","students and beginners",
             "long sessions","athletes and enthusiasts","professionals and experts"]
BENEFITS  = ["Backed by a 2-year warranty.","Rated 4.5+ stars by thousands.",
             "Ships same day.","Award-winning design.","Trusted by professionals.",
             "Free returns within 30 days.","Loved by 50,000+ buyers."]

FIRST = ["Aarav","Aditya","Akash","Amit","Ananya","Anjali","Arjun","Aryan","Divya","Ishaan",
         "Ishita","Karan","Kavya","Keerthi","Lekhana","Meera","Neha","Nikhil","Pooja","Priya",
         "Rahul","Riya","Rohan","Sahil","Shreya","Sid","Sneha","Tanya","Uday","Vikram","Yash",
         "Emma","Liam","Olivia","Noah","Ava","James","Sofia","Lucas","Isabella","Mason","Mia",
         "Ethan","Charlotte","Logan","Amelia","Jackson","Harper","Sebastian","Evelyn","Daniel",
         "Emily","Henry","Elizabeth","Alexander","Mila","Owen","Camila","Gabriel","Penelope",
         "Wei","Mei","Jun","Xiao","Li","Na","Jing","Lei","Bo","Raj","Preet","Simran","Sunita"]
LAST  = ["Sharma","Patel","Gupta","Singh","Reddy","Iyer","Mehta","Joshi","Kumar","Nair",
         "Das","Mishra","Rao","Pillai","Bose","Chatterjee","Agarwal","Verma","Menon","Shah",
         "Khan","Ahmed","Ali","Hassan","Smith","Johnson","Williams","Brown","Jones","Garcia",
         "Miller","Davis","Wilson","Anderson","Taylor","Thomas","Jackson","White","Harris",
         "Chen","Wang","Zhang","Liu","Yang","Huang","Zhou","Wu","Xu","Sun","Ma","Zhu",
         "Kaur","Gill","Dhaliwal","Sidhu","Grewal","Bains","Sandhu"]

BASE_DATE = datetime(2022, 1, 1)
def rdate(): return (BASE_DATE + timedelta(days=random.randint(0,900))).strftime("%Y-%m-%d")

REVIEW_TEMPLATES = {
    5: ["{cat} product is outstanding. {f} and {f2}. Highly recommend to everyone.",
        "Absolutely love this! {f}. Perfect for {u}. Five stars without hesitation.",
        "Exceeded all my expectations. {f} and {f2}. Great value for the price."],
    4: ["{cat} product is very good. {f}. Minor room for improvement but overall excellent.",
        "Really happy with this purchase. {f}. Would buy again.",
        "Great product overall. {f}. Solid quality and fast delivery."],
    3: ["{cat} product is decent. {f}. But {n} could be better.",
        "Average experience. {f} is good but {n} was disappointing.",
        "OK product. Does what it says. {n} needs improvement."],
}
NEGATIVES = ["the packaging","the instructions","the colour accuracy","the size",
             "the battery life","the build finish","customer support"]


# ─────────────────────────────────────────────────────────────────────────────
print("═" * 55)
print("  MetaRec Dataset Generator")
print("═" * 55)

# ── 1. Products ───────────────────────────────────────────────────────────────
N_PROD = 30_000
print(f"\n[1/3] Generating {N_PROD:,} products...")
pids, pcat = [], {}
seen_names = set()
with open(f"{OUT}/products.csv","w",newline="",encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["pid","name","category","description","price","rating","brand"])
    cat_weights = [1]*len(CATEGORIES)
    for i in range(1, N_PROD+1):
        cat   = random.choices(CATEGORIES, k=1)[0]
        brands, adjs, nouns = CAT_META[cat]
        brand = random.choice(brands)
        adj   = random.choice(adjs)
        noun  = random.choice(nouns)
        sfx   = random.choice(["","","","Pro","Plus","Elite","Max","X","SE","V2","Lite",""])
        name  = f"{brand} {adj} {noun}{' '+sfx if sfx else ''}"
        c = 0
        while name in seen_names and c < 8:
            adj = random.choice(adjs); noun = random.choice(nouns)
            name = f"{brand} {adj} {noun}{' '+sfx if sfx else ''}"; c+=1
        seen_names.add(name)
        desc = random.choice(DESC_TMPL).format(
            brand=brand, adj=adj.lower(), noun=noun.lower(),
            f1=random.choice(FEATURES), f2=random.choice(FEATURES),
            u1=random.choice(USE_CASES), u2=random.choice(USE_CASES),
            benefit=random.choice(BENEFITS)
        )
        price  = round(random.uniform(5, 2500), 2)
        rating = round(max(1.0, min(5.0, random.gauss(4.1, 0.55))), 1)
        pid    = f"P{i:06d}"
        pids.append(pid); pcat[pid] = cat
        w.writerow([pid, name, cat, desc, price, rating, brand])
print(f"  ✓ {N_PROD:,} products across {len(CATEGORIES)} categories")

# ── 2. Users ──────────────────────────────────────────────────────────────────
N_USERS = 100_000
print(f"\n[2/3] Generating {N_USERS:,} users...")
uids = []
with open(f"{OUT}/users.csv","w",newline="",encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["uid","name","personality_type","top_topics","created_at"])
    seen_names_u = {}
    for i in range(1, N_USERS+1):
        fn   = random.choice(FIRST); ln = random.choice(LAST)
        base = f"{fn} {ln}"
        cnt  = seen_names_u.get(base, 0)
        seen_names_u[base] = cnt + 1
        name = base if cnt == 0 else f"{base} {cnt+1}"
        pers = random.choice(OCEAN)
        tops = PERSONALITY_CATEGORY[pers]
        top2 = "|".join(random.sample(tops, min(3, len(tops))))
        uid  = f"U{i:06d}"
        uids.append(uid)
        w.writerow([uid, name, pers, top2, rdate()])
print(f"  ✓ {N_USERS:,} users  (distribution across 5 OCEAN types)")

# ── 3. Interactions ───────────────────────────────────────────────────────────
N_INT = 200_000
print(f"\n[3/3] Generating {N_INT:,} interactions...")
# Build pid-by-category index for realistic matching
cat_pids = {}
for pid, cat in pcat.items():
    cat_pids.setdefault(cat, []).append(pid)

with open(f"{OUT}/interactions.csv","w",newline="",encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["uid","pid","rating","review","timestamp"])
    seen = set()
    count = 0
    user_personalities = {}
    # Load personality map quickly
    with open(f"{OUT}/users.csv","r",encoding="utf-8") as uf:
        reader = csv.DictReader(uf)
        for row in reader:
            user_personalities[row["uid"]] = (row["personality_type"], row["top_topics"].split("|"))

    while count < N_INT:
        uid  = random.choice(uids)
        pers, top_topics = user_personalities[uid]
        # 70% chance pick from personality-aligned categories
        if random.random() < 0.70 and top_topics:
            cat = random.choice(top_topics)
            pool = cat_pids.get(cat, pids)
        else:
            pool = pids
        pid = random.choice(pool)
        if (uid, pid) in seen: continue
        seen.add((uid, pid))

        rating = random.choices([1,2,3,4,5], weights=[2,4,10,35,49])[0]
        r_tmpl = random.choice(REVIEW_TEMPLATES.get(rating, REVIEW_TEMPLATES[3]))
        cat_w  = pcat[pid]
        f1, f2 = random.choice(FEATURES), random.choice(FEATURES)
        u1     = random.choice(USE_CASES)
        neg    = random.choice(NEGATIVES)
        review = r_tmpl.format(cat=cat_w, f=f1, f2=f2, u=u1, n=neg)
        w.writerow([uid, pid, rating, review, rdate()])
        count += 1
        if count % 50000 == 0:
            print(f"    {count:,}/{N_INT:,}")

print(f"  ✓ {N_INT:,} interactions written")
print(f"\n{'═'*55}")
print("  ✅  All datasets generated!")
print(f"  users.csv        → {N_USERS:,} users")
print(f"  products.csv     → {N_PROD:,} products")
print(f"  interactions.csv → {N_INT:,} interactions")
print(f"{'═'*55}\n")
