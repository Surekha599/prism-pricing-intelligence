"""PRISM — global configuration & business constants for NOVA MART synthetic domain."""

RANDOM_SEED = 42
N_CUSTOMERS = 10_000
N_PRODUCTS = 300
DATE_START = "2024-09-01"
DATE_END = "2026-08-22"          # 24 months of history, data "as of" this date

COMPANY = "NOVA MART"
TAGLINE = "Turn customer behavior, market signals and inventory into smarter pricing decisions."
SYNTHETIC_NOTE = "Synthetic demonstration dataset — created for analytical demonstration."

# ---------------------------------------------------------------- categories
# price range (INR), gross margin range, designed price-elasticity range,
# return propensity, festive demand strength
CATEGORIES = {
    "Smartphones":       dict(subs=["Flagship", "Mid-Range", "Budget", "Gaming Phone"],
                              price=(8_999, 124_999), margin=(0.09, 0.17), elas=(1.2, 1.9),
                              ret=0.055, festive=1.55),
    "Laptops":           dict(subs=["Ultrabook", "Gaming", "Business", "Budget"],
                              price=(32_999, 189_999), margin=(0.10, 0.17), elas=(0.9, 1.4),
                              ret=0.050, festive=1.25),
    "Tablets":           dict(subs=["Premium", "Mid-Range", "Budget", "Kids"],
                              price=(14_999, 89_999), margin=(0.12, 0.20), elas=(1.4, 2.0),
                              ret=0.060, festive=1.35),
    "Headphones":        dict(subs=["True Wireless", "Over-Ear ANC", "Wired", "Gaming Headset"],
                              price=(999, 34_999), margin=(0.22, 0.38), elas=(1.4, 2.2),
                              ret=0.080, festive=1.30),
    "Smartwatches":      dict(subs=["Premium", "Fitness", "Budget", "Kids"],
                              price=(1_999, 44_999), margin=(0.18, 0.32), elas=(1.3, 2.0),
                              ret=0.070, festive=1.30),
    "Cameras":           dict(subs=["Mirrorless", "DSLR", "Action", "Instant"],
                              price=(17_999, 159_999), margin=(0.12, 0.20), elas=(0.6, 1.0),
                              ret=0.070, festive=1.15),
    "Gaming":            dict(subs=["Console", "Handheld Console", "Controller", "VR Headset"],
                              price=(2_999, 64_999), margin=(0.12, 0.22), elas=(0.7, 1.2),
                              ret=0.050, festive=1.30),
    "Accessories":       dict(subs=["Power Banks", "Cables & Chargers", "Cases & Covers", "Smart Home"],
                              price=(499, 9_999), margin=(0.28, 0.45), elas=(1.6, 2.4),
                              ret=0.060, festive=1.40),
    "Home Electronics":  dict(subs=["Smart TV", "Soundbar", "Projector", "Smart Speaker"],
                              price=(5_999, 149_999), margin=(0.13, 0.22), elas=(1.1, 1.6),
                              ret=0.050, festive=1.55),
}

# monthly demand curve (1.00 = average month). Oct–Nov festive peak (Diwali),
# Dec–Jan holiday / Republic Day sales, summer trough.
MONTH_CURVE = {1: 1.28, 2: 0.85, 3: 0.85, 4: 0.90, 5: 0.95, 6: 0.90,
               7: 1.00, 8: 1.18, 9: 1.12, 10: 1.45, 11: 1.50, 12: 1.30}
# gaming skews to Dec–Jan (gifting)
MONTH_CURVE_GAMING = {1: 1.45, 2: 0.90, 3: 0.85, 4: 0.90, 5: 0.95, 6: 0.90,
                      7: 1.05, 8: 1.15, 9: 1.10, 10: 1.30, 11: 1.35, 12: 1.50}

# ---------------------------------------------------------------- geography
CITIES = [  # city, state, region, population weight
    ("Mumbai", "Maharashtra", "West", 1.30), ("Delhi", "Delhi", "North", 1.30),
    ("Bengaluru", "Karnataka", "South", 1.20), ("Hyderabad", "Telangana", "South", 1.00),
    ("Chennai", "Tamil Nadu", "South", 0.95), ("Kolkata", "West Bengal", "East", 0.95),
    ("Pune", "Maharashtra", "West", 0.85), ("Ahmedabad", "Gujarat", "West", 0.70),
    ("Jaipur", "Rajasthan", "North", 0.60), ("Lucknow", "Uttar Pradesh", "North", 0.55),
    ("Indore", "Madhya Pradesh", "Central", 0.50), ("Kochi", "Kerala", "South", 0.45),
    ("Chandigarh", "Punjab", "North", 0.40), ("Bhubaneswar", "Odisha", "East", 0.35),
    ("Patna", "Bihar", "East", 0.35),
]

REGIONS = ["North", "South", "East", "West", "Central"]

# ---------------------------------------------------------------- customers
INCOME_BANDS = ["Low", "Middle", "Upper Middle", "High"]
OCCUPATIONS = ["IT Professional", "Student", "Business Owner", "Healthcare", "Finance",
               "Government Service", "Education", "Freelancer", "Manufacturing",
               "Consulting", "Homemaker", "Retired"]
EDUCATION = ["High School", "Diploma", "Bachelor's", "Master's", "Doctorate"]
LOYALTY_TIERS = ["Bronze", "Silver", "Gold", "Platinum"]
ACQ_CHANNELS = ["Organic Search", "Paid Search", "Social Media", "Referral",
                "Marketplace", "Offline Store"]
DEVICES = ["Mobile", "Desktop", "Tablet"]
PAYMENTS = ["UPI", "Credit Card", "Debit Card", "Cash on Delivery", "Net Banking", "Wallet"]

# behavioural archetypes used to synthesise statistically-coherent activity.
# (these are the *latent* segments; the app re-derives segments from RFM + clustering)
SEGMENT_ARCHETYPES = {
    "Premium Loyalists":      dict(share=0.12, orders_per_yr=7.5, deal_beta=0.10,
                                   low_disc_pen=1.00, income=["High", "Upper Middle"],
                                   cats=["Smartphones", "Laptops", "Cameras", "Smartwatches"]),
    "Deal Seekers":           dict(share=0.22, orders_per_yr=4.8, deal_beta=2.60,
                                   low_disc_pen=0.22, income=["Middle", "Low", "Upper Middle"],
                                   cats=["Accessories", "Headphones", "Tablets", "Smartphones"]),
    "High-Value Occasional":  dict(share=0.10, orders_per_yr=1.7, deal_beta=0.25,
                                   low_disc_pen=0.95, income=["High", "Upper Middle"],
                                   cats=["Laptops", "Cameras", "Home Electronics"]),
    "Frequent Budget Buyers": dict(share=0.16, orders_per_yr=6.8, deal_beta=0.90,
                                   low_disc_pen=0.70, income=["Low", "Middle"],
                                   cats=["Accessories", "Headphones", "Smartwatches", "Tablets"]),
    "Window Shoppers":        dict(share=0.25, orders_per_yr=1.1, deal_beta=0.55,
                                   low_disc_pen=0.60, income=["Middle", "Low"],
                                   cats=["Smartphones", "Headphones", "Accessories", "Gaming"]),
    "At-Risk High Value":     dict(share=0.05, orders_per_yr=3.5, deal_beta=0.20,
                                   low_disc_pen=0.95, income=["High", "Upper Middle"],
                                   cats=["Laptops", "Smartphones", "Home Electronics"]),
    "New Customers":          dict(share=0.10, orders_per_yr=2.0, deal_beta=0.45,
                                   low_disc_pen=0.80, income=["Middle", "Upper Middle", "Low"],
                                   cats=["Smartphones", "Headphones", "Accessories", "Smartwatches"]),
}

# ---------------------------------------------------------------- products
BRANDS = ["Novatech", "Aurex", "Zephyr", "Voltcore", "Kinetic", "Lumina", "Sonique",
          "Nexa", "Orbitt", "Pixon", "Aeron", "NovaSound", "NovaView", "NovaPlay"]
SUPPLIERS = ["NexaSupply", "Kestrel Distribution", "BluePeak Traders",
             "MetroLine Wholesale", "Orbit Distribution", "Vardhman Electronics"]
LIFECYCLES = ["Launch", "Growth", "Mature", "Decline"]
LIFECYCLE_W = [0.10, 0.20, 0.55, 0.15]

# name-building vocabulary per subcategory: (series words, model suffixes)
NAME_BANK = {
    "Flagship":        (["Titan", "Zenith", "Prime X", "Aura", "Vertex"], ["Pro", "Ultra", "Max"]),
    "Mid-Range":       (["Pulse", "Nimbus", "Core", "Aster"], ["S", "Plus", "Neo"]),
    "Budget":          (["Lite", "Essential", "Go", "Base"], ["", "E", "C"]),
    "Gaming Phone":    (["Fury", "Raptor", "Storm"], ["G", "Turbo", "X"]),
    "Ultrabook":       (["Book", "Air", "Feather"], ["14", "13", "15"]),
    "Gaming Laptop":   (["Raptor", "Vortex", "Phantom"], ["16", "17", "RTX"]),
    "Business Laptop": (["ProDesk", "Vertex", "Edge"], ["B14", "B15", "Elite"]),
    "Budget Laptop":   (["Essential", "Campus", "GoBook"], ["14", "15"]),
    "Premium":         (["Studio", "Signature", "Lux"], ["P11", "P12", "Pro"]),
    "Mid-Range":       (["Pulse", "Nimbus", "Core"], ["M10", "M11", "Lite"]),
    "Kids":            (["Junior", "PlayMate"], ["K7", "K9"]),
    "True Wireless":   (["Buds", "Pods", "Notes"], ["Air", "Pro", "2", "3"]),
    "Over-Ear ANC":    (["Silence", "Aura", "Halo"], ["Studio", "Max", "ANC"]),
    "Wired":           (["Tone", "Beat"], ["Wired", "Basic"]),
    "Gaming Headset":  (["Echo", "Recon"], ["G7", "G9", "Pro"]),
    "Fitness":         (["Fit", "Active", "Stride"], ["Band", "Track", "2"]),
    "Mirrorless":      (["Vision", "Clarity"], ["M50", "R7", "A1"]),
    "DSLR":            (["Legacy", "Titan"], ["D90", "D500"]),
    "Action":          (["GoVolt", "Rush"], ["4K", "5K", "X"]),
    "Instant":          (["Snap", "Retro"], ["Mini", "Pop"]),
    "Console":         (["Station", "NovaBox"], ["X", "S", "5"]),
    "Handheld Console": (["PlayPort", "NimbusGo"], ["Deck", "Lite"]),
    "Controller":      (["Grip", "PulsePad"], ["Pro", "Elite"]),
    "VR Headset":      (["Immerse", "Portal"], ["V2", "V3"]),
    "Power Banks":     (["VoltCore", "JuicePack"], ["10K", "20K", "27K"]),
    "Cables & Chargers": (["SwiftCable", "ChargePro"], ["C-to-C", "65W", "100W"]),
    "Cases & Covers":   (["ArmorShell", "SkinPro"], ["Clear", "Matte", "Folio"]),
    "Smart Home":       (["HomeSense", "AutoNest"], ["Cam", "Plug", "Hub"]),
    "Smart TV":         (["VisionX", "CineMax"], ["43\"", "55\"", "65\""]),
    "Soundbar":         (["BoomStage", "AudioSlate"], ["2.1", "5.1", "Dolby"]),
    "Projector":        (["BeamCast", "LumaTheatre"], ["HD", "4K", "Short Throw"]),
    "Smart Speaker":    (["EchoNova"], ["Mini", "Standard", "Studio"]),
}
NAME_BANK["Business"] = NAME_BANK["Business Laptop"]
NAME_BANK["Budget Laptop"] = NAME_BANK["Budget Laptop"]

# ---------------------------------------------------------------- promos
PROMO_TYPES = ["No Discount", "Percentage Discount", "Flat Discount", "Bundle",
               "Buy One Get One", "Loyalty Offer", "Flash Sale", "Seasonal Sale"]

# ---------------------------------------------------------------- KPI defs
MIN_MARGIN_FLOOR = 0.08      # contribution margin floor in optimizer
MAX_DISCOUNT = 0.30          # hard cap
PRICE_FLOOR_VS_COMP = 0.85   # never price below 85% of competitor price
PROMO_BUDGET_INR = 4_500_000  # quarterly promo budget constraint for optimizer
