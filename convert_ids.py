import pandas as pd

# Load your dataset
df = pd.read_csv("retail_store_inventory.csv")

def convert_store_id(x):
    x = str(x).strip()
    if x.upper().startswith("S"):
        return x.upper()  # already formatted
    else:
        return f"S{int(x):03d}"

def convert_product_id(x):
    x = str(x).strip()
    if x.upper().startswith("P"):
        return x.upper()  # already formatted
    else:
        return f"P{int(x):04d}"

# Apply conversions safely
df["Store ID"] = df["Store ID"].apply(convert_store_id)
df["Product ID"] = df["Product ID"].apply(convert_product_id)

df.to_csv("retail_store_inventory.csv", index=False)
print("✅ Store and Product IDs are now consistent and formatted as S### / P####!")
