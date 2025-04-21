
import streamlit as st
import pandas as pd
import numpy as np

st.title("SmartCartAI: Budget-Based Grocery Recommender")

@st.cache_data
def load_data():
    return pd.read_csv("smartcartai_hypothetical_dataset.csv")

df = load_data()

# Sidebar: Budget & Preferences
st.sidebar.header("Your Preferences")
budget = st.sidebar.number_input("Enter your total grocery budget ($)", min_value=5.0, max_value=500.0, value=50.0, step=1.0)
health_pref = st.sidebar.selectbox("Prefer Healthy Items?", ["Doesn't matter", "Healthy only"])
sustain_pref = st.sidebar.selectbox("Prefer Sustainable Items?", ["Doesn't matter", "Sustainable only"])

# Categories
all_categories = df['category'].unique().tolist()
default_categories = ["Produce", "Meat", "Dairy"]
selected_categories = st.sidebar.multiselect("Select up to 9 food categories:", options=all_categories, default=default_categories, max_selections=9)

# Category ranking and quantity preference
st.sidebar.markdown("### Rank Category Importance (1 = highest)")
category_preferences = {}
for cat in selected_categories:
    col1, col2 = st.sidebar.columns([2, 2])
    with col1:
        rank = st.number_input(f"{cat} rank", min_value=1, max_value=9, value=selected_categories.index(cat)+1, key=f"{cat}_rank")
    with col2:
        prefer_quantity = st.checkbox(f"More of {cat}", key=f"{cat}_quantity")
    category_preferences[cat] = {"rank": rank, "prefer_quantity": prefer_quantity}

# Filter dataset
filtered_df = df[df["category"].isin(selected_categories)]

if health_pref == "Healthy only":
    filtered_df = filtered_df[filtered_df["health_label"] == "Healthy"]
if sustain_pref == "Sustainable only":
    filtered_df = filtered_df[filtered_df["sustainability_label"] == "Sustainable"]

filtered_df = filtered_df.drop_duplicates(subset=["product_name", "brand"])
filtered_df = filtered_df.sort_values(by="price_per_unit").reset_index(drop=True)

# Apply preference scoring
def score_item(row):
    cat = row["category"]
    base_score = 10 - category_preferences[cat]["rank"]
    if category_preferences[cat]["prefer_quantity"]:
        base_score += 2
    return base_score / (row["price_per_unit"] + 1)

filtered_df["score"] = filtered_df.apply(score_item, axis=1)
filtered_df = filtered_df.sort_values(by="score", ascending=False).reset_index(drop=True)

# Build shopping list within budget with quantity modifiers
shopping_list = []
used_budget = 0.0

for _, row in filtered_df.iterrows():
    unit_price = row["price_per_unit"]
    max_quantity = int((budget - used_budget) // unit_price)
    if max_quantity >= 1:
        preferred_quantity = 2 if category_preferences[row["category"]]["prefer_quantity"] else 1
        quantity = min(preferred_quantity, max_quantity)
        total_price = quantity * unit_price
        used_budget += total_price

        shopping_list.append({
            "Product": row["product_name"],
            "Brand": row["brand"],
            "Category": row["category"],
            "Price per Unit": unit_price,
            "Quantity": quantity,
            "Total Price": round(total_price, 2)
        })

# Display shopping list
st.subheader("Recommended Shopping List")
if shopping_list:
    rec_df = pd.DataFrame(shopping_list)
    st.dataframe(rec_df)
    st.markdown(f"**Total Estimated Cost:** ${round(used_budget, 2)}")
else:
    st.warning("No items found within your budget and preferences. Try adjusting filters or increasing budget.")

if st.checkbox("Show Raw Data"):
    st.write(df.head())
