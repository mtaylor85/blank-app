
import streamlit as st
import pandas as pd
import numpy as np

st.title("SmartCartAI: Budget-Based Grocery Recommender")

@st.cache_data
def load_data():
    return pd.read_csv("smartcartai_hypothetical_dataset.csv")

df = load_data()

# Sidebar inputs
st.sidebar.header("Your Preferences")
budget = st.sidebar.number_input("Enter your total grocery budget ($)", min_value=5.0, max_value=500.0, value=50.0, step=1.0)
health_pref = st.sidebar.selectbox("Prefer Healthy Items?", ["Doesn't matter", "Healthy only"])
sustain_pref = st.sidebar.selectbox("Prefer Sustainable Items?", ["Doesn't matter", "Sustainable only"])

# Filter based on health and sustainability preferences
filtered_df = df.copy()
if health_pref == "Healthy only":
    filtered_df = filtered_df[filtered_df["health_label"] == "Healthy"]
if sustain_pref == "Sustainable only":
    filtered_df = filtered_df[filtered_df["sustainability_label"] == "Sustainable"]

# Drop duplicates to avoid multiple identical entries
filtered_df = filtered_df.drop_duplicates(subset=["product_name", "brand"])

# Sort by price per unit (ascending)
filtered_df = filtered_df.sort_values(by="price_per_unit").reset_index(drop=True)

# Generate a recommended shopping list within budget
shopping_list = []
total_cost = 0.0

for _, row in filtered_df.iterrows():
    if total_cost + row["price_per_unit"] <= budget:
        shopping_list.append({
            "Product": row["product_name"],
            "Brand": row["brand"],
            "Price per Unit": row["price_per_unit"]
        })
        total_cost += row["price_per_unit"]

# Display results
st.subheader("Recommended Shopping List")
if shopping_list:
    rec_df = pd.DataFrame(shopping_list)
    st.dataframe(rec_df)
    st.markdown(f"**Total Estimated Cost:** ${round(total_cost, 2)}")
else:
    st.warning("No items found within your budget and preferences. Try increasing your budget or relaxing filters.")

if st.checkbox("Show Raw Data"):
    st.write(df.head())
