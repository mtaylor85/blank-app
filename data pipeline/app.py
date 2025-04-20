import os
import zipfile
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
import matplotlib.pyplot as plt

st.title("🛒 Grocery Orders Recommender + Analyzer")

# --- Paths ---
data_dir = "data"
orders_path = os.path.join(data_dir, "orders.csv")
prior_path = os.path.join(data_dir, "order_products_prior.csv")
products_path = os.path.join(data_dir, "products.csv")
aisles_path = os.path.join(data_dir, "aisles.csv")
departments_path = os.path.join(data_dir, "departments.csv")
st.write("✅ Reached here")

# --- Load Data ---
orders_df = pd.read_csv(prior_path)
products_df = pd.read_csv(products_path)
orders_full = pd.read_csv(orders_path)
aisles_df = pd.read_csv(aisles_path)
departments_df = pd.read_csv(departments_path)
st.write("✅ Reached here")

# --- Merge & Clean ---
merged_df = orders_df.merge(products_df, on='product_id', how='left').drop_duplicates()
merged_df = merged_df.dropna(subset=['product_name'])
merged_df.columns = [col.strip().lower().replace(' ', '_') for col in merged_df.columns]
st.write("✅ Reached here")

products_full = products_df.merge(aisles_df, on='aisle_id').merge(departments_df, on='department_id')
merged_df = merged_df.merge(products_full, on='product_id', how='left')
full_df = merged_df.merge(orders_full, on='order_id', how='left')
full_df.columns = [col.strip().lower().replace(' ', '_') for col in full_df.columns]
st.write("✅ Reached here")

# --- Feature Engineering ---
user_product = full_df.groupby(['user_id', 'product_id']).agg(
    times_purchased=('reordered', 'count'),
    last_order_number=('order_number', 'max'),
    avg_cart_position=('add_to_cart_order', 'mean'),
    total_orders=('order_number', 'nunique')
).reset_index()
user_product['reorder_rate'] = user_product['times_purchased'] / user_product['total_orders']
st.write("✅ Reached here")

# --- Recommendation System ---
interaction_matrix = full_df.pivot_table(index='user_id', columns='product_id', values='reordered', fill_value=0)
user_similarity = cosine_similarity(interaction_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=interaction_matrix.index, columns=interaction_matrix.index)

def recommend_products(user_id, num_recs=5):
    if user_id not in user_similarity_df:
        return []
    sim_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:]
    similar_user_orders = interaction_matrix.loc[sim_users].sum().sort_values(ascending=False)
    already_ordered = interaction_matrix.loc[user_id][interaction_matrix.loc[user_id] > 0].index
    recommendations = similar_user_orders.drop(already_ordered).head(num_recs).index.tolist()
    return recommendations
st.write("✅ Reached here")

# --- Streamlit UI ---
user_id = st.number_input("Enter a User ID to Recommend Products For", value=1, step=1)
if st.button("Get Recommendations"):
    recs = recommend_products(user_id)
    if recs:
        rec_names = products_df[products_df['product_id'].isin(recs)]['product_name'].tolist()
        st.success(f"Top {len(rec_names)} Recommendations for User {user_id}:")
        for name in rec_names:
            st.markdown(f"- {name}")
    else:
        st.warning("User ID not found or no recommendations available.")
st.write("✅ Reached here")

# --- Time Series Forecast ---
st.header("📈 Forecasting for a Product (User 123, Product 24852)")

user_sample = full_df[full_df['user_id'] == 123]
product_sample = user_sample[user_sample['product_id'] == 24852]
ts_df = product_sample[['order_number', 'days_since_prior_order']].copy()
ts_df['ds'] = ts_df['order_number']
ts_df['y'] = 1

if len(ts_df) > 2:
    model = Prophet()
    model.fit(ts_df[['ds', 'y']])
    future = model.make_future_dataframe(periods=5)
    forecast = model.predict(future)
    fig = model.plot(forecast)
    st.pyplot(fig)
else:
    st.warning("Not enough data to generate forecast.")
st.write("✅ Reached here")

# --- Classify Products ---
products_df['is_healthy'] = products_df['product_name'].str.contains('organic|vegetable|fruit', case=False)
products_df['is_sustainable'] = products_df['product_name'].str.contains('eco|sustainable|recyclable', case=False)
st.write("✅ Reached here")

# --- Clustering ---
features = user_product[['times_purchased', 'avg_cart_position', 'reorder_rate']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)
kmeans = KMeans(n_clusters=4, random_state=42)
user_product['persona'] = kmeans.fit_predict(X_scaled)
st.write("✅ Reached here")

st.header("🧠 Customer Personas")
st.dataframe(user_product[['user_id', 'product_id', 'persona']].head())
