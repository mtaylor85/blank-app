# Applied the Streamlit fix so that we verify that it is installed and working
# !{sys.executable} -m pip install streamlit
# !{sys.executable} -m streamlit --version
# !{sys.executable} -m streamlit run smartcartai_app.py
# Mathew Taylor has confirmed Streamlit is installed and working as expected.

# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Display the app title
st.title("SmartCartAI: Product Health & Sustainability Predictor")

# Load dataset with caching to improve performance
@st.cache_data
def load_data():
    return pd.read_csv("smartcartai_hypothetical_dataset.csv")

df = load_data()

# Display raw data
if st.checkbox("Show Raw Data"):
    st.write(df.head())

# Preprocessing
features = ['category', 'brand', 'quantity', 'price_per_unit']
label_health = 'health_label'
label_sustainability = 'sustainability_label'

# Encode categorical features
df_encoded = df.copy()
le_category = LabelEncoder()
le_brand = LabelEncoder()
df_encoded['category'] = le_category.fit_transform(df_encoded['category'])
df_encoded['brand'] = le_brand.fit_transform(df_encoded['brand'])

# Split data
X = df_encoded[features]
y_health = df_encoded[label_health]
y_sustainability = df_encoded[label_sustainability]

X_train, X_test, y_train_h, y_test_h = train_test_split(X, y_health, test_size=0.2, random_state=42)
X_train, X_test, y_train_s, y_test_s = train_test_split(X, y_sustainability, test_size=0.2, random_state=42)

# Train models
rf_health = RandomForestClassifier(n_estimators=100, random_state=42)
rf_health.fit(X_train, y_train_h)

rf_sustain = RandomForestClassifier(n_estimators=100, random_state=42)
rf_sustain.fit(X_train, y_train_s)

# User input
st.sidebar.header("Input Product Features")
cat_input = st.sidebar.selectbox("Category", df['category'].unique())
brand_input = st.sidebar.selectbox("Brand", df['brand'].unique())
qty_input = st.sidebar.slider("Quantity", 1, 5, 2)
price_input = st.sidebar.slider("Price per Unit ($)", 1.0, 10.0, 5.0)

# Encode input
input_data = pd.DataFrame([{
    'category': le_category.transform([cat_input])[0],
    'brand': le_brand.transform([brand_input])[0],
    'quantity': qty_input,
    'price_per_unit': price_input
}])

# Make predictions using the trained models
pred_health = rf_health.predict(input_data)[0]
pred_sustain = rf_sustain.predict(input_data)[0]

# Display prediction results to the user
st.subheader("Predicted Labels")
st.write(f"**Health Label:** {pred_health}")
st.write(f"**Sustainability Label:** {pred_sustain}")
