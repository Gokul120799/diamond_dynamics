import streamlit as st
import pandas as pd
import pickle

# Load trained models
reg_model = pickle.load(open("regression_model.pkl", "rb"))
cluster_model = pickle.load(open("clustering_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("💎 Diamond Dynamics: Price Prediction & Market Segmentation")

# Numeric inputs
carat = st.number_input("Carat", min_value=0.1, step=0.01)
x = st.number_input("Length (x)", min_value=0.1, step=0.01)
y = st.number_input("Width (y)", min_value=0.1, step=0.01)
z = st.number_input("Depth (z)", min_value=0.1, step=0.01)

# Select box inputs
cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
color = st.selectbox("Color", ["D","E","F","G","H","I","J"])
clarity = st.selectbox("Clarity", ["IF","VVS1","VVS2","VS1","VS2","SI1","SI2","I1"])

# Encoding maps (must match training)
cut_map = {"Fair":0,"Good":1,"Very Good":2,"Premium":3,"Ideal":4}
color_map = {"D":0,"E":1,"F":2,"G":3,"H":4,"I":5,"J":6}
clarity_map = {"IF":0,"VVS1":1,"VVS2":2,"VS1":3,"VS2":4,"SI1":5,"SI2":6,"I1":7}

# Prepare input row (no depth/table anymore)
input_data = pd.DataFrame([{
    "carat": carat,
    "x": x,
    "y": y,
    "z": z,
    "cut": cut_map[cut],
    "color": color_map[color],
    "clarity": clarity_map[clarity],
    "volume": x*y*z,
    "price_per_carat": 0  # placeholder
}])

# Price Prediction
if st.button("Predict Price"):
    price_pred = reg_model.predict(input_data)[0]
    st.success(f"Predicted Diamond Price: ₹{price_pred*83:.2f}")

# Cluster Prediction
if st.button("Predict Cluster"):
    scaled_input = scaler.transform(input_data)
    cluster_pred = cluster_model.predict(scaled_input)[0]
    cluster_names = {
        0: "Premium Heavy Diamonds",
        1: "Affordable Small Diamonds",
        2: "Mid-range Balanced Diamonds"
    }
    st.info(f"Cluster: {cluster_names.get(cluster_pred, 'Unknown')}")
