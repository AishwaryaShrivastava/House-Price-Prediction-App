import streamlit as st
import pandas as pd
import joblib
import os
from pathlib import Path

# Load model
model = joblib.load("model/house_price_model.pkl")

# Constants
USER_CSV = "users.csv"
if not os.path.exists(USER_CSV):
    pd.DataFrame(columns=["username", "password"]).to_csv(USER_CSV, index=False)

# Authentication functions
def signup(username, password):
    username, password = username.strip(), password.strip()
    users = pd.read_csv(USER_CSV)
    if username in users["username"].astype(str).values:
        return False
    new_user = pd.DataFrame([[username, password]], columns=["username", "password"])
    users = pd.concat([users, new_user], ignore_index=True)
    users.to_csv(USER_CSV, index=False)
    return True

def login(username, password):
    username, password = username.strip(), password.strip()
    users = pd.read_csv(USER_CSV)
    users["username"] = users["username"].astype(str).str.strip()
    users["password"] = users["password"].astype(str).str.strip()
    return not users[(users["username"] == username) & (users["password"] == password)].empty

# Session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

# Page config
st.set_page_config(page_title="House Price Prediction App", layout="wide")

# Background color (neon light blue/green)
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #b2fefa, #0ed2f7);
        color: #000000;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🏠 Welcome to House Price Prediction App")

dropdown_mappings = {
    "Location Category": ["Urban", "Suburban", "Rural"],
    "Road Type": ["Dirt", "Paved"],
    "Zoning Type": ["Residential", "Commercial", "Agricultural", "Industrial"],
    "Water Supply Quality": ["Poor", "Average", "Good", "Excellent"],
    "Electricity Reliability": ["Poor", "Average", "Good", "Excellent"],
    "Internet Speed": ["Slow", "Moderate", "Fast", "Very Fast"],
    "Greenery Score": ["Poor", "Average", "Good", "Excellent"],
    "Pollution Index": ["High", "Medium", "Low"],
    "Land Slope": ["Flat", "Moderate", "Steep"],
    "Soil Quality Index": ["Poor", "Average", "Good"],
    "Earthquake Resistance": ["No", "Partial", "Yes"],
    "Public Transport Availability": ["None", "Some", "Frequent"]
}

# Login / Signup
if not st.session_state.logged_in:
    menu = ["Login", "Signup"]
    choice = st.sidebar.selectbox("Select Action", menu)

    if choice == "Signup":
        st.subheader("Create Account")
        new_user = st.text_input("Username")
        new_pass = st.text_input("Password", type='password')
        if st.button("Signup"):
            if signup(new_user, new_pass):
                st.success("✅ Account created! Please login.")
            else:
                st.error("❌ Username already exists!")

    elif choice == "Login":
        st.subheader("Login")
        u = st.text_input("Username")
        p = st.text_input("Password", type='password')
        if st.button("Login"):
            if login(u, p):
                st.session_state.logged_in = True
                st.session_state.username = u
                st.rerun()
            else:
                st.error("❌ Invalid username or password.")

# Main app
else:
    st.success(f"✅ Welcome {st.session_state.username}!")
    st.subheader("📊 Predict House & Land Price")

    inputs = {}
    for feature in model.feature_names_in_:
        if feature in dropdown_mappings:
            inputs[feature] = st.selectbox(f"{feature}", dropdown_mappings[feature])
        elif feature.lower() == "area in square feet":
            inputs[feature] = st.number_input("Area in Square Feet", min_value=1000.0, max_value=1e7, step=100.0)
        elif feature.lower() == "number of bedrooms":
            inputs[feature] = st.number_input("Number of Bedrooms", min_value=0, step=1)
        elif feature.lower() == "number of bathrooms":
            inputs[feature] = st.number_input("Number of Bathrooms", min_value=0, step=1)
        elif feature.lower() == "year built":
            inputs[feature] = st.number_input("Year Built", min_value=1900, max_value=3000, step=1)
        elif feature.lower() == "renovation year":
            inputs[feature] = st.selectbox("Renovation Year", [2020, 2021, 2022, 2023, 2024, 2025])
        else:
            inputs[feature] = st.number_input(f"{feature}", step=1.0)

    if st.button("Predict Price"):
        df = pd.DataFrame([inputs])
        prediction = model.predict(df)[0]
        st.success(f"💰 Estimated Price: ₹ {int(prediction):,}")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()
