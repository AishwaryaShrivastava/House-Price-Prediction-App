import streamlit as st
import pandas as pd
import joblib
import os
import bcrypt
import matplotlib.pyplot as plt
from pathlib import Path

# Constants
MODEL_PATH = "model/house_price_model.pkl"
USER_CSV = "users.csv"

# Create users.csv if missing
if not os.path.exists(USER_CSV):
    pd.DataFrame(columns=["username", "password"]).to_csv(USER_CSV, index=False)

# Streamlit page setup
st.set_page_config(page_title="House Price Predictor", layout="wide")
st.markdown("<h1 style='color:white;'>🏠 Welcome to the House Price Predictor</h1>", unsafe_allow_html=True)

# CSS Styling
st.markdown("""
    <style>
         .stApp {
            background-image: url('https://images.unsplash.com/photo-1568605114967-8130f3a36994');
            background-size: cover;
            background-attachment: fixed;
        }
        section[data-testid="stSidebar"] {
            background-color: white;
            color: black;
        }
        .main > div {
            background-color: rgba(0, 0, 0, 0.7);
            border-radius: 15px;
            padding: 2rem;
            margin: 2rem;
            color: white !important;
        }
        h1, h2, h3, h4, h5, h6, label {
            color: white !important;
        }
        input, select, textarea {
            color: black !important;
            background-color: rgba(255,255,255,0.95) !important;
        }
        .stButton > button {
            color: black !important;
            background-color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- Authentication ----------
def signup(username, password):
    if not username or not password:
        return False
    df = pd.read_csv(USER_CSV)
    if username in df["username"].values:
        return False
    hashed_pw = bcrypt.hashpw(password.strip().encode('utf-8'), bcrypt.gensalt()).decode()
    df = pd.concat([df, pd.DataFrame([[username, hashed_pw]], columns=["username", "password"])], ignore_index=True)
    df.to_csv(USER_CSV, index=False)
    return True

def login(username, password):
    df = pd.read_csv(USER_CSV)
    user_row = df[df["username"] == username]
    if user_row.empty:
        return False
    stored_hash = user_row.iloc[0]["password"]
    if pd.isna(stored_hash) or not isinstance(stored_hash, str):
        return False
    return bcrypt.checkpw(password.strip().encode('utf-8'), stored_hash.encode('utf-8'))

# Session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "page" not in st.session_state:
    st.session_state.page = "Login"

# Sidebar
st.sidebar.markdown("<h3 style='color:black;'>🏡 Navigation</h3>", unsafe_allow_html=True)
action = st.sidebar.selectbox("Select Action", ["Login", "Signup", "Predict Price"])

# Login
if action == "Login" and not st.session_state.logged_in:
    st.markdown("## Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if login(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.page = "Predict"
            st.markdown(f"<h4 style='color:white;'>✅ Welcome {username}!</h4>", unsafe_allow_html=True)
        else:
            st.error("❌ Invalid credentials")

# Signup
elif action == "Signup" and not st.session_state.logged_in:
    st.markdown("## Signup")
    username = st.text_input("Choose Username")
    password = st.text_input("Choose Password", type="password")
    if st.button("Create Account"):
        if signup(username, password):
            st.success("✅ Account created! Please login.")
        else:
            st.warning("⚠️ Username already exists or empty input.")

# Prediction
if st.session_state.logged_in or action == "Predict Price" or st.session_state.page == "Predict":
    st.markdown("## 🏠 House Price Prediction App")
    st.markdown("### 📊 Predict House & Land Price")

    col1, col2 = st.columns(2)
    with col1:
        area = st.number_input("Area in Square Feet", 100.0, 100000.0, 1000.0)
        bathrooms = st.number_input("Number of Bathrooms", 0, 20, 2)
        renovation_year = st.number_input("Renovation Year", 2000, 2030, 2020)
        road_type = st.selectbox("Road Type", ["Paved", "Dirt"])
        water_quality = st.selectbox("Water Supply Quality", ["Good", "Average", "Poor"])
        electricity = st.selectbox("Electricity Access", ["Good", "Average", "Poor"])
        crime_rate = st.selectbox("Crime Rate", ["Low", "Medium", "High"])
        traffic = st.selectbox("Traffic Level", ["Low", "Medium", "High"])
        green_space = st.selectbox("Green Space Nearby", ["Yes", "No"])
        proximity_schools = st.selectbox("Nearby Schools", ["Yes", "No"])
        proximity_mall = st.selectbox("Shopping Mall Nearby", ["Yes", "No"])
        flood_zone = st.selectbox("Flood Zone", ["Yes", "No"])
        house_type = st.selectbox("House Type", ["Detached", "Semi-Detached", "Apartment", "Bungalow"])

    with col2:
        bedrooms = st.number_input("Number of Bedrooms", 0, 20, 3)
        year_built = st.number_input("Year Built", 1900, 3000, 2000)
        location = st.selectbox("Location Category", ["Urban", "Suburban", "Rural"])
        property_type = st.selectbox("Property Type", ["Residential", "Commercial", "Agricultural"])
        neighborhood_rating = st.selectbox("Neighborhood Rating", ["Excellent", "Good", "Average", "Poor"])
        distance_to_city = st.number_input("Distance to City Center (km)", 0.0, 200.0, 10.0)
        lot_size = st.number_input("Lot Size (sqft)", 100.0, 100000.0, 500.0)
        garage = st.selectbox("Garage Availability", ["Yes", "No"])
        public_transport = st.selectbox("Public Transport Access", ["Yes", "No"])
        internet = st.selectbox("Internet Connectivity", ["Good", "Average", "Poor"])
        future_development = st.selectbox("Future Development Area", ["Yes", "No"])
        drainage = st.selectbox("Drainage Facility", ["Yes", "No"])
        slope = st.selectbox("Land Slope", ["Flat", "Moderate", "Steep"])

    if st.button("Predict Price"):
        with st.spinner("Loading model and predicting..."):
            try:
                model = joblib.load(MODEL_PATH)
                input_data = pd.DataFrame([{
                    "Area": area,
                    "Bedrooms": bedrooms,
                    "Bathrooms": bathrooms,
                    "YearBuilt": year_built,
                    "RenovationYear": renovation_year,
                    "Location": location,
                    "RoadType": road_type,
                    "PropertyType": property_type,
                    "WaterSupply": water_quality,
                    "Electricity": electricity,
                    "CrimeRate": crime_rate,
                    "Traffic": traffic,
                    "GreenSpace": green_space,
                    "ProximitySchools": proximity_schools,
                    "ProximityMall": proximity_mall,
                    "FloodZone": flood_zone,
                    "HouseType": house_type,
                    "NeighborhoodRating": neighborhood_rating,
                    "DistanceToCityCenter": distance_to_city,
                    "LotSize": lot_size,
                    "Garage": garage,
                    "PublicTransport": public_transport,
                    "Internet": internet,
                    "FutureDevelopment": future_development,
                    "Drainage": drainage,
                    "Slope": slope
                }])
                prediction = model.predict(input_data)[0]
                st.markdown(f"<h3 style='color:white;'>🏡 Estimated House Price: ₹ {prediction:,.2f}</h3>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
