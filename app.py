import streamlit as st
import pandas as pd
import joblib
import os
import bcrypt
from pathlib import Path

# Page configuration
st.set_page_config(page_title="House Price Predictor", layout="wide")

# --- Constants ---
MODEL_PATH = "model/house_price_model.pkl"
USER_CSV = "users.csv"

# --- Load model ---
model = joblib.load(MODEL_PATH)

# --- Ensure user DB exists ---
if not os.path.exists(USER_CSV):
    pd.DataFrame(columns=["username", "password"]).to_csv(USER_CSV, index=False)

# --- CSS Styling ---
st.markdown("""
 <style>
        .stApp {
            background-image: url('https://images.unsplash.com/photo-1568605114967-8130f3a36994');
            background-size: cover;
            background-attachment: fixed;
        }
        section[data-testid="stSidebar"] {
            background-color: #ffffff;
            color: #001F3F;
        }
        section[data-testid="stSidebar"] * {
            color: #001F3F !important;
        }
        .main > div {
            background-color: rgba(255, 255, 255, 0.85);
            border-radius: 15px;
            padding: 2rem;
            margin: 2rem;
        }
        .centered {
            display: flex;
            justify-content: center;
            text-align: center;
        }
        h1, h2, h3 {
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# --- Auth functions with bcrypt hashing ---
def signup(username, password):
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
    return bcrypt.checkpw(password.strip().encode('utf-8'), stored_hash.encode('utf-8'))

# --- Sidebar Navigation ---
st.sidebar.markdown("### 🏡 Navigation")
st.sidebar.markdown("Use this tool to estimate house prices")
action = st.sidebar.selectbox("Select Action", ["Login", "Signup", "Predict Price"])

# --- Session state ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# --- Login form ---
if action == "Login" and not st.session_state.logged_in:
    st.markdown("### 🏠 House Price Prediction App")
    st.markdown("## Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if login(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Welcome {username}!")
        else:
            st.error("Invalid credentials")

# --- Signup form ---
elif action == "Signup" and not st.session_state.logged_in:
    st.markdown("### 🏠 House Price Prediction App")
    st.markdown("## Signup")
    username = st.text_input("Choose Username")
    password = st.text_input("Choose Password", type="password")
    if st.button("Create Account"):
        if signup(username, password):
            st.success("Account created! Please login.")
        else:
            st.warning("Username already exists.")

# --- Prediction form ---
elif st.session_state.logged_in or action == "Predict Price":
    st.markdown(f"## 🏠 House Price Prediction App")
    if st.session_state.logged_in:
        st.success(f"✅ Welcome {st.session_state.username}!")
    st.markdown("### 📊 Predict House & Land Price")

    col1, col2 = st.columns(2)
    with col1:
        area = st.number_input("Area in Square Feet", min_value=100.0, max_value=100000.0, value=1000.0)
        bathrooms = st.number_input("Number of Bathrooms", min_value=0, max_value=20, value=2)
        renovation_year = st.number_input("Renovation Year", min_value=2000, max_value=2030, value=2020)
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
        bedrooms = st.number_input("Number of Bedrooms", min_value=0, max_value=20, value=3)
        year_built = st.number_input("Year Built", min_value=1900, max_value=3000, value=2000)
        location = st.selectbox("Location Category", ["Urban", "Suburban", "Rural"])
        property_type = st.selectbox("Property Type", ["Residential", "Commercial", "Agricultural"])
        neighborhood_rating = st.selectbox("Neighborhood Rating", ["Excellent", "Good", "Average", "Poor"])
        distance_to_city = st.number_input("Distance to City Center (km)", min_value=0.0, max_value=200.0, value=10.0)
        lot_size = st.number_input("Lot Size (sqft)", min_value=100.0, max_value=100000.0, value=500.0)
        garage = st.selectbox("Garage Availability", ["Yes", "No"])
        public_transport = st.selectbox("Public Transport Access", ["Yes", "No"])
        internet = st.selectbox("Internet Connectivity", ["Good", "Average", "Poor"])
        future_development = st.selectbox("Future Development Area", ["Yes", "No"])
        drainage = st.selectbox("Drainage Facility", ["Yes", "No"])
        slope = st.selectbox("Land Slope", ["Flat", "Moderate", "Steep"])

    # Predict
    if st.button("Predict Price"):
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

        try:
            prediction = model.predict(input_data)[0]
            st.success(f"🏡 Estimated House Price: ₹ {prediction:,.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
