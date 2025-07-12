import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

np.random.seed(42)

def generate_data(n=10000):
    data = pd.DataFrame({
        "Area in Square Feet": np.random.randint(1000, 100000, n),
        "Number of Bedrooms": np.random.randint(1, 7, n),
        "Number of Bathrooms": np.random.randint(1, 5, n),
        "Year Built": np.random.randint(2000, 2024, n),
        "Renovation Year": np.random.choice([2020, 2021, 2022, 2023, 2024, 2025], n),
        "Location Category": np.random.choice(["Urban", "Suburban", "Rural"], n),
        "Road Type": np.random.choice(["Dirt", "Paved"], n),
        "Zoning Type": np.random.choice(["Residential", "Commercial", "Agricultural", "Industrial"], n),
        "Water Supply Quality": np.random.choice(["Poor", "Average", "Good", "Excellent"], n),
        "Electricity Reliability": np.random.choice(["Poor", "Average", "Good", "Excellent"], n),
        "Internet Speed": np.random.choice(["Slow", "Moderate", "Fast", "Very Fast"], n),
        "Greenery Score": np.random.choice(["Poor", "Average", "Good", "Excellent"], n),
        "Pollution Index": np.random.choice(["High", "Medium", "Low"], n),
        "Land Slope": np.random.choice(["Flat", "Moderate", "Steep"], n),
        "Soil Quality Index": np.random.choice(["Poor", "Average", "Good"], n),
        "Earthquake Resistance": np.random.choice(["No", "Partial", "Yes"], n),
        "Public Transport Availability": np.random.choice(["None", "Some", "Frequent"], n),
    })

    quality_map = {
        "Poor": 0.9, "Average": 1.0, "Good": 1.1, "Excellent": 1.3,
        "High": 0.9, "Medium": 1.0, "Low": 1.1,
        "Slow": 0.9, "Moderate": 1.0, "Fast": 1.1, "Very Fast": 1.3,
        "No": 0.9, "Partial": 1.0, "Yes": 1.2,
        "None": 0.9, "Some": 1.0, "Frequent": 1.2
    }

    base_price = (
        data["Area in Square Feet"] * 2000 +
        data["Number of Bedrooms"] * 700000 +
        data["Number of Bathrooms"] * 500000
    )

    modifiers = (
        data["Water Supply Quality"].map(quality_map) *
        data["Electricity Reliability"].map(quality_map) *
        data["Internet Speed"].map(quality_map) *
        data["Greenery Score"].map(quality_map) *
        data["Pollution Index"].map(quality_map) *
        data["Earthquake Resistance"].map(quality_map) *
        data["Public Transport Availability"].map(quality_map)
    )

    noise = np.random.normal(0, 500000, n)
    data["Price"] = (base_price * modifiers) + noise
    return data

# Generate and split data
data = generate_data()
X = data.drop("Price", axis=1)
y = data["Price"]

# Preprocessing
categorical_features = X.select_dtypes(include='object').columns.tolist()
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
], remainder='passthrough')

# Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42))
])

# Train model
pipeline.fit(X, y)

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(pipeline, "model/house_price_model.pkl")

print("✅ Model trained and saved successfully.")
