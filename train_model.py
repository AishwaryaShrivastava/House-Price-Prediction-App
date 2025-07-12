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
        "Area in Square Feet": np.random.randint(500, 10000, n),
        "Number of Bedrooms": np.random.randint(1, 6, n),
        "Number of Bathrooms": np.random.randint(1, 4, n),
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

    data["Price"] = (
        data["Area in Square Feet"] * 500 +
        data["Number of Bedrooms"] * 300000 +
        data["Number of Bathrooms"] * 200000 +
        np.random.normal(0, 100000, n)
    )

    return data

# Generate data
data = generate_data()
X = data.drop("Price", axis=1)
y = data["Price"]

# Identify categorical features
categorical_features = X.select_dtypes(include='object').columns.tolist()

# Preprocessing and modeling pipeline
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
], remainder='passthrough')

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train model
pipeline.fit(X, y)

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(pipeline, "model/house_price_model.pkl")
print("✅ Model trained and saved!")
