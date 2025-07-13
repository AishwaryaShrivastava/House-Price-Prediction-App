import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Seed for reproducibility
np.random.seed(42)

# Generate synthetic dataset
def generate_data(n=10000):
    data = pd.DataFrame({
        "Area": np.random.uniform(500, 100000, n),
        "Bedrooms": np.random.randint(1, 6, n),
        "Bathrooms": np.random.randint(1, 4, n),
        "YearBuilt": np.random.randint(1950, 2025, n),
        "RenovationYear": np.random.choice([2000, 2010, 2020, 2023, 2024, 2025], n),
        "Location": np.random.choice(["Urban", "Suburban", "Rural"], n),
        "RoadType": np.random.choice(["Paved", "Dirt"], n),
        "PropertyType": np.random.choice(["Residential", "Commercial", "Agricultural"], n),
        "WaterSupply": np.random.choice(["Good", "Average", "Poor"], n),
        "Electricity": np.random.choice(["Good", "Average", "Poor"], n),
        "CrimeRate": np.random.choice(["Low", "Medium", "High"], n),
        "Traffic": np.random.choice(["Low", "Medium", "High"], n),
        "GreenSpace": np.random.choice(["Yes", "No"], n),
        "ProximitySchools": np.random.choice(["Yes", "No"], n),
        "ProximityMall": np.random.choice(["Yes", "No"], n),
        "FloodZone": np.random.choice(["Yes", "No"], n),
        "HouseType": np.random.choice(["Detached", "Semi-Detached", "Apartment", "Bungalow"], n),
        "NeighborhoodRating": np.random.choice(["Excellent", "Good", "Average", "Poor"], n),
        "DistanceToCityCenter": np.random.uniform(0, 50, n),
        "LotSize": np.random.uniform(500, 20000, n),
        "Garage": np.random.choice(["Yes", "No"], n),
        "PublicTransport": np.random.choice(["Yes", "No"], n),
        "Internet": np.random.choice(["Good", "Average", "Poor"], n),
        "FutureDevelopment": np.random.choice(["Yes", "No"], n),
        "Drainage": np.random.choice(["Yes", "No"], n),
        "Slope": np.random.choice(["Flat", "Moderate", "Steep"], n),
    })

    score_map = {
        "Poor": 0.9, "Average": 1.0, "Good": 1.1, "Excellent": 1.3,
        "Low": 1.2, "Medium": 1.0, "High": 0.8,
        "Yes": 1.2, "No": 0.9
    }

    base_price = (
        data["Area"] * 1500 +
        data["Bedrooms"] * 500000 +
        data["Bathrooms"] * 300000 +
        data["LotSize"] * 50
    )

    modifiers = (
        data["WaterSupply"].map(score_map) *
        data["Electricity"].map(score_map) *
        data["CrimeRate"].map(score_map) *
        data["Traffic"].map(score_map) *
        data["GreenSpace"].map(score_map) *
        data["ProximitySchools"].map(score_map) *
        data["ProximityMall"].map(score_map) *
        data["FloodZone"].map(score_map) *
        data["Garage"].map(score_map) *
        data["PublicTransport"].map(score_map) *
        data["Internet"].map(score_map) *
        data["FutureDevelopment"].map(score_map) *
        data["Drainage"].map(score_map) *
        data["NeighborhoodRating"].map(score_map)
    )

    noise = np.random.normal(0, 100000, n)
    data["Price"] = base_price * modifiers + noise
    return data

# Load and split data
data = generate_data()
X = data.drop("Price", axis=1)
y = data["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Categorical columns
categorical_cols = X.select_dtypes(include="object").columns.tolist()
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
], remainder="passthrough")

# Models to compare
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42)
}

mse_scores = {}
r2_scores = {}
predictions = {}

# Train and evaluate models
for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("regressor", model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    predictions[name] = y_pred
    mse_scores[name] = mean_squared_error(y_test, y_pred)
    r2_scores[name] = r2_score(y_test, y_pred)

    if name == "Random Forest":
        # Save only the best model (Random Forest)
        os.makedirs("model", exist_ok=True)
        joblib.dump(pipeline, "model/house_price_model.pkl")

# Plot MSE
plt.figure(figsize=(10, 5))
sns.barplot(x=list(mse_scores.keys()), y=list(mse_scores.values()), palette="Blues_d")
plt.title("📊 Mean Squared Error Comparison")
plt.ylabel("MSE")
plt.tight_layout()
plt.show()

# Plot R²
plt.figure(figsize=(10, 5))
sns.barplot(x=list(r2_scores.keys()), y=list(r2_scores.values()), palette="Greens_d")
plt.title("📈 R² Score Comparison")
plt.ylabel("R²")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# Line plot: actual vs predicted
plt.figure(figsize=(12, 6))
sample = y_test[:100].reset_index(drop=True)
for name in models:
    plt.plot(predictions[name][:100], label=f"{name} Prediction", linestyle='--')
plt.plot(sample.values, label="Actual", linewidth=2, color="black")
plt.title("📈 Actual vs Predicted Prices (First 100 Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.show()

# Heatmap
plt.figure(figsize=(14, 10))
numerical_data = data.select_dtypes(include=[np.number])
corr_matrix = numerical_data.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("🔥 Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# Final performance summary
performance_df = pd.DataFrame({
    "Model": list(models.keys()),
    "MSE": list(mse_scores.values()),
    "R² Score": list(r2_scores.values())
})

print("\n✅ Model training complete and best model (Random Forest) saved.")
print("📊 Model Performance Summary:")
print(performance_df.to_string(index=False))
