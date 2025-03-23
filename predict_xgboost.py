"""
input_data_in_csv = pd.DataFrame({
    "nco2": [1, 1],
    "nh2o": [2, 0],
    "y_init": [0.612635, -0.420629],
    "natoms": [82, 95],
    "defective": [True, False],  # Or [1, 0]
    "cell_mean": [5.643653, 5.373723],
    "cell_std": [4.755599, 5.147071],
    "pos_relaxed_mean": [8.48211, 7.654403],
    "pos_relaxed_std": [4.277549, 3.760834],
    "atomic_numbers_mean": [13.573171, 6.905263],
    "atomic_numbers_std": [12.533885, 12.555620]
})"""


import pandas as pd
import xgboost as xgb

# Load the saved model
loaded_model = xgb.Booster()
loaded_model.load_model("xgboost_nads_model.json")
print("Model loaded successfully from 'xgboost_nads_model.json'")

# Load or prepare your input data (example shown below)
# Replace this with your actual new data file or DataFrame
input_data = pd.read_csv("your_new_data.csv")  # Or create DataFrame manually

# Define the feature set (must match the training features)
features = [
    "nco2", "nh2o", "y_init", "natoms", "defective",
    "cell_mean", "cell_std", "pos_relaxed_mean", "pos_relaxed_std",
    "atomic_numbers_mean", "atomic_numbers_std"
]

# Select and preprocess the features
X = input_data[features]
X["defective"] = X["defective"].astype(int)  # Convert 'defective' to 0/1 (True -> 1, False -> 0)

# Convert to DMatrix for XGBoost prediction
dmat = xgb.DMatrix(X)

# Make predictions
predictions = loaded_model.predict(dmat)

# Add predictions to the DataFrame (optional)
input_data["predicted_nads"] = predictions

# Display some results
print("\nPredictions for the first 5 rows:")
print(input_data[features + ["predicted_nads"]].head())

# Optionally, save the results to a new CSV
input_data.to_csv("predictions_output.csv", index=False)
print("\nResults saved to 'predictions_output.csv'")
