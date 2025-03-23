import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt

# Load the dataset from the specified study
data = pd.read_csv('dataset/all_data.csv')  # Replace with actual path after downloading from https://1drv.ms/u/s!AtuVqcWZi8aAy11S2wxataTe8IMH

# Define feature columns based on your input (structural and chemical properties of MOFs)
feature_cols = [
    'motif_benzene', 'motif_PO3', 'motif_N2', 'motif_CC', 'motif_NO2', 'motif_NH2', 'motif_OH',
    'motif_aliphatic ether', 'motif_OPO', 'motif_diphenyl', 'motif_phenanthryl', 'motif_naphthyl',
    'motif_OCNCO', 'motif_aliphatic CCl', 'motif_aliphatic CBr', 'motif_aliphatic CF',
    'motif_aromatic CCl', 'motif_aromatic CBr', 'motif_aromatic CF', 'motif_PhO', 'motif_amide',
    'motif_furan', 'motif_triazine', 'motif_tetrazine', 'motif_trans-butadiene', 'motif_cis-butadiene',
    'motif_aliphatic ester', 'motif_pyrrole', 'motif_thiophene', 'motif_alkyne', 'motif_bicyclooctane',
    'motif_pyridine', 'motif_pyrazine', 
    *[f'epsilon bin {i:03d}' for i in range(56)] + [f'epsilon bin {i}' for i in range(100, 556, 10) if i <= 555],
    *[f'sigma bin {i:03d}' for i in range(56)] + [f'sigma bin {i}' for i in range(100, 556, 10) if i <= 555],
    *[f'RDF_electronegativity_{r:.2f}' for r in np.arange(2.0, 30.01, 0.1)],
    *[f'RDF_hardness_{r:.2f}' for r in np.arange(2.0, 30.01, 0.1)],
    *[f'RDF_vdWaalsVolume_{r:.2f}' for r in np.arange(2.0, 30.01, 0.1)],
    'CO2_Surf_m2/g', 'CO2_VFrac', 'Pore_1', 'CO2_Surf_m2/cm3', 'dense', 'Pore_3', 'wc'
]

# Define target column (CO₂ adsorption capacity, adjust based on dataset)
target_col = 'CO2_Surf_m2/g'  # Common name; replace with actual column name (e.g., 'CO2 Uptake (mmol/g)') after inspecting dataset

# Check available columns in the dataset
available_cols = data.columns.tolist()
print(f"Available columns in dataset: {available_cols}")

# Filter feature columns to those present in the dataset
missing_cols = [col for col in feature_cols if col not in data.columns]
if missing_cols:
    print(f"Warning: These columns are missing from the dataset: {missing_cols}")
    feature_cols = [col for col in feature_cols if col in data.columns]

# Add common MOF features if present (based on typical datasets)
common_mof_features = ['surface_area_m2g', 'void_fraction', 'pore_volume_cm3g', 'density_gcm3']
for feat in common_mof_features:
    if feat in data.columns and feat not in feature_cols:
        feature_cols.append(feat)

# Extract features (X) and target (y)
X = data[feature_cols]
y = data[target_col]

# Handle missing values (fill with mean for numeric columns)
X = X.fillna(X.mean(numeric_only=True))

# Check for categorical variables (encode if present)
if X.select_dtypes(include=['object']).columns.any():
    X = pd.get_dummies(X, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features (recommended for XGBoost)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert data to DMatrix format for XGBoost
dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
dtest = xgb.DMatrix(X_test_scaled, label=y_test)

# Define XGBoost parameters with GPU support
params = {
    'objective': 'reg:squarederror',  # Regression for CO₂ uptake prediction
    'tree_method': 'hist',           # Histogram-based method for GPU
    'device': 'cuda',                # GPU acceleration
    'max_depth': 6,                  # Tree depth
    'eta': 0.1,                      # Learning rate
    'subsample': 0.8,                # Prevent overfitting
    'colsample_bytree': 0.8,         # Feature sampling
    'eval_metric': 'rmse'            # Root Mean Squared Error for evaluation
}

# Train the model on GPU
bst = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtest, 'test')], early_stopping_rounds=10, verbose_eval=10)

# Make predictions on the test set
y_pred = bst.predict(dtest)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse:.4f}')
print(f'R² Score: {r2:.4f}')

# Feature Importance Analysis
feature_importance = bst.get_score(importance_type='weight')
feature_importance_df = pd.DataFrame({
    'Feature': [X.columns[int(f[1:])] for f in feature_importance.keys()],
    'Importance': list(feature_importance.values())
})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display top 10 most important features
print("\nTop 10 Most Important Features for CO₂ Adsorption Capacity:")
print(feature_importance_df.head(10))

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'].head(10), feature_importance_df['Importance'].head(10))
plt.xlabel('Importance')
plt.title('Top 10 Feature Importances for CO₂ Adsorption Capacity Prediction')
plt.gca().invert_yaxis()
#
plt.savefig('result.png')
plt.show()