import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.preprocessing import StandardScaler, PolynomialFeatures # type: ignore
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess a small sample of data for quick testing
df = pd.read_csv('nearest-earth-objects(1910-2024).csv')
df_sample = df.sample(n=1000, random_state=42)  # Use only 1000 samples for quick training

# Basic preprocessing
df_sample = df_sample.drop(['neo_id', 'name', 'orbiting_body'], axis=1)
df_sample = df_sample.dropna()

# Feature engineering
df_sample['estimated_diameter_mean'] = (df_sample['estimated_diameter_min'] + df_sample['estimated_diameter_max']) / 2

# Prepare features and target
X = df_sample.drop('is_hazardous', axis=1)
y = df_sample['is_hazardous']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
models = {}

# Random Forest
rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
rf_model.fit(X_train_scaled, y_train)
models['Random Forest'] = rf_model

# Logistic Regression
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_scaled, y_train)
models['Logistic Regression'] = lr_model

# Create polynomial features transformer (identity for compatibility)
poly = PolynomialFeatures(degree=1, include_bias=False)
poly.fit(X_train)

# Save models
with open('models.pkl', 'wb') as f:
    pickle.dump(models, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('poly.pkl', 'wb') as f:
    pickle.dump(poly, f)

print("Quick models trained and saved successfully!")
print(f"Random Forest accuracy: {rf_model.score(X_test_scaled, y_test):.3f}")
print(f"Logistic Regression accuracy: {lr_model.score(X_test_scaled, y_test):.3f}")

