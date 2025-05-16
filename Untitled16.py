#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load and clean data
df = pd.read_csv("E Commerce Dataset.csv")
df.drop(columns=["Unnamed: 0", "CustomerID"], inplace=True, errors='ignore')
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
df["CityTier"] = df["CityTier"].map({1: "Tier 1", 2: "Tier 2", 3: "Tier 3"}).astype("category")
df["CashbackAmount"] = df["CashbackAmount"].replace({"\\$": "", ",": ""}, regex=True)
df["CashbackAmount"] = pd.to_numeric(df["CashbackAmount"], errors='coerce').fillna(0).astype(float)
df["OrderCount"] = pd.to_numeric(df["OrderCount"], errors='coerce').fillna(0).astype(int)

# Prepare features and target
X = df[[col for col in df.columns if col != "Churn"]]
X = pd.get_dummies(X, drop_first=True)
X.fillna(0, inplace=True)
y = df["Churn"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Save model and scaler
joblib.dump(model, "churn_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler have been trained and saved successfully!")


# In[ ]:




