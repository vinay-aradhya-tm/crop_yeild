# app.py
import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

@st.cache_resource
def load_artifacts():
    # Load dataset
    df = pd.read_csv("Crop_recommendation.csv")
    X = df.drop("label", axis=1)
    y = df["label"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Train pipeline on *your* machine’s sklearn version
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(random_state=42))
    ])
    pipeline.fit(X, y_enc)

    return pipeline, le

pipeline, le = load_artifacts()

st.title("🌱 Crop Recommendation")
st.write("Enter soil and climate values:")

# Inputs
features = {}
for col in ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]:
    features[col] = st.number_input(col, value=0.0)

if st.button("Predict"):
    df_input = pd.DataFrame([features])
    pred = pipeline.predict(df_input)[0]
    crop = le.inverse_transform([pred])[0]
    st.success(f"Recommended crop: **{crop}**")
