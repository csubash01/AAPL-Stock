import os
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR,"model.pkl")
PRICE_PATH = os.path.join(BASE_DIR, "data", "AAPL_price.csv")
NEWS_PATH = os.path.join(BASE_DIR, "data", "AAPL_news_rss.csv")

# Load model
try:
    model, features = joblib.load(MODEL_PATH)
except Exception:
    st.error("⚠️ Could not load model.pkl. Train and save it first.")
    st.stop()

# Optional: load raw data
try:
    price = pd.read_csv(PRICE_PATH, encoding="latin1")
    news = pd.read_csv(NEWS_PATH, encoding="latin1")
except Exception:
    st.warning("⚠️ Could not load CSV files. App will still run with manual input.")

st.title("📈 Apple Stock Sentiment Prediction App")
st.write("Predict if AAPL stock will go **Up (1)** or **Down (0)** using daily sentiment features.")

# Inputs (only sentiment features)
sent_mean = st.number_input("Sentiment Mean (VADER compound)", value=0.0, format="%.5f")
sent_count = st.number_input("Sentiment Count (number of news)", value=0, step=1)
sent_next = st.number_input("Previous-day Sentiment (lag feature)", value=0.0, format="%.5f")

if st.button("Predict"):
    X_new = pd.DataFrame([{
        "sent_mean": sent_mean,
        "sent_count": sent_count,
        "sent_next": sent_next
    }])

    try:
        pred = model.predict(X_new)[0]
        proba = model.predict_proba(X_new)[0][pred]
        st.success(f"📊 Prediction: {'UP 📈' if pred == 1 else 'DOWN 📉'} (Confidence: {proba:.2f})")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
