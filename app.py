import streamlit as st
import pandas as pd
from preprocessing import preprocess_dataframe
from data_loader import load_data_from_csv
from predict import predict_sentiment
import os
import nltk

nltk.download('wordnet')

st.set_page_config(page_title="AI Echo – Sentiment Analyzer", layout="wide")

st.title("🤖 AI Echo: Your Smartest Conversational Partner")
st.markdown("A sentiment analysis tool for ChatGPT user reviews.")

# Load and cache dataset
@st.cache_data
def get_cleaned_data():
    df_raw = load_data_from_csv('chatgpt_reviews - chatgpt_reviews.csv')
    df = preprocess_dataframe(df_raw)
    return df

df = get_cleaned_data()

# Sidebar Navigation
tabs = ["📊 Dashboard", "💬 Review Analyzer", "📈 Model Metrics"]
choice = st.sidebar.radio("Go to", tabs)

# 📊 Dashboard Tab
if choice == "📊 Dashboard":
    st.header("Review Insights & Visualizations")

    col1, col2 = st.columns(2)
    with col1:
        st.image("assets/rating_distribution.png", caption="Rating Distribution")
        st.image("assets/positive_wc.png", caption="Positive Review Word Cloud")
    with col2:
        st.image("assets/negative_wc.png", caption="Negative Review Word Cloud")
        st.image("assets/avg_rating_time.png", caption="Avg Rating Over Time")

    st.image("assets/platform_rating.png", caption="Rating by Platform")
    st.image("assets/verified_comparison.png", caption="Verified vs Non-Verified Ratings")
    st.image("assets/review_length_boxplot.png", caption="Review Length by Rating")

# 💬 Review Analyzer Tab
elif choice == "💬 Review Analyzer":
    st.header("🔍 Analyze a User Review")
    user_input = st.text_area("Paste a review below:", height=150)
    model_option = st.selectbox("Choose a model", ["LogisticRegression", "NaiveBayes", "RandomForest"])

    if st.button("Predict Sentiment"):
        if user_input.strip():
            sentiment = predict_sentiment(user_input, model_option)
            st.success(f"Predicted Sentiment: **{sentiment}**")
        else:
            st.warning("Please enter a review.")

# 📈 Model Metrics Tab
elif choice == "📈 Model Metrics":
    st.header("📉 Model Evaluation Results")
    model_names = ["LogisticRegression", "NaiveBayes", "RandomForest"]

    for model_name in model_names:
        cm_path = f"assets/cm_{model_name}.png"
        if os.path.exists(cm_path):
            st.subheader(f"{model_name}")
            st.image(cm_path, caption=f"{model_name} – Confusion Matrix")
        else:
            st.warning(f"Confusion matrix not found for {model_name}. Please train the model first.")
