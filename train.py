# train.py
import os
import pandas as pd
from preprocessing import preprocess_dataframe
from model import train_models

# Ensure models and assets directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("assets", exist_ok=True)

# Load raw dataset
df_raw = pd.read_csv('chatgpt_reviews - chatgpt_reviews.csv')  # <-- Change to your actual dataset path

# Preprocess the dataset
df = preprocess_dataframe(df_raw)

# Train models and save them
train_models(df)

print("✅ Training complete. Models saved to /models and plots saved to /assets.")
