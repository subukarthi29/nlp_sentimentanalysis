# AI Echo: Your Smartest Conversational Partner

A complete **NLP-powered sentiment analysis dashboard** built to analyze user reviews of ChatGPT. This project classifies reviews as **Positive**, **Neutral**, or **Negative** and helps understand customer satisfaction trends using interactive visualizations and machine learning models.

---

##  Features

- Load reviews directly from **Google Sheets**
- Preprocess text (cleaning, stopword removal, lemmatization)
- EDA Visualizations: Rating trends, word clouds, platform analysis
- ML Models: Logistic Regression, Naive Bayes, Random Forest
- Model Evaluation: Accuracy, F1-Score, Confusion Matrix
- Real-time sentiment prediction for user input
- Streamlit Dashboard for interaction and insights

---

## Project Structure

```bash
AI-Echo-Sentiment-Analysis/
├── app.py                 # Streamlit app
├── data_loader.py         # Load Google Sheets data
├── preprocessing.py       # Text cleaning & labeling
├── eda.py                 # Generate plots
├── model.py               # Train & evaluate models
├── predict.py             # Predict new review sentiment
├── requirements.txt       # Python dependencies
├── models/                # Trained models + TF-IDF vectorizer
│   ├── LogisticRegression.pkl
│   ├── NaiveBayes.pkl
│   ├── RandomForest.pkl
│   └── tfidf_vectorizer.pkl
└── assets/                # All visualizations (PNG)
    ├── rating_distribution.png
    ├── positive_wc.png
    ├── negative_wc.png
    ├── avg_rating_time.png
    ├── platform_rating.png
    ├── verified_comparison.png
    ├── review_length_boxplot.png
    ├── cm_LogisticRegression.png
    ├── cm_NaiveBayes.png
    └── cm_RandomForest.png
```

## **Install dependencies**

pip install -r requirements.txt

**Make sure you also download necessary NLTK data:**
```bash
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## Train Models
**Train Traditional Models (TF-IDF + ML)**
```bash
python model.py
```
## Run the App
```bash
streamlit run app.py
```

## Dashboard Preview
- **Dashboard Tab:** EDA with visual insights from the dataset

- **Review Analyzer Tab:** Predict sentiment from user-input review text

- **Model Metrics Tab:** View confusion matrices and model evaluation results

## Dataset Description

This project uses a dataset of user reviews of ChatGPT with the following features:

| Column Name        | Description                                                 |
|--------------------|-------------------------------------------------------------|
| `review`           | User-submitted review text                                  |
| `rating`           | Numerical score from 1 to 5                                 |
| `platform`         | Platform used: Web / Mobile                                 |
| `location`         | User's country                                              |
| `verified_purchase`| Whether the user is a verified/paying subscriber            |
| `version`          | ChatGPT version (e.g., 3.5, 4.0)                            |
| `date`             | Date when the review was submitted                          |
| `title`            | Short summary of the review                                 |
| `helpful_votes`    | Number of users who found the review helpful                |
| `review_length`    | Character count of the review                               |
| `language`         | Language code of the review (e.g., en, es)                  |
| `username`         | Anonymized reviewer name (used for identifying repeat users)|

Dataset Link: [Google Sheets – ChatGPT Reviews](https://docs.google.com/spreadsheets/d/1-4CMrIp98PsaNBZ5ioy86Arz2eACmvXPWMl7xizW7Q0/edit#gid=1667878447)

## ML Models Used

- Logistic Regression

- Naive Bayes

- Random Forest

- TF-IDF Vectorization for feature extraction

## Evaluation Metrics
- Accuracy, Precision, Recall, F1-score

- Confusion Matrix (visualized)
