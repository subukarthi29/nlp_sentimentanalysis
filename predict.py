import joblib
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Load NLTK components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\W+', ' ', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def load_model_and_vectorizer(model_name='LogisticRegression'):
    model = joblib.load(f'models/{model_name}.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    return model, vectorizer

def predict_sentiment(review_text, model_name='LogisticRegression'):
    cleaned = clean_text(review_text)
    model, vectorizer = load_model_and_vectorizer(model_name)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    return prediction
