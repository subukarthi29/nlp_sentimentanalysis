import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if pd.isnull(text):
        return ""
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters and numbers
    text = re.sub(r'\W+', ' ', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def preprocess_dataframe(df):
    # Drop missing reviews
    df = df.dropna(subset=['review'])
    
    # Apply cleaning
    df['cleaned_review'] = df['review'].apply(clean_text)
    
    # Label sentiment
    def label_sentiment(rating):
        if rating >= 4:
            return 'Positive'
        elif rating == 3:
            return 'Neutral'
        else:
            return 'Negative'
    
    df['sentiment'] = df['rating'].apply(label_sentiment)
    
    return df[['date', 'review', 'cleaned_review', 'rating', 'sentiment', 'platform', 'version', 'verified_purchase', 'location']]
