import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def train_models(df):
    X = df['cleaned_review']
    y = df['sentiment']

    tfidf = TfidfVectorizer(max_features=5000)
    X_vec = tfidf.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    # Model dictionary
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100),
        "NaiveBayes": MultinomialNB()
    }

    reports = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"\n📊 {name} Classification Report:")
        report = classification_report(y_test, y_pred, output_dict=True)
        reports[name] = report
        print(classification_report(y_test, y_pred))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=["Positive", "Neutral", "Negative"])
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=["Positive", "Neutral", "Negative"], yticklabels=["Positive", "Neutral", "Negative"])
        plt.title(f'{name} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f'assets/cm_{name}.png')
        plt.close()

        # Save model and vectorizer
        joblib.dump(model, f"models/{name}.pkl")

    joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
    return reports
