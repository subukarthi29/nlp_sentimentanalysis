import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

sns.set(style="whitegrid")

def plot_rating_distribution(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(x='rating', data=df, palette='Set2')
    plt.title("Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("assets/rating_distribution.png")
    plt.close()

def plot_wordclouds(df):
    positive_text = " ".join(df[df['sentiment'] == 'Positive']['cleaned_review'])
    negative_text = " ".join(df[df['sentiment'] == 'Negative']['cleaned_review'])

    positive_wc = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
    negative_wc = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(negative_text)

    positive_wc.to_file("assets/positive_wc.png")
    negative_wc.to_file("assets/negative_wc.png")

def plot_avg_rating_over_time(df):
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    time_data = df.groupby(df['date'].dt.to_period('M')).mean(numeric_only=True)['rating']
    time_data.index = time_data.index.to_timestamp()
    
    plt.figure(figsize=(10, 4))
    sns.lineplot(x=time_data.index, y=time_data.values, marker='o')
    plt.title("Average Rating Over Time")
    plt.ylabel("Average Rating")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("assets/avg_rating_time.png")
    plt.close()

def plot_platform_rating_comparison(df):
    plt.figure(figsize=(6, 4))
    sns.barplot(x='platform', y='rating', data=df, ci=None, palette='pastel')
    plt.title("Average Rating by Platform")
    plt.tight_layout()
    plt.savefig("assets/platform_rating.png")
    plt.close()

def plot_verified_user_comparison(df):
    plt.figure(figsize=(6, 4))
    sns.barplot(x='verified_purchase', y='rating', data=df, ci=None, palette='muted')
    plt.title("Verified vs Non-Verified Rating")
    plt.tight_layout()
    plt.savefig("assets/verified_comparison.png")
    plt.close()

def plot_review_length_by_rating(df):
    df['review_length'] = df['review'].apply(lambda x: len(x) if isinstance(x, str) else 0)
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='rating', y='review_length', data=df, palette='coolwarm')
    plt.title("Review Length by Rating")
    plt.tight_layout()
    plt.savefig("assets/review_length_boxplot.png")
    plt.close()

if __name__ == "__main__":
    import os
    from preprocessing import preprocess_dataframe

    # Ensure 'assets/' directory exists
    if not os.path.exists("assets"):
        os.makedirs("assets")

    # Load data
    df = pd.read_csv('chatgpt_reviews - chatgpt_reviews.csv')

    # Preprocess data
    df = preprocess_dataframe(df)

    # Run EDA plots
    plot_rating_distribution(df)
    plot_wordclouds(df)
    plot_avg_rating_over_time(df)
    plot_platform_rating_comparison(df)
    plot_verified_user_comparison(df)
    plot_review_length_by_rating(df)

    print("All plots saved in the 'assets/' folder.")


