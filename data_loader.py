import pandas as pd

def load_data_from_csv(file_path):
    data = pd.read_csv(file_path)
    return data

csv_file_path = 'chatgpt_reviews - chatgpt_reviews.csv'
df = load_data_from_csv(csv_file_path)

