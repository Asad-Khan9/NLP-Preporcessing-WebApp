import time
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords

# nltk.download('stopwords')
# nltk.download('punkt')
start = time.time()
def remove_stopwords(content):
    if isinstance(content, str):
        tokens = nltk.word_tokenize(content)
        stopwords_set = set(stopwords.words("english"))
        filtered = [token for token in tokens if token.lower() not in stopwords_set]
        return ' '.join(filtered)

def process_column(df, column_name):
    df[column_name] = df[column_name].apply(remove_stopwords)
    return df

data_path = "dataset\\amazon_reviews.csv"

df = pd.read_csv(data_path)
df = process_column(df, 'content')

print(df.head())
end = time.time()

print(end - start)
