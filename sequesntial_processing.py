# import streamlit as st

# col1, col2, col3, col4 = st.columns(4)

# for i in range(4):
#     with eval(f"col{i+1}"):
#         st.header("A cat")
#         st.content_input(f"jdfgjd{i+1}")

#--------------------------
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
