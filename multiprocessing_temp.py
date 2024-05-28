import multiprocessing
import pandas as pd
from multiprocessing import Pool
import nltk
import re
import time

def process_chunk(df_chunk):
    """
    Function to apply transformations/operations to the 'content' column of each chunk.
    """
    import nltk
    from nltk.corpus import stopwords
    # nltk.download('stopwords')
    # nltk.download('punkt')
    
    def remove_stopwords(content):
        if isinstance(content, str):
            tokens = nltk.word_tokenize(content)
            stopwords_set = set(stopwords.words("english"))
            filtered = [token for token in tokens if token.lower() not in stopwords_set]
            return ' '.join(filtered)
    
    df_chunk['content'] = df_chunk['content'].apply(remove_stopwords)
    return df_chunk

if __name__ == '__main__':
    
    data_path = "dataset\\amazon_reviews.csv"

    df = pd.read_csv(data_path)

    num_partitions = 16
    chunks = [df.iloc[df.shape[0] // num_partitions * i: df.shape[0] // num_partitions * (i + 1)] for i in range(num_partitions)]     
    
    start = time.time()
    
    pool = Pool(processes=num_partitions)

    processed_chunks = pool.map(process_chunk, chunks)

    pool.close()
    pool.join()
    processed_df = pd.concat(processed_chunks)

    print(processed_df["content"].head())
    end = time.time()
    print(f"Processing time: {end - start} seconds")
