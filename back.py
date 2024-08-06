import pandas as pd
import nltk
import contractions as contr
from nltk.stem import WordNetLemmatizer

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download("wordnet")
# nltk.download("omw-1.4")

#---------sequesntial processing-----------
def reutrn_columns(df):
    return(df.columns)

def toLower0(df, column):
    df[column] = df[column].str.lower()
    return df

def remove_punctuation(text):
    if isinstance(text, str):
        tokenizer = nltk.RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(text)
        return ' '.join(tokens)
    else:
        return text
    
def remove_stopwords(text):

    if isinstance(text, str):
        
        stopwords_set = set(nltk.corpus.stopwords.words("english"))
        tokens = nltk.word_tokenize(text)
        stopwords = nltk.corpus.stopwords.words("english")
        filtered = [token for token in tokens if token.lower() not in stopwords]
        return ' '.join(filtered)

def expand_contractions(text):
    if isinstance(text, str):
        expanded_words = [] 
        for word in text.split():
            expanded_words.append(contr.fix(word))  
        
        return ' '.join(expanded_words)

def lemmatize(text):
    wnl = WordNetLemmatizer()
    if isinstance(text, str):
        text_tokens = nltk.word_tokenize(text)
        lemmatized_tokens = [wnl.lemmatize(word, pos="v") for word in text_tokens]
        return ' '.join(lemmatized_tokens)


#---------------------multiprocessing----------------
def to_lower(text):
    if isinstance(text, str):
        return text.lower()
    return text

def remove_punctuation(text):
    if isinstance(text, str):
        tokenizer = nltk.RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(text)
        return ' '.join(tokens)
    return text

def remove_stopwords_from_text(text, stopwords_set):
    if isinstance(text, str):
        tokens = nltk.word_tokenize(text)
        filtered = [token for token in tokens if token.lower() not in stopwords_set]
        return ' '.join(filtered)
    return text

def expand_contractions(text):
    if isinstance(text, str):
        expanded_words = [contr.fix(word) for word in text.split()]
        return ' '.join(expanded_words)
    return text

def lemmatize(text):
    wnl = WordNetLemmatizer()
    if isinstance(text, str):
        text_tokens = nltk.word_tokenize(text)
        lemmatized_tokens = [wnl.lemmatize(word, pos="v") for word in text_tokens]
        return ' '.join(lemmatized_tokens)
    return text
#-----------multiprocessing--------------------
from multiprocessing import Pool

def process_chunk(df_chunk, func, column, *args):
    df_chunk[column] = df_chunk[column].apply(lambda x: func(x, *args))
    return df_chunk

# def parallelize_dataframe(df, func, column, *args, num_partitions=16):
#     """
#     Split the dataframe into chunks, apply the processing function in parallel, and recombine.
#     """
#     chunks = [df.iloc[df.shape[0] // num_partitions * i: df.shape[0] // num_partitions * (i + 1)] for i in range(num_partitions)]
#     pool = Pool(processes=num_partitions)
#     processed_chunks = pool.starmap(process_chunk, [(chunk, func, column, *args) for chunk in chunks])
#     pool.close()
#     pool.join()
#     return pd.concat(processed_chunks)

import multiprocessing

def parallelize_dataframe(df, func, column, *args):
    """
    Split the dataframe into chunks based on CPU count, apply the processing function in parallel, and recombine.
    """
    # Get the number of CPU cores
    num_cores = multiprocessing.cpu_count()
    
    # Use one less than the total number of cores, but at least 1
    num_partitions = max(1, num_cores - 1)
    
    # Split the dataframe into chunks
    chunk_size = df.shape[0] // num_partitions
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, df.shape[0], chunk_size)]
    
    # Create a pool of workers
    pool = Pool(processes=num_partitions)
    
    # Apply the function to each chunk in parallel
    processed_chunks = pool.starmap(process_chunk, [(chunk, func, column, *args) for chunk in chunks])
    
    # Close the pool and wait for the work to finish
    pool.close()
    pool.join()
    
    # Combine the processed chunks
    return pd.concat(processed_chunks)





def toLower(df, column, multiprocessFlag):
    if multiprocessFlag == 1:
        return parallelize_dataframe(df, to_lower, column)
    return toLower0(df, column)

def removePunctuation(df, column, multiprocessFlag):
    if multiprocessFlag == 1:
        processed_df = parallelize_dataframe(df, remove_punctuation, column)
        return processed_df
    df[column] = df[column].apply(remove_punctuation)
    return df

def removeStopwords(df, column, multiprocessFlag):
    if multiprocessFlag == 1:
        stopwords_set = set(nltk.corpus.stopwords.words("english"))
        processed_df = parallelize_dataframe(df, remove_stopwords_from_text, column, stopwords_set)
        return processed_df
    df[column] = df[column].apply(lambda text: remove_stopwords_from_text(text, set(nltk.corpus.stopwords.words("english"))))
    return df


def expandContractions(df, column, multiprocessFlag):
    if multiprocessFlag == 1:
        processed_df = parallelize_dataframe(df, expand_contractions, column)
        return processed_df
    df[column] = df[column].apply(expand_contractions)
    return df


def lemmatizeText(df, column, multiprocessFlag):
    if multiprocessFlag == 1:
        processed_df = parallelize_dataframe(df, lemmatize, column)
        return processed_df
    df[column] = df[column].apply(lemmatize)
    return df
