# import pandas as pd
# import numpy as np
# import nltk
# import string
# import contractions as contr
# from nltk.stem import WordNetLemmatizer

# def reutrn_columns(df):
#     return(df.columns)

# def toLower(df, column):
#     df[column] = df[column].str.lower()
#     return df

# def remove_punctuation(text):
#     if isinstance(text, str):
#         tokenizer = nltk.RegexpTokenizer(r'\w+')
#         tokens = tokenizer.tokenize(text)
#         return ' '.join(tokens)
#     else:
#         return text
    
# def remove_stopwords(text):
#     if isinstance(text, str):
#         tokens = nltk.word_tokenize(text)
#         stopwords = nltk.corpus.stopwords.words("english")
#         filtered = [token for token in tokens if token.lower() not in stopwords]
#         return ' '.join(filtered)

# def expand_contractions(text):
#     if isinstance(text, str):
#         expanded_words = [] 
#         for word in text.split():
#             expanded_words.append(contr.fix(word))  
        
#         return ' '.join(expanded_words)

# # def lemmatize(text):
# #     wnl = WordNetLemmatizer()
# #     if isinstance(text, str):
# #         text_tokens = nltk.word_tokenize(text)
# #         return " ".join(wnl.lemmatize(text_tokens, pos="v"))
    
# # =========================================
# # nltk.download("wordnet")
# # nltk.download("omw-1.4")
# # =========================================
    
# def lemmatize(text):
#     wnl = WordNetLemmatizer()
#     if isinstance(text, str):
#         text_tokens = nltk.word_tokenize(text)
#         lemmatized_tokens = [wnl.lemmatize(word, pos="v") for word in text_tokens]
#         return ' '.join(lemmatized_tokens)

#--------------------------------------------------------------------------------------

import pandas as pd
import nltk
import contractions as contr
from nltk.stem import WordNetLemmatizer
from multiprocessing import Pool

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download("wordnet")
# nltk.download("omw-1.4")


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

def process_chunk(df_chunk, func, column, *args):
    df_chunk[column] = df_chunk[column].apply(lambda x: func(x, *args))
    return df_chunk

def parallelize_dataframe(df, func, column, *args, num_partitions=16):
    """
    Split the dataframe into chunks, apply the processing function in parallel, and recombine.
    """
    chunks = [df.iloc[df.shape[0] // num_partitions * i: df.shape[0] // num_partitions * (i + 1)] for i in range(num_partitions)]
    pool = Pool(processes=num_partitions)
    processed_chunks = pool.starmap(process_chunk, [(chunk, func, column, *args) for chunk in chunks])
    pool.close()
    pool.join()
    return pd.concat(processed_chunks)

def toLower(df, column):
    return parallelize_dataframe(df, to_lower, column)

def removePunctuation(df, column):
    return parallelize_dataframe(df, remove_punctuation, column)

def removeStopwords(df, column):
    stopwords_set = set(nltk.corpus.stopwords.words("english"))
    return parallelize_dataframe(df, remove_stopwords_from_text, column, stopwords_set)

def expandContractions(df, column):
    return parallelize_dataframe(df, expand_contractions, column)

def lemmatizeText(df, column):
    return parallelize_dataframe(df, lemmatize, column)
