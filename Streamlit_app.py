# import streamlit as st
# import pandas as pd
# from io import StringIO
# import time
# import os
# import pydeck as pdk
# import numpy as np
# from back import toLower, reutrn_columns, remove_punctuation, remove_stopwords, expand_contractions, lemmatize

# st.set_page_config(layout="wide")


# import base64

# # Function to get the base64 encoded string of a binary file
# def get_base64_of_bin_file(bin_file):
#     with open(bin_file, 'rb') as f:
#         data = f.read()
#     return base64.b64encode(data).decode()

# # Define custom CSS styles
# styles = """
# <style>
# .title-container {
#     display: flex;
#     align-items: center;
#     justify-content: center;
# }

# .title-image {
#     margin-right: 10px;
#     height: 60px;
# }

# .title-text {
#     font-size: 3rem;
#     font-weight: bold;
# }
# </style>
# """

# # Apply the custom CSS styles
# st.markdown(styles, unsafe_allow_html=True)

# # Render the title with an image
# title_html = """
# <div class="title-container">
#     <img class="title-image" src="data:image/png;base64,{}" alt="Title Image">
#     <span class="title-text">NLP Data Preprocessing Tool</span>
# </div>
# """.format(get_base64_of_bin_file("titleimage\\table.png"))

# st.markdown(title_html, unsafe_allow_html=True)
# # # st.title("<center>NLP Data Preprocessing</center>", unsafe_allow_html = True)
# # st.header("<center>Centered Header</center>", unsafe_allow_html=True)

# # st.write("Prepare your dataset here for machine learning")

# st.markdown("<center>Perform end-to-end data preprocessing for your NLP projects</center>", unsafe_allow_html=True)
# st.divider()
# st.write("Upload  the file here")  
# uploaded_file = st.file_uploader("200MB")
# st.divider()  
# if uploaded_file is not None:
#     df1 = pd.read_csv(uploaded_file)
#     # df2 = df1
#     st.header("Initial Dataframe")
#     st.write(df1)

#     file_path = os.path.join("dataset\\", uploaded_file.name)
#     with open(file_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())
    
#     columns_ = df1.columns
#     columns = {"None"}.union(columns_)
#     num_rows = df1.shape[0]
    
#     st.divider()
#     st.subheader("Text Cleaning")

#     col1, col2 = st.columns([0.32, 0.68])
#     with col1:
#         tab1, tab2, tab3 = st.tabs(["Lowercasing", "Remove Punctuation", "Remove Stopwords"])
#         with tab1:
#             st.subheader("Preprocessing Methods")
#             # column_ = st.selectbox("select your column to preprocess", set(columns), index=0)
#             Lowercase_columns = st.multiselect('Select the columns to be lowercased',columns_)
               
#             for i in Lowercase_columns: 
#                 with st.spinner(f'preprocessing column {i}'):
#                     time.sleep(3) 
#                     toLower(df1, i)
#             # toLower(df1, column_)
#         st.divider()
#         with tab2:
#             Remove_punctuation_columns = st.multiselect('Select the columns to remove punctuations from',columns_)
#             # st.write('After removing punctuation from "{}" column'.format("last_sold_agency"))
#             for i in Remove_punctuation_columns:
#                 df1[i] = df1[i].apply(remove_punctuation)

#         with tab3:
#             stopword_columns = st.multiselect('select columns to remove stopwords from', columns_)

#             for i in stopword_columns:  
#                 df1[i] = df1[i].apply(remove_stopwords)
            
#         st.subheader("Stemming/Lemmatization")
#         tab4, tab5, tab6 = st.tabs(["Contractions", "Lemmatization", "Stemming"])

#         with tab4: 
#             st.subheader("Contractions")
#             contractions_ = st.multiselect('Select the columns for contractions',columns_)
#             for i in contractions_:
#                 df1[i] = df1[i].apply(expand_contractions)
#         with tab5:
#             st.subheader("Lemmatization")
#             lemmatize_ = st.multiselect('Select the columns for lemmatization',columns_)
#             for i in lemmatize_:
#                 df1[i] = df1[i].apply(lemmatize)
#         with tab6:
                
#             st.subheader("eryu")


#         # st.markdown("*Streamlit* is **really** ***cool***.")
#         st.divider()  
#     with col2:
#         # st.success("Initial Dataframe")
#         # st.write(df2)
#         st.header("Processed Dataframe")
#         st.write(df1)

#-----------------------------------------------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import time
import os
from back import toLower, removePunctuation, removeStopwords, expandContractions, lemmatizeText

st.set_page_config(layout="wide")

import base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# custom CSS styles
styles = """
<style>
.title-container {
    display: flex;
    align-items: center;
    justify-content: center;
}

.title-image {
    margin-right: 10px;
    height: 60px;
}

.title-text {
    font-size: 3rem;
    font-weight: bold;
}
</style>
"""

# Apply the custom CSS styles
st.markdown(styles, unsafe_allow_html=True)

# Render the title with an image
title_html = """
<div class="title-container">
    <img class="title-image" src="data:image/png;base64,{}" alt="Title Image">
    <span class="title-text">NLP Data Preprocessing Tool</span>
</div>
""".format(get_base64_of_bin_file("titleimage\\table.png"))

st.markdown(title_html, unsafe_allow_html=True)
st.markdown("<center>Perform end-to-end data preprocessing for your NLP projects</center>", unsafe_allow_html=True)
st.divider()
st.write("Upload the file here")  
uploaded_file = st.file_uploader("200MB")
st.divider()  
if uploaded_file is not None:
    df1 = pd.read_csv(uploaded_file)
    st.header("Initial Dataframe")
    st.write(df1)

    file_path = os.path.join("dataset\\", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    columns_ = df1.columns
    columns = {"None"}.union(columns_)
    num_rows = df1.shape[0]
    
    st.divider()
    st.subheader("Text Cleaning")

    col1, col2 = st.columns([0.32, 0.68])
    with col1:
        tab1, tab2, tab3 = st.tabs(["Lowercasing", "Remove Punctuation", "Remove Stopwords"])
        with tab1:
            st.subheader("Preprocessing Methods")
            Lowercase_columns = st.multiselect('Select the columns to be lowercased', columns_)
            for i in Lowercase_columns:
                with st.spinner(f'Processing column {i}'):
                    df1 = toLower(df1, i)
        st.divider()
        with tab2:
            Remove_punctuation_columns = st.multiselect('Select the columns to remove punctuations from', columns_)
            for i in Remove_punctuation_columns:
                with st.spinner(f'Processing column {i}'):
                    df1 = removePunctuation(df1, i)
        with tab3:
            stopword_columns = st.multiselect('Select columns to remove stopwords from', columns_)
            for i in stopword_columns:
                with st.spinner(f'Processing column {i}'):
                    df1 = removeStopwords(df1, i)

        st.subheader("Stemming/Lemmatization")
        tab4, tab5 = st.tabs(["Contractions", "Lemmatization"])
        with tab4: 
            st.subheader("Contractions")
            contractions_ = st.multiselect('Select the columns for contractions', columns_)
            for i in contractions_:
                with st.spinner(f'Processing column {i}'):
                    df1 = expandContractions(df1, i)
        with tab5:
            st.subheader("Lemmatization")
            lemmatize_ = st.multiselect('Select the columns for lemmatization', columns_)
            for i in lemmatize_:
                with st.spinner(f'Processing column {i}'):
                    df1 = lemmatizeText(df1, i)

        st.divider()  
    with col2:
        st.header("Processed Dataframe")
        st.write(df1)
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')
        csv = convert_df(df1)

        st.download_button(
           "Download Processed Data Frame",
           csv,
           "file.csv",
           "text/csv",
           key='download-csv'
        )
    
