import streamlit as st
import pandas as pd
import time
import os
from back import toLower, removePunctuation, removeStopwords, expandContractions, lemmatizeText

st.set_page_config(layout="wide")
import multiprocessing

num_cores = multiprocessing.cpu_count()
# print(f"Number of CPU cores: {num_cores}")
st.write(num_cores)
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



# title_html = """
# <div class="title-container">
#     <img class="title-image" src="data:image/png;base64,{}" alt="Title Image">
#     <span class="title-text">NLP Data Preprocessing Tool</span>
# </div>
# """.format(get_base64_of_bin_file("titleimage\\table.png"))



# st.markdown(title_html, unsafe_allow_html=True)
st.markdown("<center>Perform end-to-end data preprocessing for your NLP projects</center>", unsafe_allow_html=True)
st.divider()
st.write("Upload the file here")  
uploaded_file = st.file_uploader("200MB")
st.divider()  
if uploaded_file is not None:
    df1 = pd.read_csv(uploaded_file)
    st.header("Initial Dataframe (Sample of 100 rows)")
    st.write(df1.sample(n=min(100, len(df1))))

    # file_path = os.path.join("dataset\\", uploaded_file.name)
    # with open(file_path, "wb") as f:
    #     f.write(uploaded_file.getbuffer())
    
    columns_ = df1.columns
    columns = {"None"}.union(columns_)
    num_rows = df1.shape[0]
    multiprocessFlag = 1 if num_rows > 25000 else 0
    st.divider()
    st.header("Preprocessing Methods")
    st.subheader("Text Cleaning")
    processLog = []
    col1, col2 = st.columns([0.32, 0.68])
    with col1:
        tab1, tab2, tab3 = st.tabs(["Lowercasing", "Remove Punctuation", "Remove Stopwords"])
        with tab1:
            Lowercase_columns = st.multiselect('Select the columns to be lowercased', columns_)
            for i in Lowercase_columns:
                with st.spinner(f'Processing column {i}'):
                    df1 = toLower(df1, i, multiprocessFlag)
                    processLog.append(f'Processed column "{i}": lowercased')
        # st.divider()
        with tab2:
            Remove_punctuation_columns = st.multiselect('Select the columns to remove punctuations from', columns_)
            for i in Remove_punctuation_columns:
                with st.spinner(f'Processing column {i}'):
                    df1 = removePunctuation(df1, i, multiprocessFlag)
                    processLog.append(f'Processed column "{i}": removed punctuations')
        with tab3:
            stopword_columns = st.multiselect('Select columns to remove stopwords from', columns_)
            for i in stopword_columns:
                with st.spinner(f'Processing column {i}'):
                    df1 = removeStopwords(df1, i, multiprocessFlag)
                    processLog.append(f'Processed column "{i}": removed stopwords')
        st.subheader("Stemming/Lemmatization")
        tab4, tab5 = st.tabs(["Contractions", "Lemmatization"])
        with tab4: 
            st.subheader("Contractions")
            contractions_ = st.multiselect('Select the columns for contractions', columns_)
            for i in contractions_:
                with st.spinner(f'Processing column {i}'):
                    df1 = expandContractions(df1, i, multiprocessFlag)
                    processLog.append(f'Processed column "{i}": expanded contractions')
        with tab5:
            st.subheader("Lemmatization")
            lemmatize_ = st.multiselect('Select the columns for lemmatization', columns_)
            for i in lemmatize_:
                with st.spinner(f'Processing column {i}'):
                    df1 = lemmatizeText(df1, i, multiprocessFlag)
                    processLog.append(f'Processed column "{i}": lemmatized')

        # st.divider()  
    with col2:
        st.header("Processed Dataframe")
        st.write(df1)
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')
        csv = convert_df(df1)

        st.download_button("Download Processed Dataframe",csv,"file.csv","text/csv",key='download-csv')

    st.divider()
    st.header("Log")
    for i in processLog:
        st.code(i)
    # st.code(processLog)
