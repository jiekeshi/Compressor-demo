import streamlit as st
st.set_page_config(
   page_title="Compressor-demo",
   page_icon="🧊",
   layout="wide",
   initial_sidebar_state="expanded",
)

"""
# Welcome to Compressor!

This demo aims to show that the compressed models produced by Compressor outperform the heavy pre-trained models in efficiency.

Below are the examples of using different models to predict whether a given code snippet is vulnerable or not:
"""

txt = st.text_area('Code to analyze', '''
    // An vulnerable example form ReVeal dataset
    int getulong(const char * numstr, unsigned long int * result) {
    long long int val;
    char * endptr;
    errno = 0;
    val = strtoll(numstr, & endptr, 0);
    if (('\\0' == * numstr) || ('\\0' != * endptr) || (ERANGE == errno) || (val != (unsigned long int) val)) {
        return 0;
    }
    * result = (unsigned long int) val;
    return 1;
}
    ''')

option = st.selectbox(
    'Which model would you like to use?',
    ('CodeBERT (481 MB)', 'GraphCodeBERT (481 MB)', 'Compressor-CodeBERT (3 MB)', 'Compressor-GraphCodeBERT (3 MB)'))

col1, col2, col3 = st.columns(3)
col1.metric("Prediction", "70 °F")
col2.metric("Latency", "9 mph")
col3.metric("Memory", "86%")
