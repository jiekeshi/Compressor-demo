import streamlit as st
st.set_page_config(
   page_title="Compressor-demo",
   page_icon="🧊",
   layout="wide",
   initial_sidebar_state="expanded",
)

from inference import CodeBERT_compressor_predict, CodeBERT_predict, GraphCodeBERT_compressor_predict, GraphCodeBERT_predict

"""
# Welcome to Compressor!

This demo aims to show that the compressed models produced by Compressor outperform the heavy pre-trained models in efficiency.

Below are the examples of using different models to predict whether a given code snippet is vulnerable or not:
"""

txt = st.text_area('Code to analyze', '''
    // An example form ReVeal dataset
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
    ''', height=450)

col1, col2, col3, col4 = st.columns(4)
latency, pred = CodeBERT_predict(txt)
if pred:
    pred = str(pred)
else:
    pred = str(pred)
col1.metric("Model", "CodeBERT")
col2.metric("Model Size", "481 MB")
col3.metric("Latency", str(round(latency*1000, 2))+" ms")
col4.metric("Prediction", pred)

col1, col2, col3, col4 = st.columns(4)
latency, pred = CodeBERT_compressor_predict(txt)
if pred:
    pred = "Vulnerable"
else:
    pred = "Safe"
col1.metric("Model", "CB-Compressor")
col2.metric("Model Size", "3 MB")
col3.metric("Latency", str(round(latency*1000, 2))+" ms")
col4.metric("Prediction", pred)

col1, col2, col3, col4 = st.columns(4)
latency, pred = GraphCodeBERT_predict(txt)
if pred:
    pred = "Vulnerable"
else:
    pred = "Safe"
col1.metric("Model", "GraphCodeBERT")
col2.metric("Model Size", "481 MB")
col3.metric("Latency", str(round(latency*1000, 2))+" ms")
col4.metric("Prediction", pred)

col1, col2, col3, col4 = st.columns(4)
latency, pred = GraphCodeBERT_compressor_predict(txt)
if pred:
    pred = "Vulnerable"
else:
    pred = "Safe"
col1.metric("Model", "GCB-Compressor")
col2.metric("Model Size", "3 MB")
col3.metric("Latency", str(round(latency*1000, 2))+" ms")
col4.metric("Prediction", pred)

"""
The above results show that the compressed models are able to achieve the same prediction results as the pre-trained models while being significantly smaller and faster.
"""