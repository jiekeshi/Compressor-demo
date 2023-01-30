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
    aaaa
    ''')

option = st.selectbox(
    'Which model would you like to use?',
    ('CodeBERT (481 MB)', 'GraphCodeBERT (481 MB)', 'Compressor (3 MB)'))

st.write('Prediction', st.metric(label="Temperature", value="70 °F", delta="1.2 °F"))
st.write('Cost', st.metric(label="Temperature", value="70 °F", delta="1.2 °F"))
