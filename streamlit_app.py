from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st

"""
# Welcome to Compressor!

This demo aims to show that the compressed models produced by Compressor outperform the heavy pre-trained models in efficiency.

Below are the examples of using different models to predict whether a given code snippet is vulnerable or not:
"""

txt = st.text_area('Text to analyze', '''
    It was the best of times, it was the worst of times, it was
    the age of wisdom, it was the age of foolishness, it was
    the epoch of belief, it was the epoch of incredulity, it
    was the season of Light, it was the season of Darkness, it
    was the spring of hope, it was the winter of despair, (...)
    ''')
st.write('Sentiment:', txt)