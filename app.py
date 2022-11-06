import numpy as np
from matplotlib import pyplot as plt
from scipy.io.wavfile import write
from scipy.fft import rfft, rfftfreq
from scipy.fft import irfft
import streamlit as st
import plotly.graph_objects as go
from scipy.signal import find_peaks
import pandas as pd
from scipy.io import wavfile
from IPython.display import Audio
import functions as fn

st.set_page_config(layout="wide")


def head():
    # with st.sidebar:
    st.markdown("""
            <h1 style='text-align: center; margin-bottom: -35px; margin-top:-80px'>
            Equalizer
            </h1>
        """, unsafe_allow_html=True
                )

    st.caption("""
            <p style='text-align: center; position: relative; margin-top:-25px;'>
            by team x
            </p>
        """, unsafe_allow_html=True
               )


def body():
    file = st.sidebar.file_uploader("Upload file", type=['wav', 'csv'])
    with st.sidebar:
        option = st.selectbox(
            "Option", ["Uniform", "Music", "Vowels", "Medical", "Synthetic"])
    if option == "Uniform":
        with st.sidebar:
            col1, col2 = st.columns(2)
            with col1:
                comp_1 = st.slider("comp_1", min_value=0, max_value=10)
                comp_2 = st.slider("comp_2", min_value=0, max_value=10)
                comp_3 = st.slider("comp_3", min_value=0, max_value=10)
                comp_4 = st.slider("comp_4", min_value=0, max_value=10)
                comp_5 = st.slider("comp_5", min_value=0, max_value=10)
            with col2:
                comp_6 = st.slider("comp_6", min_value=0, max_value=10)
                comp_7 = st.slider("comp_7", min_value=0, max_value=10)
                comp_8 = st.slider("comp_8", min_value=0, max_value=10)
                comp_9 = st.slider("comp_9", min_value=0, max_value=10)
                comp_10 = st.slider("comp_10", min_value=0, max_value=10)
    elif option == "Music":
        with st.sidebar:
            col1, col2 = st.columns(2)
            with col1:
                guitar = st.slider("guitar", 0, 10,1)
                biano = st.slider("biano",0,10,1)
            with col2:
                flute = st.slider("flute", 0,10,1)
            spectroCheckBox = st.checkbox('Show spectrogram')
        if file is not None:
            fn.audio_fourier_transform(file,guitar,flute,biano,spectroCheckBox)
    elif option == "Vowels":
        with st.sidebar:
            col1, col2 = st.columns(2)
            with col1:
                comp_1 = st.slider("comp_1", min_value=0, max_value=10)
                comp_2 = st.slider("comp_2", min_value=0, max_value=10)
                comp_3 = st.slider("comp_3", min_value=0, max_value=10)
                comp_4 = st.slider("comp_4", min_value=0, max_value=10)
                comp_5 = st.slider("comp_5", min_value=0, max_value=10)
            with col2:
                comp_6 = st.slider("comp_6", min_value=0, max_value=10)
                comp_7 = st.slider("comp_7", min_value=0, max_value=10)
                comp_8 = st.slider("comp_8", min_value=0, max_value=10)
                comp_9 = st.slider("comp_9", min_value=0, max_value=10)
                comp_10 = st.slider("comp_10", min_value=0, max_value=10)
    elif option == "Medical":
        with st.sidebar:
            col1, col2 = st.columns(2)
            with col1:
                comp_1 = st.slider("comp_1", min_value=0, max_value=10)
            with col2:
                comp_2 = st.slider("comp_2", min_value=0, max_value=10)
    elif option == "Synthetic":
        with st.sidebar:
            col1, col2 = st.columns(2)
            with col1:
                comp_1 = st.slider("comp_1", min_value=0, max_value=10)
                comp_2 = st.slider("comp_2", min_value=0, max_value=10)
            with col2:
                comp_3 = st.slider("comp_3", min_value=0, max_value=10)
                comp_4 = st.slider("comp_4", min_value=0, max_value=10)


if __name__ == "__main__":
    head()
    body()
