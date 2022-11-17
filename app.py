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
import random

st.set_page_config(layout="wide")

with open('style.css') as fileStyle:
    st.markdown(f'<style>{fileStyle.read()}</style>',unsafe_allow_html=True)

if 'stopPoint' not in st.session_state:
    st.session_state['stopPoint'] = 0
if 'played' not in st.session_state:
    st.session_state['played'] = False
if 'startPoint' not in st.session_state:
    st.session_state['startPoint'] = 0
if 'play' not in st.session_state:
    st.session_state['play'] = False

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
    graph_container = st.container()
    file = st.sidebar.file_uploader("Upload file", type=['wav', 'csv'])
    with st.sidebar:
        option = st.selectbox(
            "Option", ["Music (uniform frequency sliders)", "Music", "Vowels", "Medical", "Synthetic", "Pitch modifier"])
    if option == "Music (uniform frequency sliders)":
       
        col1 ,col2,col3,col4,col5,col6,col7,col8,col9,col10 = st.columns(10) 

        
    
        # comp_1 = st.slider("20-2000 Hz", min_value=0,
        #                    max_value=10, value=1)
        
        
       
        
        spectroCheckBox = st.sidebar.checkbox('Show spectrogram')
        if file is not None:
            with col1:
                comp_1 = fn.vertical_slider("20-2000 Hz",1, 1,0, 10,4)
            with col2:
                comp_2 = fn.vertical_slider("2-4 KHz",1, 1,0, 10,5)
            with col3:
                comp_3 = fn.vertical_slider("4-6 KHz",1, 1,0, 10,6)
            with col4:
                comp_4 = fn.vertical_slider("6-8 KHz",1, 1,0, 10,7)
            with col5:
                comp_5 = fn.vertical_slider("8-10 KHz ",1, 1,0, 10,8)
            with col6:
                comp_6 = fn.vertical_slider("10-12 KHz",1, 1,0, 10,9)
            with col7:
                comp_7 = fn.vertical_slider("12-14 KHz",1, 1,0, 10,10)
            with col8:
                comp_8 = fn.vertical_slider("14-16 KHz",1, 1,0, 10,11)
            with col9:
                comp_9 = fn.vertical_slider("16-18 KHz",1, 1,0, 10,12)
            with col10:
                comp_10 = fn.vertical_slider("18-20 KHz",1, 1,0, 10,13)
                
            with graph_container:
                fn.uniform_audio_fourier_transform(
                    file, comp_1, comp_2, comp_3, comp_4, comp_5, comp_6, comp_7, comp_8, comp_9, comp_10, spectroCheckBox)
    elif option == "Music":
        col1, col2,col3 = st.columns(3)
        
        # with st.sidebar:
        #     col1, col2 = st.columns(2)
        #     with col1:
        #         # guitar = st.slider("guitar", 0, 10, 1)
        #         piano = st.slider("piano", 0, 10, 1)
        #     with col2:
        #         flute = st.slider("flute", 0, 10, 1)
        
        spectroCheckBox = st.sidebar.checkbox('Show spectrogram')
        if file is not None:
            with col1:
                guitar = fn.vertical_slider("Guitar",1,1, 0, 10, 1)
            with col2:
                piano = fn.vertical_slider("Piano",1,1, 0, 10, 2)
            with col3:
                flute = fn.vertical_slider("Fluet",1,1, 0, 10, 3)

            with graph_container:
                fn.audio_fourier_transform(
                    file, guitar, flute, piano, spectroCheckBox)
    elif option == "Vowels":
        with st.sidebar:
            col1, col2 = st.columns(2)
            with col1:
                er_vowel = st.slider("/er/", min_value=0,
                                     max_value=10, value=1)
                a_vowel = st.slider("/a/", min_value=0, max_value=10, value=1)

            with col2:
                iy_vowel = st.slider("/iy/", min_value=0,
                                   max_value=10, value=1)
                oo_vowel = st.slider("/oo/", min_value=0,
                                   max_value=10, value=1)

            uh_vowel = st.slider("/uh/", min_value=0, max_value=10, value=1)
            spectroCheckBox = st.checkbox('Show spectrogram')
        if file is not None:
            fn.vowel_audio_fourier_transform(
                file, er_vowel, a_vowel, iy_vowel, oo_vowel, uh_vowel, spectroCheckBox)
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

    elif option == "Pitch modifier":
        with st.sidebar:
            semitone = st.slider("Semitone", -10, 10, value=0)
            spectroCheckBox = st.checkbox('Show spectrogram')

        if file is not None:
            fn.pitch_modifier(file, semitone, spectroCheckBox)

    

if __name__ == "__main__":
    head()
    body()
