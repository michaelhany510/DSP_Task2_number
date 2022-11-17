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

uniformModeSlidersNames = ['20 - 2000 hz','2 - 4 khz','4 - 6 khz','6 - 8 khz','8 - 10 khz','10 - 12 khz','12 - 14 khz','14 - 16 khz','16 - 18 khz', '18 - 20 khz']

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
            by team 9
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
       
        columns = st.columns(10) 
        
        spectroCheckBox = st.sidebar.checkbox('Show spectrogram')
        uniformModeSliders = []
        if file is not None:
            i = 0
            for name in uniformModeSlidersNames:
                with columns[i]:
                    uniformModeSliders.append(fn.vertical_slider(name,1, 1,0, 10,10+i))
                i += 1
            with graph_container:
                fn.uniform_audio_fourier_transform(file, uniformModeSliders, spectroCheckBox)
    elif option == "Music":
        col1, col2,col3 = st.columns(3)
        
        spectroCheckBox = st.sidebar.checkbox('Show spectrogram')
        if file is not None:
            with st.container():
                with col1:
                    guitar = fn.vertical_slider("Guitar",1,1, 0, 10, 1)
                with col2:
                    piano = fn.vertical_slider("Piano",1,1, 0, 10, 2)
                with col3:
                    flute = fn.vertical_slider("Flute",1,1, 0, 10, 3)

            with graph_container:
                fn.audio_fourier_transform(
                    file, guitar, flute, piano, spectroCheckBox)
    elif option == "Vowels":
        
        if file is not None:

            with st.container():
                with st.sidebar:
                
                    spectroCheckBox = st.checkbox('Show spectrogram')
            
                col1, col2,col3,col4,col5 = st.columns(5)
                with col1:
                    ʃ_slider = fn.vertical_slider("/ʃ/",1,1, 0, 10,4 )
                with col2:
                    ʊ_slider = fn.vertical_slider("/ʊ/",1,1, 0, 10,5 )
                with col3:   
                    a_slider = fn.vertical_slider("/a/",1,1, 0, 10,6 )
                with col4:   
                    r_slider = fn.vertical_slider("/r/",1,1, 0, 10,8 )
                with col5:   
                    b_slider = fn.vertical_slider("/b/",1,1, 0, 10,9 )
                with graph_container:
            
                    fn.vowel_audio_fourier_transform(
                    file, ʃ_slider, ʊ_slider, a_slider, r_slider, b_slider, spectroCheckBox)
                
            
            


 
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
        fn.arrhythima()

    elif option == "Pitch modifier":
        with st.sidebar:
            semitone = st.slider("Semitone", -10, 10, value=0)
            spectroCheckBox = st.checkbox('Show spectrogram')

        if file is not None:
            fn.pitch_modifier(file, semitone, spectroCheckBox)

    

if __name__ == "__main__":
    head()
    body()
