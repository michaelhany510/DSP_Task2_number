import random 
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit_vertical_slider  as svs
from scipy.fftpack import fft
from scipy.fft import fft, fftfreq, fftshift
from scipy.fft import rfft, rfftfreq
from scipy.fft import irfft
from scipy.io.wavfile import write
from scipy.signal import find_peaks
import wave
import IPython.display as ipd

def getPeaksFrequencies(xAxis,yAxis):
    amplitude = np.abs(rfft(yAxis))
    frequency = rfftfreq(len(xAxis), (xAxis[1]-xAxis[0]))
    indices = find_peaks(amplitude)
    return np.round(frequency[indices[0]],1)


#-------------------------------------- Fourier Transform on Audio ----------------------------------------------------
def audio_fourier_transform(audio_file):

    st.audio(audio_file, format='audio/wav') # displaying the audio
    obj = wave.open(audio_file, 'rb')
    sample_rate = obj.getframerate()      # number of samples per second
    n_samples   = obj.getnframes()        # total number of samples in the whole audio
    duration    = n_samples / sample_rate # duration of the audio file
    signal_wave = obj.readframes(-1)      # amplitude of the sound

    
    signal_y_axis = np.frombuffer(signal_wave, dtype=np.int32)
    signal_x_axis = np.linspace(0, duration, len(signal_y_axis))

    plotting(signal_x_axis[:1000],signal_y_axis[:1000])
    
    
    yf = rfft(signal_y_axis) # returns complex numbers of the y axis in the data frame
    xf = rfftfreq(len(signal_y_axis), (signal_x_axis[1]-signal_x_axis[0])) # returns the frequency x axis after fourier transform

    peaks = find_peaks(yf)   # computes peaks of the signal 
    peaks_indeces = peaks[0] # indeces of frequency with high peaks
    points_per_freq = len(xf) / (xf[-1]) # NOT UNDERSTANDABLE 
      
    plotting(xf,np.abs(yf))

    
    guitar =  st.slider('Guitar',0,10,1)
    flute = st.slider('Flute',0,10,1)
    biano = st.slider('Biano',0,10,1)
    

    # these three lines determine the range that will be modified by the slider
    yf[int(0*points_per_freq):int(600*points_per_freq)] *= guitar
    yf[int(600*points_per_freq):int(3000*points_per_freq)] *= flute
    yf[int(3000*points_per_freq):int(20000*points_per_freq)] *= biano
    
    
    plotting(xf,np.abs(yf))
    
    modified_signal = irfft(yf) # returning the inverse transform after modifying it with sliders 
    tryyy = np.int32(modified_signal)

    plotting(signal_x_axis[:1000],tryyy[:1000])
    
    write("example.wav", sample_rate, tryyy)
    st.audio("example.wav", format='audio/wav')

def plotting(x_axis_fourier,fft_out):
    # Plotting audio Signal
    figure, axis = plt.subplots()
    plt.subplots_adjust(hspace=1)
    axis.plot(x_axis_fourier,fft_out)
    st.plotly_chart(figure,use_container_width=True)