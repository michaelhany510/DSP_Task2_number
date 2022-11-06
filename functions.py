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
import librosa
import librosa.display

def getPeaksFrequencies(xAxis,yAxis):
    amplitude = np.abs(rfft(yAxis))
    frequency = rfftfreq(len(xAxis), (xAxis[1]-xAxis[0]))
    indices = find_peaks(amplitude)
    return np.round(frequency[indices[0]],1)


#-------------------------------------- Fourier Transform on Audio ----------------------------------------------------
def audio_fourier_transform(audio_file,guitar,flute,piano,spectroCheckBox):
    column1,column2 = st.columns(2)
    with column1:    
        st.audio(audio_file, format='audio/wav') # displaying the audio
    obj = wave.open(audio_file, 'rb')
    sample_rate = obj.getframerate()      # number of samples per second
    n_samples   = obj.getnframes()        # total number of samples in the whole audio
    duration    = n_samples / sample_rate # duration of the audio file
    signal_wave = obj.readframes(-1)      # amplitude of the sound

    
    signal_y_axis = np.frombuffer(signal_wave, dtype=np.int32)
    signal_x_axis = np.linspace(0, duration, len(signal_y_axis))
    with column1:
        if not spectroCheckBox:
            plotting(signal_x_axis[:1000],signal_y_axis[:1000])
        else:
            plot_spectro(audio_file.name)
    
    
    yf = rfft(signal_y_axis) # returns complex numbers of the y axis in the data frame
    xf = rfftfreq(len(signal_y_axis), (signal_x_axis[1]-signal_x_axis[0])) # returns the frequency x axis after fourier transform

    peaks = find_peaks(yf)   # computes peaks of the signal 
    peaks_indeces = peaks[0] # indeces of frequency with high peaks
    points_per_freq = len(xf) / (xf[-1]) # NOT UNDERSTANDABLE 
    
    # with column1:
    #     plotting(xf,np.abs(yf))


    # these three lines determine the range that will be modified by the slider
    yf[:int(600*points_per_freq)] *= guitar
    yf[int(600*points_per_freq):int(3000*points_per_freq)] *= flute
    yf[int(3000*points_per_freq):] *= piano
    
    
    
    modified_signal = irfft(yf) # returning the inverse transform after modifying it with sliders 
    tryyy = np.int32(modified_signal)

    
    write("example.wav", sample_rate, tryyy)
    with column2:
        st.audio("example.wav", format='audio/wav')
        if not spectroCheckBox:
            plotting(signal_x_axis[:1000],tryyy[:1000])
        else:
            plot_spectro("example.wav")
    # with column2:
    #     plotting(xf,np.abs(yf))
    
def uniform_audio_fourier_transform(audio_file,comp_1,comp_2,comp_3,comp_4,comp_5,comp_6,comp_7,comp_8,comp_9,comp_10,spectroCheckBox):
    column1,column2 = st.columns(2)
    with column1:    
        st.audio(audio_file, format='audio/wav') # displaying the audio
    obj = wave.open(audio_file, 'rb')
    sample_rate = obj.getframerate()      # number of samples per second
    n_samples   = obj.getnframes()        # total number of samples in the whole audio
    duration    = n_samples / sample_rate # duration of the audio file
    signal_wave = obj.readframes(-1)      # amplitude of the sound

    
    signal_y_axis = np.frombuffer(signal_wave, dtype=np.int32)
    signal_x_axis = np.linspace(0, duration, len(signal_y_axis))
    with column1:
        if not spectroCheckBox:
            plotting(signal_x_axis[:1000],signal_y_axis[:1000])
        else:
            plot_spectro(audio_file.name)
    
    
    yf = rfft(signal_y_axis) # returns complex numbers of the y axis in the data frame
    xf = rfftfreq(len(signal_y_axis), (signal_x_axis[1]-signal_x_axis[0])) # returns the frequency x axis after fourier transform

    peaks = find_peaks(yf)   # computes peaks of the signal 
    peaks_indeces = peaks[0] # indeces of frequency with high peaks
    points_per_freq = len(xf) / (xf[-1]) # NOT UNDERSTANDABLE 
    
    # with column1:
    #     plotting(xf,np.abs(yf))


    # these three lines determine the range that will be modified by the slider
    yf[int(20*points_per_freq):int(2000*points_per_freq)] *= comp_1
    yf[int(2000*points_per_freq):int(4000*points_per_freq)] *= comp_2
    yf[int(4000*points_per_freq):int(6000*points_per_freq)] *= comp_3
    yf[int(6000*points_per_freq):int(8000*points_per_freq)] *= comp_4
    yf[int(8000*points_per_freq):int(10000*points_per_freq)] *= comp_5
    yf[int(10000*points_per_freq):int(12000*points_per_freq)] *= comp_6
    yf[int(12000*points_per_freq):int(14000*points_per_freq)] *= comp_7
    yf[int(14000*points_per_freq):int(16000*points_per_freq)] *= comp_8
    yf[int(16000*points_per_freq):int(18000*points_per_freq)] *= comp_9
    yf[int(18000*points_per_freq):int(20000*points_per_freq)] *= comp_10
    
    
    
    
    modified_signal = irfft(yf) # returning the inverse transform after modifying it with sliders 
    tryyy = np.int32(modified_signal)

    
    write("example.wav", sample_rate, tryyy)
    with column2:
        st.audio("example.wav", format='audio/wav')
        if not spectroCheckBox:
            plotting(signal_x_axis[:1000],tryyy[:1000])
        else:
            plot_spectro("example.wav")
    # with column2:
    #     plotting(xf,np.abs(yf))
   
def plotting(x_axis_fourier,fft_out):
    # Plotting audio Signal
    figure, axis = plt.subplots()
    plt.subplots_adjust(hspace=1)
    axis.plot(x_axis_fourier,fft_out)
    st.plotly_chart(figure,use_container_width=True)
# def plotSpecGram(data,sampling_rate):
#     # Plotting spectrogram
#     # figure, axis = plt.subplots()
#     # plt.subplots_adjust(hspace=1)
#     # axis.specgram(data,sampling_rate,cmap = plt.cm.bone)
#     figure = plt.figure()
#     figure.patch.set_facecolor('xkcd:#0e1117')
#     st.pyplot(figure,use_container_width=True)

def plot_spectro(audio_file):
    # if type(audio_file == 'str'):
    # y, sr = librosa.load(audio_file)
    # D = librosa.stft(y)  # STFT of y
    # S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    # fig = plt.figure(figsize=[10,6])
    # librosa.display.specshow(S_db)
    # st.pyplot(fig)
    
    y, sr = librosa.load(audio_file)
    D = librosa.stft(y)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax)
    ax.set(title='')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    st.pyplot(fig)
    
    # y, sr = librosa.load(audio_file)
    # # D = librosa.stft(y)  # STFT of y
    # # S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    # fig, ax = plt.subplots()
    # img = librosa.feature.melspectrogram(y,sr)
    # ax.set(title='')
    # # fig.colorbar(img, ax=ax, format="%+2.f dB")
    # st.pyplot(fig)