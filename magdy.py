import random 
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import librosa
import librosa.display
import wave
import streamlit_vertical_slider  as svs
from scipy.io import wavfile as wav
from scipy.fftpack import fft
from scipy.fft import fft, fftfreq, fftshift
from scipy.fft import rfft, rfftfreq
from scipy.fft import irfft
from scipy.io.wavfile import write
from scipy.signal import find_peaks


def getPeaksFrequencies(xAxis,yAxis):
    
    amplitude = np.abs(rfft(yAxis))
    frequency = rfftfreq(len(xAxis), (xAxis[1]-xAxis[0]))
    indices = find_peaks(amplitude)
    return np.round(frequency[indices[0]],1)


#  ----------------------------------- TRIAL FOURIER TRANSFORM FUNCTION ---------------------------------------------------
def dataframe_fourier_transform(dataframe):

    signal_y_axis = (dataframe.iloc[:,1]).to_numpy() # dataframe y axis
    signal_x_axis = (dataframe.iloc[:,0]).to_numpy() # dataframe x axis

    duration    = signal_x_axis[-1] # the last point in the x axis (number of seconds in the data frame)
    sample_rate = len(signal_y_axis) / duration # returns number points per second

    fourier_y_axis = rfft(signal_y_axis) # returns complex numbers of the y axis in the data frame
    fourier_x_axis = rfftfreq(len(signal_y_axis), (signal_x_axis[1]-signal_x_axis[0])) # returns the frequency x axis after fourier transform


    frequencies = getPeaksFrequencies(signal_x_axis,signal_y_axis)
    points_per_freq = len(fourier_x_axis) / (sample_rate/2) #points per freq is the index of 1 HZ freq. 
    
    fourier_y_axis = dataframe_creating_sliders(frequencies, points_per_freq, fourier_y_axis) # calling creating sliders function

    modified_signal = irfft(fourier_y_axis) # returning the inverse transform after modifying it with sliders 

    fig, axs = plt.subplots()
    fig.set_size_inches(14,5)
    placeOfPeaksOnXAxis = (frequencies*points_per_freq).astype(int)
    plt.plot(fourier_x_axis, np.abs(fourier_y_axis)) #plotting signal before modifying
    plt.plot(placeOfPeaksOnXAxis, np.abs(fourier_y_axis)[placeOfPeaksOnXAxis], marker="o") # plotting peaks points
    
    st.plotly_chart(fig,use_container_width=True)

    fig2, axs2 = plt.subplots()
    fig2.set_size_inches(14,5)
    plt.plot(signal_x_axis,modified_signal) # ploting signal after modifying
    st.plotly_chart(fig2,use_container_width=True)

#  ----------------------------------- CREATING SLIDERS ---------------------------------------------------------------
def dataframe_creating_sliders(frequencies,points_per_freq,fourier_y_axis):

    # peak_frequencies = fourier_x_axis[frequencies[:]]
    columns = st.columns(10)

    for index, frequency in enumerate(frequencies):

        with columns[index]:
            slider_range = svs.vertical_slider(min_value=0.0, max_value=2.0, default_value=1.0, step=0.1, key=index)
            
        # these three lines determine the range that will be modified by the slider
        target_idx   = int(points_per_freq * (frequencies[index]-1)) 
        target_idx_2 = int(points_per_freq * (frequencies[index]+1))
        if slider_range is not None:
            fourier_y_axis[target_idx - 1 : target_idx_2 + 2] *= slider_range

    return fourier_y_axis

def wave_static_sliders(fourier_y_axis,pointsPerFrequency):

    # peak_frequencies = fourier_x_axis[peaks_indeces[:]]
    columns = st.columns(10)
    frequencies = np.arange(20,21000,20)

    for i in range(10) :
        minValue = (2000*i)
        maxValue = (2000*(i+1))
        # with columns[i]:
            # slider_range = svs.vertical_slider(min_value=0, max_value=10, default_value=1, step=1, key=i)
            # st.write(minValue)
            # st.write(maxValue)
        
        
        # these three lines determine the range that will be modified by the slider
        target_idx   = int(minValue*pointsPerFrequency) 
        target_idx_2 = int( maxValue*pointsPerFrequency)
        # if slider_range is not None:
        #     fourier_y_axis[target_idx : target_idx_2 + 2] *= slider_range

    return fourier_y_axis

#-------------------------------------- Fourier Transform on Audio ----------------------------------------------------
def fourier_for_audio(uploaded_file):
 
    st.audio(uploaded_file, format='audio/wav') # displaying the audio
    obj = wave.open(uploaded_file, 'rb')
    sample_rate = obj.getframerate()   # number of samples per second
    n_samples = obj.getnframes()       # total number of samples in the whole audio
    duration = n_samples / sample_rate # duration of the audio file
    signal_wave = obj.readframes(-1)   # amplitude of the sound

    signal_y_axis = np.frombuffer(signal_wave, dtype=np.int32)
    signal_x_axis = np.linspace(0, duration, len(signal_y_axis))

    # plotting in time domain before processing
    plotting(signal_x_axis,signal_y_axis)
        
    fft_out = rfft(signal_y_axis) # el fourier bt3t el amplitude eli hnshtghl beha fl equalizer
    fft_out = np.abs(fft_out)# np.abs 3shan el rsm

    x_axis_fourier = rfftfreq(len(signal_y_axis),(1/sample_rate)) #3shan mbd2sh mn -ve
    
    # plotting in frequency domain before processing
    plotting(x_axis_fourier,fft_out)

    
    points_per_freq = len(x_axis_fourier) / (sample_rate/2) #points per freq is the index of 1 HZ freq. 
    fourier_y_axis = wave_static_sliders( fft_out,points_per_freq) # calling creating sliders function
    
    
    
    # plotting in frequency domain after processing
    plotting(x_axis_fourier,fourier_y_axis)
    modified_signal = irfft(fourier_y_axis) # returning the inverse transform after modifying it with sliders 

    biobio = np.int32(modified_signal)
    
    # plotting in time domain after processing
    plotting(signal_x_axis,biobio)

    
    write("audioTest.wav", sample_rate, biobio)
    st.audio("audioTest.wav", format='audio/wav')

#-------------------------------------- PLOTTING AUDIO ----------------------------------------------------
def plotting(x_axis_fourier,fft_out):
    # Plotting audio Signal
    figure, axis = plt.subplots()
    plt.subplots_adjust(hspace=1)
    axis.plot(x_axis_fourier,fft_out)
    st.plotly_chart(figure,use_container_width=True)
#------------- tagroba fashla ------------