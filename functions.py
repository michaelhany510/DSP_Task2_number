import random
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit_vertical_slider as svs
from scipy.fftpack import fft
from scipy.fft import fft, fftfreq, fftshift
from scipy.fft import rfft, rfftfreq
from scipy.fft import irfft
from scipy.io.wavfile import write
from scipy.signal import find_peaks
import wave
from scipy import signal
import IPython.display as ipd
import librosa
import librosa.display
import soundfile as sf
import pyrubberband as pyrb
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import altair as alt


def getPeaksFrequencies(xAxis, yAxis):
    amplitude = np.abs(rfft(yAxis))
    frequency = rfftfreq(len(xAxis), (xAxis[1]-xAxis[0]))
    indices = find_peaks(amplitude)
    return np.round(frequency[indices[0]], 1)


def initial_time_graph(df1, df2):
    resize = alt.selection_interval(bind='scales')
    chart1 = alt.Chart(df1).mark_line().encode(
        x=alt.X('y:T', axis=alt.Axis(title='date', labels=False)),
        y=alt.Y('x:Q', axis=alt.Axis(title='value'))
    ).properties(
        width=600,
        height=300
    ).add_selection(
        resize
    )

    chart2 = alt.Chart(df2).mark_line().encode(
        x=alt.X('y:T', axis=alt.Axis(title='date', labels=False)),
        y=alt.Y('x:Q', axis=alt.Axis(title='value'))
    ).properties(
        width=600,
        height=300
    ).add_selection(
        resize
    )

    chart = alt.concat(chart1, chart2)
    return chart


# -------------------------------------- Fourier Transform on Audio ----------------------------------------------------
def audio_fourier_transform(audio_file, guitar, flute, piano, spectroCheckBox):
    column1, column2 = st.columns(2)
    with column1:
        st.audio(audio_file, format='audio/wav')  # displaying the audio
    obj = wave.open(audio_file, 'rb')
    sample_rate = obj.getframerate()      # number of samples per second
    n_samples = obj.getnframes()        # total number of samples in the whole audio
    duration = n_samples / sample_rate  # duration of the audio file
    signal_wave = obj.readframes(-1)      # amplitude of the sound

    signal_y_axis = np.frombuffer(signal_wave, dtype=np.int32)
    signal_x_axis = np.linspace(0, duration, len(signal_y_axis))
    with column1:
        if not spectroCheckBox:
            plotting(signal_x_axis[:1000], signal_y_axis[:1000])
        else:
            plot_spectro(audio_file.name)

    # returns complex numbers of the y axis in the data frame
    yf = rfft(signal_y_axis)
    # returns the frequency x axis after fourier transform
    xf = rfftfreq(len(signal_y_axis), (signal_x_axis[1]-signal_x_axis[0]))

    peaks = find_peaks(yf)   # computes peaks of the signal
    peaks_indeces = peaks[0]  # indeces of frequency with high peaks
    points_per_freq = len(xf) / (xf[-1])  # NOT UNDERSTANDABLE

    # with column1:
    #     plotting(xf,np.abs(yf))

    # these three lines determine the range that will be modified by the slider
    yf[:int(600*points_per_freq)] *= guitar
    yf[int(600*points_per_freq):int(3000*points_per_freq)] *= flute
    yf[int(3000*points_per_freq):] *= piano

    # returning the inverse transform after modifying it with sliders
    modified_signal = irfft(yf)
    tryyy = np.int32(modified_signal)

    write("example.wav", sample_rate, tryyy)
    with column2:
        st.audio("example.wav", format='audio/wav')
        if not spectroCheckBox:
            plotting(signal_x_axis[:1000], tryyy[:1000])
        else:
            plot_spectro("example.wav")
    # with column2:
    #     plotting(xf,np.abs(yf))


def uniform_audio_fourier_transform(audio_file, comp_1, comp_2, comp_3, comp_4, comp_5, comp_6, comp_7, comp_8, comp_9, comp_10, spectroCheckBox):
    column1, column2 = st.columns(2)
    with column1:
        st.audio(audio_file, format='audio/wav')  # displaying the audio

    obj = wave.open(audio_file, 'rb')
    sample_rate = obj.getframerate()      # number of samples per second
    n_samples = obj.getnframes()        # total number of samples in the whole audio
    duration = n_samples / sample_rate  # duration of the audio file
    signal_wave = obj.readframes(-1)      # amplitude of the sound

    signal_y_axis = np.frombuffer(signal_wave, dtype=np.int32)
    signal_x_axis = np.linspace(0, duration, len(signal_y_axis))
    with column1:
        if not spectroCheckBox:
            plotting(signal_x_axis[:1000], signal_y_axis[:1000])

        else:
            plot_spectro(audio_file.name)
    # returns complex numbers of the y axis in the data frame
    yf = rfft(signal_y_axis)
    # returns the frequency x axis after fourier transform
    xf = rfftfreq(len(signal_y_axis), (signal_x_axis[1]-signal_x_axis[0]))

    peaks = find_peaks(yf)   # computes peaks of the signal
    peaks_indeces = peaks[0]  # indeces of frequency with high peaks
    points_per_freq = len(xf) / (xf[-1])  # NOT UNDERSTANDABLE

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

    # returning the inverse transform after modifying it with sliders
    modified_signal = irfft(yf)
    tryyy = np.int32(modified_signal)

    write("example.wav", sample_rate, tryyy)
    with column2:
        st.audio("example.wav", format='audio/wav')
        if not spectroCheckBox:
            plotting(signal_x_axis[:1000], tryyy[:1000])
        else:
            plot_spectro("example.wav")

    df1 = pd.DataFrame({'x': signal_x_axis[:1000], 'y': signal_y_axis[:1000]})
    df2 = pd.DataFrame({'x': signal_x_axis[:1000], 'y': tryyy[:1000]})
    with column1:
        plot = st.altair_chart(initial_time_graph(df1[:100], df2[:100]))
        st.write(df1)
    if st.button(label="Play"):
        for i in range(0, 1000):
            # df1 = pd.DataFrame({'x':signal_x_axis[i:i+10], 'y':signal_y_axis[i:i+10]})
            # df2 = pd.DataFrame({'x':signal_x_axis[i:i+10], 'y':tryyy[i:i+10]})
            with column1:
                plot.altair_chart(initial_time_graph(
                    df1[i:i+100], df2[i:i+100]))
    # with column2:
    #     plotting(xf,np.abs(yf))


def vowel_triang_window(y, start, end, val, ppf):
    target = y[int(start*ppf):int(end*ppf)]
    if val == 0:
        window = -(signal.windows.triang(len(target))-1)
    elif val==1:
        return target
    else:
        window = val * signal.windows.triang(len(target))

    return [target[i]*window[i] for i in range(len(window))]


def vowel_audio_fourier_transform(file, er_vowel, a_vowel, iy_vowel, oo_vowel, uh_vowel, spectroCheckBox):
    column1, column2 = st.columns(2)
    with column1:
        st.audio(file, format='audio/wav')

    obj = wave.open(file, 'rb')
    sample_rate = obj.getframerate()      # number of samples per second
    n_samples = obj.getnframes()        # total number of samples in the whole audio
    duration = n_samples / sample_rate  # duration of the audio file
    signal_wave = obj.readframes(-1)      # amplitude of the sound

    signal_y_axis = np.frombuffer(signal_wave, dtype=np.int16)
    signal_x_axis = np.linspace(0, duration, len(signal_y_axis))

    with column1:
        if not spectroCheckBox:
            plotting(signal_x_axis[:1000], signal_y_axis[:1000])
        else:
            plot_spectro(file)

    yf = rfft(signal_y_axis)
    xf = rfftfreq(len(signal_y_axis), (signal_x_axis[1]-signal_x_axis[0]))

    points_per_freq = len(xf) / (xf[-1])
    # er vowel frequencies
    yf[int(440*points_per_freq):int(540*points_per_freq)
       ] = vowel_triang_window(yf, 440, 540, er_vowel, points_per_freq)
    yf[int(1300*points_per_freq):int(1400*points_per_freq)
       ] = vowel_triang_window(yf, 1300, 1400, er_vowel, points_per_freq)
    yf[int(1640*points_per_freq):int(1740*points_per_freq)
       ] = vowel_triang_window(yf, 1640, 1740, er_vowel, points_per_freq)
    # a vowel frequencies
    yf[int(680*points_per_freq):int(780*points_per_freq)
       ] = vowel_triang_window(yf, 680, 780, a_vowel, points_per_freq)
    yf[int(1040*points_per_freq):int(1140*points_per_freq)
       ] = vowel_triang_window(yf, 1040, 1140, a_vowel, points_per_freq)
    yf[int(2390*points_per_freq):int(2490*points_per_freq)
       ] = vowel_triang_window(yf, 2390, 2490, a_vowel, points_per_freq)
    # iy vowel frequencies
    yf[int(220*points_per_freq):int(320*points_per_freq)
       ] = vowel_triang_window(yf, 220, 320, iy_vowel, points_per_freq)
    yf[int(2240*points_per_freq):int(2340*points_per_freq)
       ] = vowel_triang_window(yf, 2240, 2340, iy_vowel, points_per_freq)
    yf[int(2960*points_per_freq):int(3060*points_per_freq)
       ] = vowel_triang_window(yf, 2960, 3060, iy_vowel, points_per_freq)
    # oo vowel frequencies
    yf[int(250*points_per_freq):int(350*points_per_freq)
       ] = vowel_triang_window(yf, 250, 350, oo_vowel, points_per_freq)
    yf[int(820*points_per_freq):int(920*points_per_freq)
       ] = vowel_triang_window(yf, 820, 920, oo_vowel, points_per_freq)
    yf[int(2360*points_per_freq):int(2460*points_per_freq)
       ] = vowel_triang_window(yf, 2360, 2460, oo_vowel, points_per_freq)
    # uh vowel frequencies
    yf[int(470*points_per_freq):int(570*points_per_freq)
       ] = vowel_triang_window(yf, 470, 570, uh_vowel, points_per_freq)
    yf[int(1140*points_per_freq):int(1240*points_per_freq)
       ] = vowel_triang_window(yf, 1140, 1240, uh_vowel, points_per_freq)
    yf[int(2340*points_per_freq):int(2440*points_per_freq)
       ] = vowel_triang_window(yf, 2340, 2440, uh_vowel, points_per_freq)
    modified_signal = irfft(yf)
    norm_modified = np.int16(modified_signal)

    write("vowel_modified.wav", sample_rate, norm_modified)

    with column2:
        st.audio("vowel_modified.wav", format='audio/wav')
        if not spectroCheckBox:
            plotting(signal_x_axis[:1000], norm_modified[:1000])
        else:
            plot_spectro("vowel_modified.wav")


def plotting(x_axis_fourier, fft_out):
    # Plotting audio Signal
    # figure, axis = plt.subplots()
    # plt.subplots_adjust(hspace=1)
    # axis.plot(x_axis_fourier,fft_out)
    figure = make_subplots(rows=2, cols=2, shared_yaxes=True)
    # figure.update_xaxes(matches='x')
    figure.add_trace(go.Scatter(y=fft_out, x=x_axis_fourier,
                     mode="lines", name="Signal"), row=1, col=1)
    #figure.add_trace(go.Scatter(y=fft_out,x=x_axis_fourier, mode="lines",name="transformed"), row=1, col=2)
    figure.update_xaxes(matches='x')

    st.plotly_chart(figure, use_container_width=True)


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


def pitch_modifier(audio_file, semitone, spectroCheckBox):
    """
    Modifies the pitch of a given audio file
    Arguments:
        audio_file: Audio file in .wav format
        semitone: half a tone higher or lower
        spectroCheckBox: if you want to view the signal as a spectogram
    """
    fr = 20
    column1, column2 = st.columns(2)
    with column1:
        st.audio(audio_file, format='audio/wav')  # displaying the audio
    obj = wave.open(audio_file, 'rb')
    sample_rate = obj.getframerate()      # number of samples per second
    n_samples = obj.getnframes()        # total number of samples in the whole audio
    duration = n_samples / sample_rate  # duration of the audio file
    signal_wave = obj.readframes(-1)      # amplitude of the sound

    signal_y_axis = np.frombuffer(signal_wave, dtype=np.int32)
    signal_x_axis = np.linspace(0, duration, len(signal_y_axis))
    with column1:
        if not spectroCheckBox:
            plotting(signal_x_axis[:1000], signal_y_axis[:1000])
        else:
            plot_spectro(audio_file.name)

    y_shifted = librosa.effects.pitch_shift(signal_y_axis.astype(
        float), sample_rate, n_steps=semitone)  # shifting the audio according to the semitone given

    # reconstructing the proccessed signal
    sf.write(file='example.wav', samplerate=sample_rate, data=y_shifted)

    y_normalized = np.int32(y_shifted)

    with column2:
        st.audio("example.wav", format='audio/wav')
        if not spectroCheckBox:
            plotting(signal_x_axis[:1000], y_normalized[:1000])
        else:
            plot_spectro("example.wav")
    # with column2:
    #     plotting(xf,np.abs(yf))

    # --------------------------------------------------------------- TESTING DYNAMIC GRAPHS ------------------------------------------------------------------
