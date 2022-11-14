from IPython.display import Audio
from scipy.io import wavfile
import numpy as np
from matplotlib import pyplot as plt
from scipy.io.wavfile import write
from scipy.fft import rfft, rfftfreq
from scipy.fft import irfft
from scipy import signal
import streamlit as st
import plotly.graph_objects as go
from scipy.signal import find_peaks
import pandas as pd
import librosa
import soundfile
st. set_page_config(layout="wide")
st.title("Vowels EQ")
tone, sample_rate = soundfile.read(
    r"C:\Users\DELL\General\DSP Tasks\DSP_Task2_number\vowels\vvv.wav")

# tone = tone[:, 0]
duration = tone.shape[0]/sample_rate
# fig = plt.figure()
# plt.plot(np.linspace(0, np.ceil(duration), tone.shape[0]), tone)
# st.plotly_chart(fig, use_container_width=True)
# duration = int(duration)

# st.write(tone.shape[0])
# st.write(sample_rate)
# st.write(duration)
# st.write(duration*sample_rate)
# normalized = tone.astype(np.int16)
yf = rfft(tone)
xf = rfftfreq(tone.shape[0], 1/sample_rate)
# smaller = xf < 2000
# xf = xf*smaller
st.write("fft")
fig = plt.figure()
plt.plot(xf, np.abs(yf))
st.plotly_chart(fig, use_container_width=True)

points_per_freq = len(xf) / (sample_rate / 2)
st.write(points_per_freq)

yf[int(100*points_per_freq):int(500*points_per_freq)] *= 0
yf[int(9000*points_per_freq):int(14000*points_per_freq)] *= 0

st.write("after processing")
fig = plt.figure()
plt.plot(xf, np.abs(yf))
st.plotly_chart(fig, use_container_width=True)

modified = irfft(yf)

# norm_modified = modified.astype(np.int16)
soundfile.write("modified.wav", modified, sample_rate)

# fig = plt.figure()
# plt.plot(np.linspace(0, np.ceil(duration),
#          modified.shape[0]), modified)
# st.plotly_chart(fig, use_container_width=True)

# for m -monkey
tone, sample_rate = soundfile.read("modified.wav")
duration = tone.shape[0]/sample_rate
# normalized = np.int16((tone/tone.max())*32767)
yf = rfft(tone)
xf = rfftfreq(tone.shape[0], 1/sample_rate)
st.write("returning to frequency domain")
fig = plt.figure()
plt.plot(xf, np.abs(yf))
st.plotly_chart(fig, use_container_width=True)


# for s
#yf[int(2980*points_per_freq):int(3670*points_per_freq)] *= 0
# yf[int(3670*points_per_freq):int(4740*points_per_freq)]
# for v
#yf[int(140*points_per_freq):int(308*points_per_freq)] *= 10
#yf[int(320*points_per_freq):int(370*points_per_freq)] *= 10
# # for Z
# yf[int(130*points_per_freq):int(240*points_per_freq)] *= 0.2
# yf[int(350*points_per_freq):int(470*points_per_freq)] *= 0.2
# yf[int(260*points_per_freq):int(350*points_per_freq)] *= 0.2
# yf[int(8000*points_per_freq):int(14000*points_per_freq)] *= 0
# for e
# yf[int(342*points_per_freq):int(365*points_per_freq)] *= 0
# yf[int(310*points_per_freq):int(330*points_per_freq)] *= 0
# yf[int(170*points_per_freq):int(250*points_per_freq)]*= 0
# yf[int(685*points_per_freq):int(695*points_per_freq)]*= 0
# yf[int(702*points_per_freq):int(720*points_per_freq)] *= 0
# yf[int(840*points_per_freq):int(1100*points_per_freq)] *= 0
# for i:
# yf[int(280*points_per_freq):int(360*points_per_freq)] *= 0
# yf[int(210*points_per_freq):int(280*points_per_freq)] *= 0.1
# yf[int(130*points_per_freq):int(210*points_per_freq)] *= 0
# yf[int(340*points_per_freq):int(470*points_per_freq)] *= 0
# yf[int(3000*points_per_freq):int(3800*points_per_freq)] *= 0.5
# yf[int(5000*points_per_freq):int(6300*points_per_freq)] *= 0.5
# for t
# yf[int(3000*points_per_freq):int(11000*points_per_freq)] *= 4
# yf[int(1600*points_per_freq):int(3000*points_per_freq)] *= 4
# for m
# yf[int(150*points_per_freq):int(450*points_per_freq)] *= 0


# yf[[int(points_per_freq*0),int(points_per_freq*60),int(points_per_freq*170), int(points_per_freq*310),int(points_per_freq*600),int(points_per_freq*1000),int(points_per_freq*3000),int(points_per_freq*6000),int(points_per_freq*12000),int(points_per_freq*14000),int(points_per_freq*16000) ]]*=0
#beat (woman)
# yf[int(300*points_per_freq)] *= 0
# yf[int(2800*points_per_freq)] *= 0
# yf[int(3300*points_per_freq)] *= 0
