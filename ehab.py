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

from scipy.io import wavfile
from IPython.display import Audio
st. set_page_config(layout="wide")
st.title("Vowels EQ")
sample_rate, tone = wavfile.read("flat_back.wav")


def vowel_triang_window(y, start, end, val):
    target = y[int(start*points_per_freq):int(end*points_per_freq)]
    if val == 0:
        window = -(signal.windows.triang(len(target))-1)
    else:
        window = val * signal.windows.triang(len(target))

    return [target[i]*window[i] for i in range(len(window))]


# tone = tone[:, 0]
duration = tone.shape[0]/sample_rate
fig = plt.figure()
plt.plot(np.linspace(0, np.ceil(duration), tone.shape[0]), tone)
st.plotly_chart(fig, use_container_width=True)
# duration = int(duration)

st.write(tone.shape[0])
st.write(sample_rate)
st.write(duration)
st.write(duration*sample_rate)
normalized = np.int16((tone/tone.max())*32767)
yf = rfft(normalized)
xf = rfftfreq(tone.shape[0], 1/sample_rate)
# smaller = xf < 2000
# xf = xf*smaller

fig = plt.figure()
plt.plot(xf, np.abs(yf))
st.plotly_chart(fig, use_container_width=True)

points_per_freq = len(xf) / (sample_rate / 2)
st.write(points_per_freq)

yf[int(460*points_per_freq):int(860*points_per_freq)
   ] = vowel_triang_window(yf, 460, 860, 0)
yf[int(1520*points_per_freq):int(1920*points_per_freq)
   ] = vowel_triang_window(yf, 1520, 1920, 0)
yf[int(2210*points_per_freq):int(2610*points_per_freq)
   ] = vowel_triang_window(yf, 2210, 2610, 0)
yf[int(3150*points_per_freq):int(3550*points_per_freq)
   ] = vowel_triang_window(yf, 3150, 3550, 0)
yf[int(3650*points_per_freq):int(4050*points_per_freq)
   ] = vowel_triang_window(yf, 3650, 4050, 0)


fig = plt.figure()
plt.plot(xf, np.abs(yf))
st.plotly_chart(fig, use_container_width=True)

modified = irfft(yf)

norm_modified = np.int16(modified*(32767/modified.max()))
wavfile.write("modified.wav", sample_rate, norm_modified)

fig = plt.figure()
plt.plot(np.linspace(0, np.ceil(duration),
         norm_modified.shape[0]), norm_modified)
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

# 50:350
# 500:1200
# 2500:4500
