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
st. set_page_config(layout="wide")
st.title("hello world")
sample_rate, tone = wavfile.read("dsp.wav")
# tone = tone[:, 0]
duration = tone.shape[0]/sample_rate
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
# Our target frequency is 4000 Hz
target_idx1 = int(points_per_freq * 27)
target_idx2 = int(points_per_freq * 227)
yf[target_idx1: target_idx2] *= 0

target_idx1 = int(points_per_freq * 480)
target_idx2 = int(points_per_freq * 680)
yf[target_idx1: target_idx2] *= 0

target_idx1 = int(points_per_freq * 1699)
target_idx2 = int(points_per_freq * 1899)
yf[target_idx1: target_idx2] *= 0

target_idx1 = int(points_per_freq * 2505)
target_idx2 = int(points_per_freq * 2705)
yf[target_idx1: target_idx2] *= 0

target_idx1 = int(points_per_freq * 3577)
target_idx2 = int(points_per_freq * 3777)
yf[target_idx1: target_idx2] *= 0

fig = plt.figure()
plt.plot(xf, np.abs(yf))
st.plotly_chart(fig, use_container_width=True)

modified = irfft(yf)

norm_modified = np.int16(modified*(32767/modified.max()))

wavfile.write("modified.wav", sample_rate, norm_modified)
