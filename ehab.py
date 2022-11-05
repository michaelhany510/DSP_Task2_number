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
st.title("hello world")
sample_rate, tone = wavfile.read("piano.wav")
tone = tone[:, 0]
duration = tone.shape[0]/sample_rate
st.write(duration)
normalized = np.int16((tone/tone.max())*32767)
yf = rfft(normalized)
xf = rfftfreq(tone.shape[0], 1/sample_rate)
smaller = xf < 2000
xf = xf*smaller
fig = plt.figure()
plt.plot(xf, np.abs(yf))
st.plotly_chart(fig, use_container_width=True)
scaler = yf > 2000000000
yf = yf*scaler
plt.plot(xf, np.abs(yf))
st.plotly_chart(fig, use_container_width=True)
