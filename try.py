import numpy as np
from matplotlib import pyplot as plt
from scipy.io.wavfile import write
from scipy.fft import rfft, rfftfreq
from scipy.fft import irfft
import streamlit as st
import plotly.graph_objects as go
import plotly_express as px

SAMPLE_RATE = 44100  # Hertz
DURATION = 5  # Seconds
fig = go.Figure()

uploaded_file = st.file_uploader(label="Uploading Signal", type = ['csv',".wav"])

def fourier_plotting(x_axis_fourier,fft_out):
    # Plotting audio Signal
    figure, axis = plt.subplots()
    plt.subplots_adjust(hspace=1)
    axis.plot(x_axis_fourier,fft_out)
    st.plotly_chart(figure,use_container_width=True)

Fig = go.Figure()

def plotting (x,y):
  
    fig = px.line(x=x,y=y, labels={'x': 'Time (seconds)', 'y': 'Amplitude'})

    st.plotly_chart(fig, use_container_width=True)

def one_argument_plotting (y):
  
    fig = px.line(y=y, labels={'x': 'Time (seconds)', 'y': 'Amplitude'})

    st.plotly_chart(fig, use_container_width=True)

def generate_sine_wave(freq, sample_rate, duration):
    x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    frequencies = x * freq
    # 2pi because np.sin takes radians
    y = np.sin((2 * np.pi) * frequencies)
    return x, y

def FT(signal,Fs,duration):

    DURATION = duration
    SAMPLE_RATE = int(Fs)
    mixed_tone = signal
    normalized_tone = np.int16((mixed_tone / mixed_tone.max()) * 32767)

    one_argument_plotting(normalized_tone[:1000])
    plt.plot(normalized_tone[:1000])
    plt.show()

    # Remember SAMPLE_RATE = 44100 Hz is our playback rate
    write("myUploadedSignal.wav", SAMPLE_RATE, normalized_tone)


    # Number of samples in normalized_tone
    N = int(SAMPLE_RATE * DURATION)

    yf = rfft(normalized_tone)
    xf = rfftfreq(N, 1 / SAMPLE_RATE)
    print(xf)

    fourier_plotting(xf,np.abs(yf))
    plt.plot(xf, np.abs(yf))
    plt.show()

    # The maximum frequency is half the sample rate
    points_per_freq = len(xf) / (SAMPLE_RATE / 2)

    # Our target frequency is 4000 Hz
    target_idx = int(points_per_freq * 7500)
    target_idx_2 = int(points_per_freq * 16000)
    yf[target_idx - 1 : target_idx_2 + 2] = 0
   
    fourier_plotting(xf,np.abs(yf))
    plt.plot(xf, np.abs(yf))
    plt.show()


    new_sig = irfft(yf)

    one_argument_plotting(new_sig[:1000])
    plt.plot(new_sig[:1000])
    plt.show()

    norm_new_sig = np.int16(new_sig * (32767 / new_sig.max()))

    write("cleanUploaded.wav", SAMPLE_RATE, norm_new_sig)

from scipy.io.wavfile import read,write
from IPython.display import Audio

Fs, data = read('mixkit-retro-game-emergency-alarm-1000.wav')
data = data[:,0]
duration = data.shape[0]/Fs
FT(data,Fs,duration)
print('Sampling Frequency is', Fs)


# # Generate a 2 hertz sine wave that lasts for 5 seconds
# x, y = generate_sine_wave(2, SAMPLE_RATE, DURATION)
# plt.plot(x, y)
# plt.show()

# _, nice_tone = generate_sine_wave(400, SAMPLE_RATE, DURATION)
# _, noise_tone = generate_sine_wave(4000, SAMPLE_RATE, DURATION)
# noise_tone = noise_tone * 0.3

# mixed_tone = nice_tone + noise_tone
# normalized_tone = np.int16((mixed_tone / mixed_tone.max()) * 32767)

# plt.plot(normalized_tone[:1000])
# plt.show()

# # Remember SAMPLE_RATE = 44100 Hz is our playback rate
# write("mysinewave.wav", SAMPLE_RATE, normalized_tone)


# # Number of samples in normalized_tone
# N = SAMPLE_RATE * DURATION

# yf = rfft(normalized_tone)
# xf = rfftfreq(N, 1 / SAMPLE_RATE)
# print(xf)

# plt.plot(xf, np.abs(yf))
# plt.show()

# # The maximum frequency is half the sample rate
# points_per_freq = len(xf) / (SAMPLE_RATE / 2)

# # Our target frequency is 4000 Hz
# target_idx = int(points_per_freq * 4000)
# yf[target_idx - 1 : target_idx + 2] = 0

# plt.plot(xf, np.abs(yf))
# plt.show()


# new_sig = irfft(yf)

# plt.plot(new_sig[:1000])
# plt.show()

# norm_new_sig = np.int16(new_sig * (32767 / new_sig.max()))

# write("clean.wav", SAMPLE_RATE, norm_new_sig)
