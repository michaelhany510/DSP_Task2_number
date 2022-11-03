# from time import time
# import numpy as np
# from scipy import fftpack


# def getFmaxMyCode(signalX,signalY):
#     # print(np.round(sig,2))
#     sig_fft = fftpack.fft(signalY)
#     # print(sig_fft)
#     amplitude = np.abs(sig_fft)
#     print(np.round(amplitude,2))
#     angle = np.angle(sig_fft)
#     # print(angle)

#     sampleFreq = fftpack.fftfreq(sig.size,d=timeStep)
#     print(sampleFreq)

#     amp_freq = np.array([amplitude,sampleFreq])
#     ampPosition = amp_freq[0,:].argmax()
#     peakFreq = amp_freq[1,ampPosition]
#     return peakFreq

# timeStep= 0.05

# timeVec = np.arange(0,10,timeStep)
# period = 5
# sig = (np.sin(2*np.pi*timeVec/period) + 0.25*np.random.randn(timeVec.size))
# #every 5*5 points it completes 1/4 a cycle
# #every 5*20 point it completes a cycle

# # print(np.round(sig,2))
# sig_fft = fftpack.fft(sig)
# # print(sig_fft)
# amplitude = np.abs(sig_fft)
# print(np.round(amplitude,2))
# angle = np.angle(sig_fft)
# # print(angle)

# sampleFreq = fftpack.fftfreq(sig.size,d=timeStep)
# print(sampleFreq)

# amp_freq = np.array([amplitude,sampleFreq])
# ampPosition = amp_freq[0,:].argmax()
# peakFreq = amp_freq[1,ampPosition]
# print(ampPosition)
# print(peakFreq)

# high_freq_fft = sig_fft.copy()
# high_freq_fft[np.abs(sampleFreq) > peakFreq] = 0

# print(high_freq_fft)

# filteredSignal = fftpack.ifft(high_freq_fft)
# print(np.abs(filteredSignal))


import numpy as np
from matplotlib import pyplot as plt

SAMPLE_RATE = 44100  # Hertz
DURATION = 5  # Seconds

def generate_sine_wave(freq, sample_rate, duration):
    x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    frequencies = x * freq
    # 2pi because np.sin takes radians
    y = np.sin((2 * np.pi) * frequencies)
    return x, y

# Generate a 2 hertz sine wave that lasts for 5 seconds
x, y = generate_sine_wave(2, SAMPLE_RATE, DURATION)
plt.plot(x, y)
plt.show()

_, nice_tone = generate_sine_wave(400, SAMPLE_RATE, DURATION)
_, noise_tone = generate_sine_wave(4000, SAMPLE_RATE, DURATION)
noise_tone = noise_tone * 0.3

mixed_tone = nice_tone + noise_tone
normalized_tone = np.int16((mixed_tone / mixed_tone.max()) * 32767)

plt.plot(normalized_tone[:1000])
plt.show()
from scipy.io.wavfile import write

# Remember SAMPLE_RATE = 44100 Hz is our playback rate
write("mysinewave.wav", SAMPLE_RATE, normalized_tone)

from scipy.fft import rfft, rfftfreq

# Number of samples in normalized_tone
N = SAMPLE_RATE * DURATION

yf = rfft(normalized_tone)
xf = rfftfreq(N, 1 / SAMPLE_RATE)
print(xf)

plt.plot(xf, np.abs(yf))
plt.show()

# The maximum frequency is half the sample rate
points_per_freq = len(xf) / (SAMPLE_RATE / 2)

# Our target frequency is 4000 Hz
target_idx = int(points_per_freq * 4000)
yf[target_idx - 1 : target_idx + 2] = 0

plt.plot(xf, np.abs(yf))
plt.show()

from scipy.fft import irfft

new_sig = irfft(yf)

plt.plot(new_sig[:1000])
plt.show()

norm_new_sig = np.int16(new_sig * (32767 / new_sig.max()))

write("clean.wav", SAMPLE_RATE, norm_new_sig)
