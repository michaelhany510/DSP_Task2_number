# def fourier_for_audio:
    # sample_rate, amplitude = wav.read(uploaded_file)  # kam sample fl sec fl track,amplitude l data
    # # st.write(amplitude)
    # amplitude = np.frombuffer(amplitude, "int32")     # str code khd mn dof3t 4 - 3 ayam search
    # # st.write(amplitude)
    # duration = len(amplitude)/sample_rate
    # plotting(np.linspace(0,duration,len(amplitude)),amplitude)
   
# SAMPLE_RATE = 44100  # Hertz
# DURATION = 5  # Seconds
# fig = go.Figure()

# # uploaded_file = st.file_uploader(label="Uploading Signal", type = ['csv',".wav"])

# def generate_sine_wave(freq, sample_rate, duration):
#     x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
#     frequencies = x * freq
#     # 2pi because np.sin takes radians
#     y = np.sin((2 * np.pi) * frequencies)
#     return x, y

# def FT(signal,Fs,duration):

#     DURATION = duration
#     SAMPLE_RATE = int(Fs)
#     mixed_tone = signal
#     normalized_tone = np.int16((mixed_tone / mixed_tone.max()) * 32767)

#     plt.plot(normalized_tone[:1000])
#     plt.show()

#     # Remember SAMPLE_RATE = 44100 Hz is our playback rate
#     write("myUploadedSignal.wav", SAMPLE_RATE, normalized_tone)


#     # Number of samples in normalized_tone
#     N = int(SAMPLE_RATE * DURATION)

#     yf = rfft(normalized_tone)
#     xf = rfftfreq(N, 1 / SAMPLE_RATE)
#     print(xf)

#     plt.plot(xf, np.abs(yf))
#     plt.show()

#     # The maximum frequency is half the sample rate
#     points_per_freq = len(xf) / (SAMPLE_RATE / 2)

#     # Our target frequency is 4000 Hz
#     target_idx = int(points_per_freq * 10)
#     target_idx_2 = int(points_per_freq * 4000)
#     yf[target_idx - 1 : target_idx_2 + 2] = 0

#     plt.plot(xf, np.abs(yf))
#     plt.show()


#     new_sig = irfft(yf)

#     plt.plot(new_sig[:1000])
#     plt.show()

#     norm_new_sig = np.int16(new_sig * (32767 / new_sig.max()))

#     write("cleanUploaded04.wav", SAMPLE_RATE, norm_new_sig)

# from scipy.io.wavfile import read,write
# from IPython.display import Audio

# Fs, data = read('mixkit-retro-game-emergency-alarm-1000.wav')
# data = data[:,0]
# duration = data.shape[0]/Fs
# FT(data,Fs,duration)
# print('Sampling Frequency is', Fs)
# import scipy
# def getPeaksFrequencies(xAxis,yAxis):
    
#     amplitude = np.abs(scipy.fft.rfft(yAxis))
#     frequency = scipy.fft.rfftfreq(len(xAxis), (xAxis[1]-xAxis[0]))
#     indices = find_peaks(amplitude)
#     return frequency[indices[0]]

    

# def trial_fourier(dataframe,Fs,duration):

    
#     DURATION = duration
#     SAMPLE_RATE = int(Fs)

#     signal_y_axis = (dataframe.iloc[:,1]).to_numpy()
#     signal_x_axis = (dataframe.iloc[:,0]).to_numpy()


#     normalized_signal_y_axis = np.int16((signal_y_axis / signal_y_axis.max()) * 32767)

#     plt.plot(normalized_signal_y_axis[:1000])
#     plt.show()

#     # Remember SAMPLE_RATE = 44100 Hz is our playback rate
#     write("michael.wav", SAMPLE_RATE, normalized_signal_y_axis)

#     # Number of samples in normalized_signal_y_axis
#     number_of_samples = int(SAMPLE_RATE * DURATION)

#     yf = rfft(normalized_signal_y_axis)

#     xf = rfftfreq(number_of_samples, 1 / SAMPLE_RATE)
#     # peaks = find_peaks(yf, height=1)
#     # peaks_indeces = peaks[0]
#     # peaks_height = peaks[1]

#     # st.write(peaks_height["peak_heights"])

#     # peaks_psotion = signal_y_axis[peaks_indeces]
    
#     peaksPositions = np.round(getPeaksFrequencies(signal_x_axis,signal_y_axis),1)
    
#     print(peaksPositions)
    
#     points_per_freq = len(xf) / (SAMPLE_RATE / 2)
    
#     fig, axs = plt.subplots()
#     fig.set_size_inches(14,5)
#     plt.plot(xf, np.abs(yf))
#     plt.show()

#     target_idx = int(points_per_freq * (peaksPositions[0]-1))
#     target_idx_2 = int(points_per_freq * peaksPositions[0]+1)
#     yf[target_idx - 1 : target_idx_2 + 2] = 0

#     plt.plot(xf,np.abs(yf))
#     plt.show()


#     new_sig = irfft(yf)

#     # one_argument_plotting(new_sig[:1000])
#     plt.plot(new_sig[:1000])
#     plt.show()

#     norm_new_sig = np.int16(new_sig * (32767 / new_sig.max()))

#     write("cleanUploaded04.wav", SAMPLE_RATE, norm_new_sig)

#     norm_new_sig = np.int16(new_sig * (32767 / new_sig.max()))

#     write("cleanUploaded04.wav", SAMPLE_RATE, norm_new_sig)

#     plt.plot(xf, np.abs(yf))
#     # plt.scatter(peaks_indeces,peaks_hight, color='black' , marker="o" ,linestyle="")
#     st.plotly_chart(fig,use_container_width=True)
    
    
# df = pd.read_csv('Signal 6.csv')
# df = pd.DataFrame(df)
# trial_fourier(df,2000,1)


# freq_range_detec(norm_new_sig)
    # yf[int(2500*points_per_freq):] *= 0
    #tympani...  
    # deleting from 0 to 1000 eliminates drums
    #deleting from 0 to 3300 eleminates drums and biano
    #deleting from 1000 to end keeps just drums
    #keeping from 1000 to 3000 keeps pure biano and very litttle tympani
    #same for 1000 to 2900
    #keeping from 1000 to 2500 gives pure biano
    
#  ----------------------------------- JUST  A REFRENCE CODE TO HELP WHILE CREATING SLIDER ---------------------------------------------------------------
# def creating_sliders(names_list):

#     # Side note: we can change sliders colors and can customize sliders as well.
#     # names_list = [('Megzawy', 100),('Magdy', 150)]
#     columns = st.columns(len(names_list))
#     boundary = int(50)
#     sliders_values = []
#     sliders = {}

#     for index, tuple in enumerate(names_list): # ---> [ { 0, ('Megzawy', 100) } , { 1 , ('Magdy', 150) } ]
#         # st.write(index)
#         # st.write(i)
#         min_value = tuple[1] - boundary
#         max_value = tuple[1] + boundary
#         key = f'member{random.randint(0,10000000000)}'
#         with columns[index]:
#             sliders[f'slidergroup{key}'] = svs.vertical_slider(key=key, default_value=tuple[1], step=1, min_value=min_value, max_value=max_value)
#             if sliders[f'slidergroup{key}'] == None:
#                 sliders[f'slidergroup{key}'] = tuple[1]
#             sliders_values.append((tuple[0], sliders[f'slidergroup{key}']))
# names_list = [('A', 100),('B', 150),('C', 75),('D', 25),('E', 150),('F', 60),('G', 86),('H', 150),('E', 150),('G', 25),('K', 99),('L', 150),
#                 ('M', 150),('M', 55),('N', 150)]
# fn.creating_sliders(names_list)
