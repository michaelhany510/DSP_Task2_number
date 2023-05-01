# Audio Equalizer app
[![Python Version](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/downloads/release/python-3100/)
[![Framework](https://img.shields.io/badge/framework-Streamlit-red)](https://streamlit.io/)
[![Dependencies](https://img.shields.io/badge/dependencies-up--to--date-brightgreen)](requirements.txt)

Python web application that processes different audio files.

## Description
This app has the following features:
- Upload an audio file to be equalized on different modes (uniform - music - medical).
- Play the audio before and after processing.
- Plot the audio and its spectrogram (dynamic plots are also supported).

## Dependencies
- **Python 3.10**
- **JavaScript**
- **HTML**
- **CSS**
### Used libraries
- soundfile
- wave
- numpy
- flask
- PIL
- werkzeug.utils
- os
- cv2
- matplotlib.pyplot
- plotly.express
- matplotlib.pyplot

## Preview
#### Uploading an Audio file.
![Audio](audio.png)

#### Spectrogram after filtering.
![Spec](spectrogram.png)

## Installation
1. Clone the repository `git clone https://github.com/michaelhany510/DSP_Task2_number`
2. Install the requirements `pip install -r requirements.txt`

## How to use
1. Run the app `streamlit run app.py`
2. Upload an audio file in .wav format.
3. Select the mode of equalization.
4. Play the audio before and after processing.
5. Plot the audio and its spectrogram.
