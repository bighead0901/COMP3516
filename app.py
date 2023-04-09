from flask import Flask, request
import keras
import numpy as np
import librosa
from numpy import correlate
from scipy.io import wavfile
from scipy.signal import butter, lfilter, correlate
import os
import base64
from pydub import AudioSegment

app = Flask(__name__)

def predict(model, new_features):
    gender_prediction = model.predict_classes(np.array([new_features]))
    robotic_prediction = model.predict(np.array([new_features]))

    gender = "Male" if gender_prediction[0] == 0 else "Female"
    robotic = "Non-robotic" if robotic_prediction[0] < 0.5 else "Robotic"

    return gender, robotic

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def extract_new_features(audio_samples, sample_rate):
    # Extract features as in process_file
    pitch = librosa.piptrack(y=audio_samples, sr=sample_rate, fmin=80, fmax=400)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_samples, sr=sample_rate)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio_samples, sr=sample_rate)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_samples, sr=sample_rate)
    mfcc = librosa.feature.mfcc(y=audio_samples, sr=sample_rate)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio_samples)
    chroma_stft = librosa.feature.chroma_stft(y=audio_samples, sr=sample_rate)
    rms = librosa.feature.rms(y=audio_samples)

    # Return the features
    return {
        'pitch': np.mean(pitch),
        'spectral_centroid': np.mean(spectral_centroid),
        'spectral_contrast': np.mean(spectral_contrast),
        'spectral_rolloff': np.mean(spectral_rolloff),
        'mfcc': np.mean(mfcc, axis=1),
        'zero_crossing_rate': np.mean(zero_crossing_rate),
        'chroma_stft': np.mean(chroma_stft, axis=1),
        'rms': np.mean(rms)
    }

def process_audio(audio_samples, sample_rate):
    # Clear background noise, music, or other weird sound
    # (IMPLEMENTATION OF NOISE REDUCTION AS NEEDED)

    # Check if there are multiple speakers
    '''
    if detect_multiple_speakers(audio_samples, sample_rate):
        print('Multiple Speaking Voices in the Background')
        return None
    '''

    # Apply a bandpass filter
    lowcut = 300
    highcut = 3400
    filtered_audio = butter_bandpass_filter(audio_samples, lowcut, highcut, sample_rate)

    # Apply an autocorrelation filter
    autocorrelated_audio = correlate(filtered_audio, filtered_audio)

    # Extract features
    features = extract_new_features(autocorrelated_audio, sample_rate)
    return features

def predict_gender_and_robotic(model_file, audio_data):
    # Load the model from the specified file path
    model = keras.models.load_model(model_file)

    #path = os.path.join(os.path.abspath(os.curdir), "recording-01c4e23d-af8b-4586-a154-a2a2420d90c6.wav") 
    path = os.path.join(os.path.abspath(os.curdir), "file.m4a")
    print(path)
    audio = AudioSegment.from_file(path)
    audio_samples = np.array(audio.get_array_of_samples())

    #default number of channels = 2, but after tranferring it changes
    print(audio.channels)
    channels = audio.channels
    if (channels == 2):
        audio_samples = audio_samples.reshape((-1, 2))
        audio_samples = audio_samples.mean(axis=1)  # Use only one channel

    #default sr 44100, but after tranferring it changes
    print(audio.frame_rate)
    sample_rate=audio.frame_rate

    # Process the recorded audio
    processed_audio = process_audio(audio_samples, sample_rate)

    # Extract the features from the processed audio
    new_features = extract_new_features(processed_audio, sample_rate)

    # Make gender and robotic predictions using the loaded model
    gender_prediction, robotic_prediction = predict(model, new_features)

    # Print the predictions
    print("Gender prediction:", gender_prediction)
    print("Robotic prediction:", robotic_prediction)
    return gender_prediction, robotic_prediction

@app.route('/api/audio', methods=['GET', 'POST'])
def get_score():
    if request.method == 'POST':
        length = request.content_length
        content_type = request.content_type
        data = request.get_data()
        model_file = os.path.join(os.path.abspath(os.curdir), "human_robot_voice.h5")

        with open("file.m4a","wb") as file:
            file.write(data)

        #print(data)
        result = predict_gender_and_robotic(model_file, data)
        return f"""Content Type is  {content_type} and data is {data} \n length is {length}\n result is {result}"""
    elif request.method == 'GET':
        return 'get method received'