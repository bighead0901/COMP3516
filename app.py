from flask import Flask, request
import keras
from keras.models import load_model
import numpy as np
import librosa
from numpy import correlate
from scipy.io import wavfile
from scipy.signal import butter, lfilter, correlate
import os
from pydub import AudioSegment
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioSegmentation as aS
import noisereduce as nr
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import pyaudio

app = Flask(__name__)
directory = os.path.abspath(os.curdir)


def process_file(filename):
    if ' ' in filename:
        print(f"Skipping file {filename} due to spaces in filename")
        return None
    elif filename.endswith('.wav') or filename.endswith('.mp3'):
        try:
            # Load the audio file and convert it to a numpy array
            file_path = os.path.join(directory, filename)
            audio = AudioSegment.from_file(file_path)
            audio_samples = np.array(audio.get_array_of_samples())
            if audio.channels == 2:
                audio_samples = audio_samples.reshape((-1, 2))
                audio_samples = audio_samples.mean(axis=1)  # Use only one channel
            sample_rate = audio.frame_rate

            # Apply a bandpass filter
            lowcut = 300
            highcut = 3400
            filtered_audio = butter_bandpass_filter(audio_samples, lowcut, highcut, sample_rate)

            # Apply an autocorrelation filter
            autocorrelated_audio = correlate(filtered_audio, filtered_audio)

            features = extract_new_features(autocorrelated_audio, sample_rate)

            # Code  Gender  VoiceType
            # 0       M         H
            # 1       M         R
            # 2       F         H
            # 3       F         R

            # Determine gender and voice type based on filename
            code = 0
            if filename.startswith('ch1') or filename.startswith('ch2'):
                code = 2
            elif filename.startswith('ch3'):
                code = 0
            elif filename.startswith('rob_ch1'):
                code = 1
            elif filename.startswith('rob_ch2') or filename.startswith('rob_ch3'):
                code = 3

            # Add the features to the global features_list
            features['code'] = code

            return features
        
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            return None
    else:
        print(f"Skipping file {filename} due to unsupported format")
        return None

def predict(file_list=None, model_path=None, audio_data=None):
    # Load the pre-trained model
    model = load_model(model_path)

    # Process the audio files or audio data and store the features in a DataFrame
    features_list = []
    if audio_data is not None:
        features = process_audio(audio_data[0], audio_data[1])
        if features is not None:
            features_list.append(features)
    else:
        for file in file_list:
            features = process_file(file)
            if features is not None:
                features_list.append(features)

    features_df = pd.DataFrame(features_list)

    # Normalize and encode the features
    features_df = normalize_features(features_df)
    features_df = encode_categorical_variables(features_df)

    # Get the input features for the model
    X = features_df
    if audio_data is None:
        X = features_df.drop(columns=['code'])

    # Make predictions using the model
    result = model.predict(X)

    return np.argmax(result, axis=1)

def normalize_features(features_df):
    features_to_scale = ['pitch', 'spectral_centroid', 'spectral_contrast', 'spectral_rolloff', 'zero_crossing_rate', 'rms']
    scaler = StandardScaler()
    features_df[features_to_scale] = scaler.fit_transform(features_df[features_to_scale])
    return features_df

def encode_categorical_variables(features_df):
    # Expand the mfcc and chroma_stft columns
    features_df = expand_column(features_df, 'mfcc')
    features_df = expand_column(features_df, 'chroma_stft')

    return features_df

def expand_column(df, column_name):
    expanded_df = pd.DataFrame(df[column_name].tolist(), index=df.index)
    expanded_df.columns = [f"{column_name}_{i}" for i in range(len(expanded_df.columns))]
    return df.drop(columns=[column_name]).join(expanded_df)

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
    spectral_contrast = librosa.feature.spectral_contrast(y=audio_samples, sr=sample_rate, fmin=100)    
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

def detect_multiple_speakers(audio, threshold=2):
    buffer = audio.astype(np.float32) / np.iinfo(np.int16).max

    if buffer.size == 0:
        print("Warning: Empty signal. Skipping speaker diarization.")
        return False

    segments, speaker_indices, spk_info = aS.speaker_diarization(buffer, n_speakers=-1)
    
    # Count unique speaker indices
    unique_speaker_count = len(np.unique(speaker_indices))

    # Return True if more than one speaker found, False otherwise
    return unique_speaker_count >= threshold

def process_audio(audio_samples, sample_rate):
    # Denoise audio using the noisereduce library
    denoised_audio = nr.reduce_noise(y=audio_samples, sr=sample_rate).astype(np.int16)

    # Check if there are multiple speakers
    if detect_multiple_speakers(denoised_audio, sample_rate):
        print('Multiple Speaking Voices in the Background')
        return None

    lowcut = 300
    highcut = 3400
    filtered_audio = butter_bandpass_filter(denoised_audio, lowcut, highcut, sample_rate)

    # Apply an autocorrelation filter
    autocorrelated_audio = correlate(filtered_audio, filtered_audio)

    # Extract features using the librosa library
    return extract_new_features(autocorrelated_audio, sample_rate)

def predict_gender_and_robotic(model_file):
    #path = os.path.join(os.path.abspath(os.curdir), "recording-01c4e23d-af8b-4586-a154-a2a2420d90c6.wav") 
    path = os.path.join(os.path.abspath(os.curdir), "file.m4a")
    print(path)
    audio = AudioSegment.from_file(path)
    
    p = pyaudio.PyAudio()
    stream = p.open(format =
                p.get_format_from_width(audio.sample_width),
                channels = audio.channels,
                rate = audio.frame_rate,
                output = True)
    i = 0
    data = audio[:1024]._data
    while data:
        stream.write(data)
        i += 1024
        data = audio[i:i + 1024]._data

    stream.close()    
    p.terminate()

    audio_data = np.frombuffer(b''.join(data), dtype=np.int16)
    #print(audio_samples == audio_data)

    predictions = predict(model_path=model_file, audio_data=(audio_data, audio.frame_rate))  
    return predictions

    """
    # Load the model from the specified file path
    model = keras.models.load_model(model_file)

    audio_samples = np.array(audio.get_array_of_samples())

    #default number of channels = 2 in sampling
    print(audio.channels)
    channels = audio.channels
    if (channels == 2):
        audio_samples = audio_samples.reshape((-1, 2))
        audio_samples = audio_samples.mean(axis=1)  # Use only one channel
    
    #default sr 44100, but after tranferring it changes
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
    return gender_prediction, robotic_prediction"""

@app.route('/api/audio', methods=['GET', 'POST'])
def get_score():
    if request.method == 'POST':
        length = request.content_length
        content_type = request.content_type
        data = request.get_data()
        model_file = os.path.join(os.path.abspath(os.curdir), "model.h5")
        print(f"""Content Type is  {content_type} \n length is {length}""")
        with open("file.m4a","wb") as file:
            file.write(data)

        #print(data)
        result = predict_gender_and_robotic(model_file)
        return f"""Content Type is  {content_type} and data is {data} \n length is {length}\n result is {result}"""
    elif request.method == 'GET':
        return 'get method received'