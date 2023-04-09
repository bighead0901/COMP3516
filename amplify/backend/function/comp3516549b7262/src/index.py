
import json
import os
import numpy as np
import librosa
import librosa.display
import pandas as pd
import soundfile as sf
import time
import pickle
from numpy import correlate
from scipy.io import wavfile
from scipy.signal import butter, lfilter, correlate
import matplotlib.pyplot as plt
from pydub import AudioSegment
from multiprocessing import Pool, Manager
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def handler(event, context):
  print('received event:')
  print(event)
  
  return {
      'statusCode': 200,
      'headers': {
          'Access-Control-Allow-Headers': '*',
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
      },
      'body': json.dumps('Hello from your new Amplify Python lambda!')
  }

def predict_gender_and_robotic(model_file):
  # Load the model
  model = load_model(model_file)

  while True:
      user_input = input("Press enter to record audio or type 'stop' to exit: ")
      if user_input.lower() == 'stop':
          break

      # Record a new sound sample
      audio_data, sample_rate = record_audio()

      # Process the recorded audio
      processed_audio = process_audio(audio_data, sample_rate)
      if processed_audio is None:
          continue

      # Extract the features from the processed audio
      new_features = extract_new_features(processed_audio, sample_rate)

      # Make gender and robotic predictions using the loaded model
      gender_prediction, robotic_prediction = predict(model, new_features)

      # Print the predictions
      print("Gender prediction:", gender_prediction)
      print("Robotic prediction:", robotic_prediction)