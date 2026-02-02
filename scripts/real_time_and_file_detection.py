import streamlit as st
import sounddevice as sd
import numpy as np
import queue
import time
import datetime
import csv
import os
import librosa
from tensorflow.keras.models import load_model
import joblib
from plyer import notification
import threading

# Load model and scaler (adjust paths accordingly)
model = load_model('final_scream_model.h5')
scaler = joblib.load('data/processed/scaler.pkl')

# Constants
duration = 1.0
sample_rate = 22050
buffer_size = int(duration * sample_rate)
hop_size = int(buffer_size / 2)
threshold = 0.4
consec_threshold = 3

log_file = 'scream_detection_log.csv'

# Initialize CSV log if missing
if not os.path.isfile(log_file):
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Probability'])

q = queue.Queue()
consecutive_count = 0
buffer = np.zeros(buffer_size, dtype=np.float32)

# Audio callback for streaming mic input
def audio_callback(indata, frames, time_, status):
    if status:
        print(status)
    q.put(indata[:, 0].copy())

# Feature extraction function
def extract_features(y, sr=22050):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std  = np.std(mfccs, axis=1)
    chroma     = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    centroid   = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    rolloff    = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zcr        = np.mean(librosa.feature.zero_crossing_rate(y))
    features = np.concatenate([mfccs_mean, mfccs_std, chroma, [centroid, rolloff, zcr]])
    return features

# Real-time detection function running in separate thread
def real_time_detection():
    global consecutive_count, buffer

    stream = sd.InputStream(channels=1, samplerate=sample_rate, callback=audio_callback, blocksize=hop_size)
    stream.start()
    st.session_state['running'] = True

    try:
        while st.session_state.get('running', False):
            audio_chunk = q.get()
            buffer[:-hop_size] = buffer[hop_size:]
            buffer[-hop_size:] = audio_chunk

            features = extract_features(buffer, sample_rate)
            features_scaled = scaler.transform(features.reshape(1, -1))
            features_scaled = features_scaled[..., np.newaxis]

            prob = model.predict(features_scaled)[0][0]

            if prob >= threshold:
                consecutive_count += 1
                if consecutive_count >= consec_threshold:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open(log_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([timestamp, f"{prob:.2f}"])

                    notification.notify(
                        title='Scream Detector Alert',
                        message='🚨 Scream detected!',
                        timeout=5
                    )

                    st.session_state['last_detection'] = f"Scream detected at {timestamp} (prob={prob:.2f})"
                    consecutive_count = 0  # reset after alert
            else:
                consecutive_count = 0

            # Update latest probability for UI
            st.session_state['last_prob'] = prob
            time.sleep(hop_size / sample_rate)
    except Exception as e:
        print("Real-time detection stopped:", e)
    finally:
        stream.stop()
        stream.close()
        st.session_state['running'] = False

# Streamlit app UI
st.title("Real-time & File Upload Scream Detection")

# Start/Stop real-time detection buttons
if 'running' not in st.session_state:
    st.session_state['running'] = False

col1, col2 = st.columns(2)
with col1:
    if st.button("Start Real-time Detection") and not st.session_state['running']:
        st.info("Starting real-time detection...")
        detection_thread = threading.Thread(target=real_time_detection, daemon=True)
        detection_thread.start()
with col2:
    if st.button("Stop Real-time Detection") and st.session_state['running']:
        st.session_state['running'] = False
        st.info("Stopping real-time detection...")

# Show last detection info and probability
if 'last_detection' in st.session_state:
    st.success(st.session_state['last_detection'])
if 'last_prob' in st.session_state:
    st.write(f"Last detection probability: {st.session_state['last_prob']:.2f}")

st.markdown("---")

# File upload detection
uploaded_file = st.file_uploader("Upload an audio file for detection", type=["wav", "mp3", "flac"])

if uploaded_file is not None:
    y, sr = librosa.load(uploaded_file, sr=22050, duration=1.0)
    features = extract_features(y, sr)
    features_scaled = scaler.transform(features.reshape(1, -1))
    features_scaled = features_scaled[..., np.newaxis]
    prob = model.predict(features_scaled)[0][0]
    st.write(f"Scream probability: {prob:.2f}")
    if prob >= threshold:
        st.success("Scream detected!")
    else:
        st.info("No scream detected.")

st.markdown("---")

# Display latest logs
st.write("### Latest Detection Logs")
if os.path.isfile(log_file):
    import pandas as pd
    df = pd.read_csv(log_file, parse_dates=['Timestamp'])
    st.dataframe(df.tail(10).reset_index(drop=True))
else:
    st.info("No detection logs found yet.")
