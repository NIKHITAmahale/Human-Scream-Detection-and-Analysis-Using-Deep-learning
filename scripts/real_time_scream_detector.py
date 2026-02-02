import sounddevice as sd
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import joblib
import queue
import time
import os
import datetime
import csv
import winsound
from plyer import notification
from twilio.rest import Client

# Twilio config — Replace with your actual credentials
account_sid = 'ACdd39bef7f5a4c3f5701b8d1db9babe34'
auth_token = '11d2bc4f63afe309f7062daea729a23d'
twilio_client = Client(account_sid, auth_token)
twilio_from_number = '+1 667 327 1698'
twilio_to_number = '+919880049036'

def send_sms_alert(message):
    try:
        twilio_client.messages.create(
            body=message,
            from_=twilio_from_number,
            to=twilio_to_number
        )
        print("SMS alert sent")
    except Exception as e:
        print(f"Failed to send SMS alert: {e}")

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

model = load_model('final_scream_model.h5')
scaler = joblib.load('data/processed/scaler.pkl')

duration = 1.0
sample_rate = 22050
buffer_size = int(duration * sample_rate)
hop_size = int(buffer_size / 2)
q = queue.Queue()

def audio_callback(indata, frames, time_, status):
    if status:
        print(status)
    q.put(indata[:, 0].copy())

stream = sd.InputStream(channels=1, samplerate=sample_rate, callback=audio_callback, blocksize=hop_size)
stream.start()

log_file = 'scream_detection_log.csv'
if not os.path.isfile(log_file):
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Probability'])

threshold = 0.4
consec_threshold = 3
consecutive_count = 0

def process_audio_real_time():
    global consecutive_count
    buffer = np.zeros(buffer_size, dtype=np.float32)
    print("Listening for screams... Press Ctrl+C to stop.")

    while True:
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
                print(f"🚨 Scream detected! Probability: {prob:.2f}")

                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(log_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([timestamp, f"{prob:.2f}"])

                winsound.Beep(1000, 500)

                notification.notify(
                    title='Scream Detector Alert',
                    message='🚨 Scream detected!',
                    timeout=5
                )

                send_sms_alert(f"Scream detected! Probability: {prob:.2f}")

        else:
            if consecutive_count > 0:
                print(f"Scream detection streak interrupted at count {consecutive_count}")
            consecutive_count = 0
            print(f"Probability: {prob:.2f}")

        time.sleep(hop_size / sample_rate)

try:
    process_audio_real_time()
except KeyboardInterrupt:
    print("Stopping real-time detection...")
    stream.stop()
    stream.close()
