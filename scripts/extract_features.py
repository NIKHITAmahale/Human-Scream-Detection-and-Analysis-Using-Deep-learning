import librosa
import numpy as np

def extract_features(file_path, sample_rate=22050):
    y, sr = librosa.load(file_path, sr=sample_rate, duration=5.0)
    
    # Mel-Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel)
    
    # MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    
    # Tonnetz
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    
    # RMS Energy
    rms = librosa.feature.rms(y=y)
    
    # Spectral Bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    
    # Combine feature stats (mean and std)
    features = np.concatenate([
        np.mean(mfccs, axis=1), np.std(mfccs, axis=1),
        np.mean(mel_db, axis=1), np.std(mel_db, axis=1),
        np.mean(contrast, axis=1), np.std(contrast, axis=1),
        np.mean(tonnetz, axis=1), np.std(tonnetz, axis=1),
        np.mean(rms, axis=1), np.std(rms, axis=1),
        np.mean(spec_bw, axis=1), np.std(spec_bw, axis=1)
    ])
    
    return features
   