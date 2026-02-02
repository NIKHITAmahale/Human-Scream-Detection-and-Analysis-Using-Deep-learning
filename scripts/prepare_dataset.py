import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import json
from audio_augment import augment_audio  # your augmentation function


def extract_features(y, sr):
    """Extract MFCC, chroma, and spectral features for one audio clip."""
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=512)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std  = np.std(mfccs, axis=1)
    chroma    = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    centroid  = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    rolloff   = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zcr       = np.mean(librosa.feature.zero_crossing_rate(y))

    feature_vector = np.concatenate(
        [mfcc_mean, mfcc_std, chroma, [centroid, rolloff, zcr]]
    )
    return feature_vector


def load_with_repeats(folder_path, label, n_repeats=1, augment=False):
    """
    Load each file n_repeats times (with optional augmentation) and extract features.
    This is used to oversample screams with multiple augmented versions.
    """
    features, labels = [], []
    files = [f for f in os.listdir(folder_path)
             if f.lower().endswith(('.wav', '.mp3', '.flac'))]

    for fname in files:
        path = os.path.join(folder_path, fname)
        for _ in range(n_repeats):
            try:
                y, sr = librosa.load(path, sr=22050, duration=5.0)
                if augment:
                    y = augment_audio(y, sr)

                feature_vector = extract_features(y, sr)
                features.append(feature_vector)
                labels.append(label)
                print(f"✅ {fname}")
            except Exception as e:
                print(f"❌ {fname}: {e}")

    return features, labels


def prepare_dataset():
    base = os.getcwd()
    scream_dir    = os.path.join(base, 'data', 'raw_audio', 'scream_samples')
    nonscream_dir = os.path.join(base, 'data', 'raw_audio', 'non_scream_samples')
    processed     = os.path.join(base, 'data', 'processed')
    os.makedirs(processed, exist_ok=True)

    # Oversample screams: more repeats with augmentation
    print("Processing screams with stronger augmentation...")
    f1, l1 = load_with_repeats(
        scream_dir,
        label=1,
        n_repeats=3,      # try 2 or 3; tune if needed
        augment=True
    )

    # Non-screams: single pass with light augmentation
    print("Processing non-screams with augmentation...")
    f0, l0 = load_with_repeats(
        nonscream_dir,
        label=0,
        n_repeats=1,
        augment=True
    )

    X = np.array(f1 + f0)
    y = np.array(l1 + l0)

    print(f"Total samples: {len(X)}, Features per sample: {X.shape[1]}")

    # Stratified split to keep class ratio in train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Standardize features
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Save processed arrays
    np.save(os.path.join(processed, 'X_train.npy'), X_train_s)
    np.save(os.path.join(processed, 'X_test.npy'),  X_test_s)
    np.save(os.path.join(processed, 'y_train.npy'), y_train)
    np.save(os.path.join(processed, 'y_test.npy'),  y_test)
    joblib.dump(scaler, os.path.join(processed, 'scaler.pkl'))

    # Save summary info
    info = {
        'total_samples': int(len(X)),
        'scream_samples': int(len(f1)),
        'non_scream_samples': int(len(f0)),
        'feature_dim': int(X.shape[1]),
        'train_samples': int(len(X_train_s)),
        'test_samples': int(len(X_test_s))
    }
    with open(os.path.join(processed, 'dataset_info.json'), 'w') as jf:
        json.dump(info, jf, indent=2)

    print("🎯 Dataset prepared! Files saved in data\\processed")
    print("Dataset info:", info)


if __name__ == "__main__":
    prepare_dataset()
