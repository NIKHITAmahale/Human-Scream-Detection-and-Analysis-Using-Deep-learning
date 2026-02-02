# Dataset Collection Script for Human Scream Detection
import os
import requests
import librosa
import numpy as np
from config.config import SCREAM_DIR, NON_SCREAM_DIR
import pandas as pd

def create_sample_urls():
    """
    URLs for sample audio files (you'll need to expand this)
    """
    sample_data = {
        'scream_urls': [
            # Add actual URLs to scream audio files
            # Example: 'https://freesound.org/data/previews/123/123456_12345-hq.mp3'
        ],
        'non_scream_urls': [
            # Add URLs to non-scream audio files (speech, music, ambient sounds)
            # Example: 'https://freesound.org/data/previews/456/456789_67890-hq.mp3'
        ]
    }
    return sample_data

def download_audio_file(url, save_path):
    """
    Download audio file from URL
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"✅ Downloaded: {save_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")
        return False

def validate_audio_file(file_path):
    """
    Validate that audio file can be processed
    """
    try:
        y, sr = librosa.load(file_path, duration=5.0)  # Load first 5 seconds
        if len(y) > 0:
            print(f"✅ Valid audio: {file_path}")
            return True
        else:
            print(f"⚠️ Empty audio: {file_path}")
            return False
    except Exception as e:
        print(f"❌ Invalid audio {file_path}: {e}")
        return False

def collect_dataset():
    """
    Main dataset collection function
    """
    print("Starting dataset collection...")
    
    # Create directories if they don't exist
    os.makedirs(SCREAM_DIR, exist_ok=True)
    os.makedirs(NON_SCREAM_DIR, exist_ok=True)
    
    # For now, just create placeholder for manual collection
    print(f"📁 Scream samples directory: {SCREAM_DIR}")
    print(f"📁 Non-scream samples directory: {NON_SCREAM_DIR}")
    
    print("\n📋 MANUAL COLLECTION INSTRUCTIONS:")
    print("1. Download scream audio samples and place in:", SCREAM_DIR)
    print("2. Download non-scream samples and place in:", NON_SCREAM_DIR)
    print("3. Aim for at least 100 samples of each type")
    print("4. Supported formats: .wav, .mp3, .flac")

if __name__ == "__main__":
    collect_dataset()
