# Project Configuration Settings
import os

# Audio Processing Settings
SAMPLE_RATE = 22050
N_MFCC = 13
HOP_LENGTH = 512
N_FFT = 2048

# Model Settings
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# File Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_AUDIO_DIR = os.path.join(DATA_DIR, 'raw_audio')
SCREAM_DIR = os.path.join(RAW_AUDIO_DIR, 'scream_samples')
NON_SCREAM_DIR = os.path.join(RAW_AUDIO_DIR, 'non_scream_samples')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Alert Settings
CONFIDENCE_THRESHOLD = 0.7
ALERT_EMAIL = "your_email@example.com"
