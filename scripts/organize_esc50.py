import pandas as pd
import shutil
import os

# 1. Update these two paths if needed
ESC50_META = r".\meta\esc50.csv"
ESC50_AUDIO = r".\external_audio"


# 2. Your project’s scream/non-scream folders
SCREAM_DIR = r"C:\Users\rnaik\HumanScreamDetection\data\raw_audio\scream_samples"
NON_SCREAM_DIR = r"C:\Users\rnaik\HumanScreamDetection\data\raw_audio\non_scream_samples"

# 3. Categories to include
scream_categories = {"crying_baby", "coughing", "sneezing", "breathing"}
non_scream_categories = {
    "clapping", "footsteps", "laughing", "brushing_teeth",
    "rain", "wind", "car_horn", "dog", "cat"
}

# 4. Load metadata
df = pd.read_csv(ESC50_META)

# 5. Ensure target folders exist
os.makedirs(SCREAM_DIR, exist_ok=True)
os.makedirs(NON_SCREAM_DIR, exist_ok=True)

# 6. Copy files based on category
for _, row in df.iterrows():
    src = os.path.join(ESC50_AUDIO, row["filename"])
    if row["category"] in scream_categories:
        dst = os.path.join(SCREAM_DIR, row["filename"])
    elif row["category"] in non_scream_categories:
        dst = os.path.join(NON_SCREAM_DIR, row["filename"])
    else:
        continue
    shutil.copy2(src, dst)

print("✅ ESC-50 files organized successfully!")
