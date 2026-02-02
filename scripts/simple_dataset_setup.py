# Simple Dataset Setup Script
import os

# Define paths manually
base_dir = os.getcwd()
scream_dir = os.path.join(base_dir, 'data', 'raw_audio', 'scream_samples')
non_scream_dir = os.path.join(base_dir, 'data', 'raw_audio', 'non_scream_samples')

# Create directories
os.makedirs(scream_dir, exist_ok=True)
os.makedirs(non_scream_dir, exist_ok=True)

print("✅ Dataset directories created!")
print(f"📁 Scream samples: {scream_dir}")
print(f"📁 Non-scream samples: {non_scream_dir}")
print("\n📋 Next steps:")
print("1. Visit freesound.org")
print("2. Download audio samples")
print("3. Place scream samples in scream_samples folder")
print("4. Place non-scream samples in non_scream_samples folder")
