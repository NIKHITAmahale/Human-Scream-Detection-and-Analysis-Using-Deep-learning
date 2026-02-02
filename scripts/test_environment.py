# Test if all libraries are installed correctly
import sys
import importlib

required_packages = [
    'librosa', 'tensorflow', 'sklearn', 'numpy', 
    'pandas', 'matplotlib', 'scipy', 'seaborn', 'soundfile'
]

# Optional packages (for real-time audio)
optional_packages = ['sounddevice', 'pyaudio']

print("Testing Environment Setup...")
print("=" * 40)

for package in required_packages:
    try:
        importlib.import_module(package)
        print(f"✅ {package}: OK")
    except ImportError:
        print(f"❌ {package}: FAILED")

print("\nAudio Capture Libraries:")
for package in optional_packages:
    try:
        importlib.import_module(package)
        print(f"✅ {package}: OK")
    except ImportError:
        print(f"⚠️ {package}: Not available")

print("=" * 40)
print("Environment test completed!")
print("\nNote: sounddevice can replace pyaudio for real-time audio capture")
