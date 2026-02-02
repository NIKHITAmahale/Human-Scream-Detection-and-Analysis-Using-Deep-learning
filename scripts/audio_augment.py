from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

augmenter = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-8, max_semitones=8, p=0.5),
    Shift(p=0.5),  # Correct usage without min_fraction or max_fraction
])

def augment_audio(samples, sample_rate):
    return augmenter(samples=samples, sample_rate=sample_rate)
