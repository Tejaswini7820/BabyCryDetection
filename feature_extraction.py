import librosa
import numpy as np

SAMPLE_RATE = 22050
DURATION = 5
MAX_LEN = SAMPLE_RATE * DURATION
N_MFCC = 40

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    # Pad / trim
    if len(audio) > MAX_LEN:
        audio = audio[:MAX_LEN]
    else:
        audio = np.pad(audio, (0, MAX_LEN - len(audio)))

    features = []

    # MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # Delta
    delta = librosa.feature.delta(mfcc)
    features.extend(np.mean(delta, axis=1))

    # Delta-Delta
    delta2 = librosa.feature.delta(mfcc, order=2)
    features.extend(np.mean(delta2, axis=1))

    # Spectral features
    features.append(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
    features.append(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)))
    features.append(np.mean(librosa.feature.zero_crossing_rate(audio)))
    features.append(np.mean(librosa.feature.rms(y=audio)))

    return np.array(features)
