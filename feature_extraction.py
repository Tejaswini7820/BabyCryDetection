import librosa
import numpy as np

SAMPLE_RATE = 22050
DURATION = 5
MAX_LEN = SAMPLE_RATE * DURATION
N_MFCC = 40

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        # ----------------------------
        # Pad / Trim
        # ----------------------------
        if len(audio) > MAX_LEN:
            audio = audio[:MAX_LEN]
        else:
            audio = np.pad(audio, (0, MAX_LEN - len(audio)))

        features = []

        # ----------------------------
        # MFCC
        # ----------------------------
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))

        # ----------------------------
        # Delta MFCC
        # ----------------------------
        delta = librosa.feature.delta(mfcc)
        features.extend(np.mean(delta, axis=1))
        features.extend(np.std(delta, axis=1))

        # ----------------------------
        # Delta-Delta MFCC
        # ----------------------------
        delta2 = librosa.feature.delta(mfcc, order=2)
        features.extend(np.mean(delta2, axis=1))
        features.extend(np.std(delta2, axis=1))

        # ----------------------------
        # Spectral & Energy features
        # ----------------------------
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(audio)
        rms = librosa.feature.rms(y=audio)

        features.append(np.mean(spectral_centroid))
        features.append(np.std(spectral_centroid))

        features.append(np.mean(spectral_bandwidth))
        features.append(np.std(spectral_bandwidth))

        features.append(np.mean(zcr))
        features.append(np.std(zcr))

        features.append(np.mean(rms))
        features.append(np.std(rms))

        # ----------------------------
        # Pitch (F0)
        # ----------------------------
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch_values = pitches[pitches > 0]

        pitch_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0
        pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else 0

        features.append(pitch_mean)
        features.append(pitch_std)

        return np.array(features)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return np.zeros(40 * 6 + 12)

