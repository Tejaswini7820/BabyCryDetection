import librosa
import soundfile as sf
import numpy as np

def preprocess_audio(input_wav, output_wav, target_sr=22050):
    """
    Normalize + trim silence + resample
    """

    # Load
    y, sr = librosa.load(input_wav, sr=target_sr, mono=True)

    # Trim silence
    y, _ = librosa.effects.trim(y, top_db=20)

    # Normalize loudness
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    # Save clean audio
    sf.write(output_wav, y, target_sr)
