import numpy as np
import librosa
import python_speech_features

def extract_frame_features(y, sr):
    # MFCC
    mfcc = python_speech_features.mfcc(y, samplerate=sr, winstep=512 / sr)
    
    # Energy
    energy = np.array([
        np.sum(np.square(y[i * 512: (i + 1) * 512]))
        for i in range(mfcc.shape[0])
    ]).reshape(-1, 1)
    
    # Pitch (using librosa's piptrack)
    pitches, magnitudes = librosa.piptrack(y=y.astype(float), sr=sr, hop_length=512)
    pitch = []
    for i in range(pitches.shape[1]):
        index = magnitudes[:, i].argmax()
        pitch_val = pitches[index, i]
        pitch.append(pitch_val)
    pitch = np.array(pitch).reshape(-1, 1)
    
    # Объединение всех признаков
    features = np.hstack([mfcc, energy, pitch])
    return features
