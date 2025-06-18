import numpy as np
import librosa


FRAME_LENGTH = 2048
HOP_LENGTH = 512
N_MFCC = 13

def extract_frame_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                hop_length=HOP_LENGTH, n_fft=FRAME_LENGTH)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.vstack([mfcc, mfcc_delta, mfcc_delta2]).T
    return features
