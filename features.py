import numpy as np
import librosa
from python_speech_features import mfcc, delta

def extract_frame_features(signal, sample_rate):
    mfcc_feat = mfcc(signal, samplerate=sample_rate, numcep=13)
    d_mfcc_feat = delta(mfcc_feat, 2)
    dd_mfcc_feat = delta(d_mfcc_feat, 2)
    combined = np.hstack((mfcc_feat, d_mfcc_feat, dd_mfcc_feat))  # shape: (frames, 39)
    return combined
