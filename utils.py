import torch
import torch.nn as nn
import python_speech_features
import os
import numpy as np

# === Модель ===
class BiLSTMModel(nn.Module):
    def __init__(self, input_size=39, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 2)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return self.fc(out)

# === Функция извлечения признаков (пример) ===
def extract_features(audio, sr, frame_size=0.025, hop_size=0.01):
    mfcc = python_speech_features.mfcc(audio, samplerate=sr, winlen=frame_size, winstep=hop_size, numcep=13)
    delta = python_speech_features.delta(mfcc, 2)
    delta2 = python_speech_features.delta(delta, 2)
    feats = np.hstack([mfcc, delta, delta2])
    hop_len = int(sr * hop_size)
    return feats, hop_len