import torch
import torch.nn as nn
import torchaudio
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
    frame_len = int(sr * frame_size)
    hop_len = int(sr * hop_size)
    mfcc = torchaudio.transforms.MFCC(
        sample_rate=sr,
        n_mfcc=13,
        melkwargs={"n_fft": 400, "hop_length": hop_len, "n_mels": 26}
    )(torch.tensor(audio).float())
    delta = torchaudio.functional.compute_deltas(mfcc)
    delta2 = torchaudio.functional.compute_deltas(delta)
    feat = torch.cat([mfcc, delta, delta2], dim=0).T
    return feat.numpy(), hop_len