import torch.nn as nn

class RTModel(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=64, num_layers=2, output_dim=2):
        super(RTModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        logits = self.classifier(output)
        return logits