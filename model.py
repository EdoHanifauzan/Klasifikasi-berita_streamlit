import torch
import torch.nn as nn

class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim=768,
        hidden_dim=256,
        output_dim=5,
        dropout=0.3
    ):
        super(BiLSTMClassifier, self).__init__()

        self.bilstm = nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(
            hidden_dim * 2,
            output_dim
        )

        self.activation = nn.Sigmoid()

    def forward(self, x):

        lstm_out, _ = self.bilstm(x)

        forward_last = lstm_out[:, -1, :256]
        backward_first = lstm_out[:, 0, 256:]

        out = torch.cat(
            (forward_last, backward_first),
            dim=1
        )

        out = self.fc(out)

        return self.activation(out)