import torch
import torch.nn as nn
from torchsummary import summary

from hyperparameters import hp


class CNNLayerNorm(nn.Module):

    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        x = x.transpose(2, 3).contiguous() # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous() # (batch, channel, feature, time)


class ResidualCNN(nn.Module):

    def __init__(self, channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.layers = nn.Sequential(
            CNNLayerNorm(n_feats),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(channels, channels, kernel, stride, padding=kernel//2),
            CNNLayerNorm(n_feats),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(channels, channels, kernel, stride, padding=kernel//2)
        )

    def forward(self, x):
        residual = x    # (batch, channel, feature, time)
        x = self.layers(x)
        x += residual
        return x        # (batch, channel, feature, time)

class BiGRU(nn.Module):

    def __init__(self, rnn_dim, hidden_size, dropout):
        super(BiGRU, self).__init__()
        self.bigru = nn.Sequential(
            nn.LayerNorm(rnn_dim),
            nn.GELU(),
            nn.GRU(input_size=rnn_dim, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x, _ = self.bigru(x)
        x = self.dropout(x)
        return x


class ASR(nn.Module):

    def __init__(self, dropout, hidden_size, rnn_layers, rescnn_layers, n_mels):

        super(ASR, self).__init__()

        self.dropout = dropout
        self.n_mels = n_mels // 2
        self.lin_start = 128
        self.lin_end = 29
        self.hidden_size = hidden_size
        self.gru_layers = rnn_layers
        self.rescnn_layers = rescnn_layers

        # Process Mel Spectogram via Residual Conv2D Layers
        self.rescnn_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            *[ResidualCNN(32, kernel=3, stride=1, dropout=dropout, n_feats=self.n_mels) for _ in range(rescnn_layers)]
        )

        # Linear layers
        self.fc1 = nn.Sequential(
            nn.LayerNorm(self.n_mels * 32),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.n_mels * 32, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )

        # GRU architecture
        self.gru = nn.Sequential(*[
                    BiGRU(rnn_dim=hidden_size if i==0 else hidden_size*2,
                                    hidden_size=hidden_size, dropout=dropout)
                    for i in range(self.gru_layers)
                ])

        # Linear Layers
        self.fc2 = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.lin_end),
            nn.LayerNorm(self.lin_end),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )


    def hidden(self, bs):
        return torch.zeros(self.gru_layers * 2, bs, self.hidden_size).to("cuda")

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.rescnn_layers(x)
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        x = x.transpose(1,2) # Since linear layers require input of shape (batch, time, channels=n_mels)
        x = self.fc1(x)
        # x, h = self.gru(x, self.hidden(x.size(0)))
        x = self.gru(x)
        x = self.fc2(x)
        return x, None


class ASR1(nn.Module):

    def __init__(self, dropout, rnn_dim, n_rnn_layers, n_cnn_layers, n_feats):
        super(ASR1, self).__init__()
        n_feats = n_feats//2
        self.cnn = nn.Conv2d(1, 32, 3, stride=2, padding=3//2)  # cnn for extracting heirachal features

        # n residual cnn layers with filter size of 32
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats)
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(n_feats*32, rnn_dim)
        self.birnn_layers = nn.Sequential(*[
            BiGRU(rnn_dim=rnn_dim if i==0 else rnn_dim*2,
                             hidden_size=rnn_dim, dropout=dropout)
            for i in range(n_rnn_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, 29)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2) # (batch, time, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x, None


if __name__ == "__main__":
    # asr = ASR(hp["dropout"], hp["hidden_size"], hp["rnn_layers"], hp["cnn_layers"], hp["n_mels"])
    asr = ASR(hp["dropout"], hp["hidden_size"], hp["rnn_layers"], hp["cnn_layers"], hp["n_mels"])
    asr = asr.cuda()
    print(summary(asr, (128, 680)))
    print("Total Parameters:", sum(p.numel() for p in asr.parameters()))
