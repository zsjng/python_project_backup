import torch
import torch.nn as nn
import numpy as np


class Encoder(nn.Module):
    def __init__(self, hidden_nodes=118, coding_dim=10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(520, hidden_nodes),
            nn.Tanh(),
            nn.Linear(hidden_nodes, coding_dim),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.mlp(x)
        return out


class Decoder(nn.Module):
    def __init__(self, hidden_nodes=128, coding_dim=10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(coding_dim, hidden_nodes),
            nn.Tanh(),
            nn.Linear(hidden_nodes, 520),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.mlp(x)
        return out


class Predict_head(nn.Module):
    def __init__(self, coding_dim=10, mode='fc'):
        super().__init__()
        if mode == 'fc':
            self.head = nn.Sequential(nn.Linear(coding_dim, 13))  # 13个输出节点，分13类
        elif mode == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(coding_dim, int(coding_dim*2)),
                nn.Linear(int(coding_dim*2), 13))
        else:
            raise ValueError("'mode' must be 'fc' or 'mlp'!")

    def forward(self, x):
        return self.head(x)


class AutoEncoder(nn.Module):
    def __init__(self, hidden_nodes=128, coding_dim=10, head_mode='fc'):
        super().__init__()
        self.encoder = Encoder(hidden_nodes, coding_dim)
        self.decoder = Decoder(hidden_nodes, coding_dim)
        self.pred_head = Predict_head(coding_dim, mode=head_mode)

    def forward(self, x):
        coding_vec = self.encoder(x)
        pred = self.pred_head(coding_vec)
        out = self.decoder(coding_vec)
        return pred, out


if __name__ == '__main__':
    Input = torch.randn([10, 520])
    AutoEncoder = AutoEncoder(hidden_nodes=128, coding_dim=10, head_mode='mlp')
    out = AutoEncoder(Input)
    Pred = out[0]
    Output = out[1]
    print('============Auto-encoder==============')
    print(Output.shape)
    print('============Pred_head==============')
    print(Pred.shape)