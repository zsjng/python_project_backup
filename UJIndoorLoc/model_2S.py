import torch
import torch.nn as nn
import numpy as np


class Encoder(nn.Module):
    def __init__(self, hidden_nodes=118, coding_dim=10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(520, hidden_nodes),
            # nn.ReLU(),
            nn.Linear(hidden_nodes, coding_dim)
        )

    def forward(self, x):
        out = self.mlp(x)
        return out


class Decoder(nn.Module):
    def __init__(self, hidden_nodes=128, coding_dim=10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(coding_dim, hidden_nodes),
            # nn.ReLU(),
            nn.Linear(hidden_nodes, 520)
        )

    def forward(self, x):
        out = self.mlp(x)
        return out


class AutoEncoder(nn.Module):
    """
    two-stage model:
    1 pretrain AE
    2 fine-tune PH
    """
    def __init__(self, hidden_nodes=128, coding_dim=10):
        super().__init__()
        self.encoder = Encoder(hidden_nodes, coding_dim)
        self.decoder = Decoder(hidden_nodes, coding_dim)

    def forward(self, x):
        coding_vec = self.encoder(x)
        out = self.decoder(coding_vec)
        return coding_vec, out


class Predict_head(nn.Module):
    """Linear Proj for two predict layer"""
    def __init__(self, coding_dim=10):
        super().__init__()
        self.head = nn.Linear(coding_dim, 13)

    def forward(self, x):
        return self.head(x)


class Predict_head2(nn.Module):
    """Symmetry MLP (hidden nodes = 2 * input nodes) for two predict layer"""
    def __init__(self, coding_dim=10):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(coding_dim, int(2 * coding_dim)),
            nn.Linear(int(2 * coding_dim), 13)
        )

    def forward(self, x):
        return self.head(x)


# class Predict_head3(nn.Module):
#     """Asymmetry MLP (hidden nodes = classes * input nodes) for two predict layer"""
#     def __init__(self, coding_dim=10):
#         super().__init__()
#         self.floor_head = nn.Sequential(
#             nn.Linear(coding_dim, int(5*coding_dim)),
#             nn.Linear(int(5*coding_dim), 5)
#         )
#         self.building_head = nn.Sequential(
#             nn.Linear(coding_dim, int(3 * coding_dim)),
#             nn.Linear(int(3 * coding_dim), 3)
#         )
#
#     def forward(self, x):
#         pred_floorID = self.floor_head(x)
#         pred_buildingID = self.building_head(x)
#         return pred_floorID, pred_buildingID


if __name__ == '__main__':
    Input = torch.randn([10, 520])
    AutoEncoder = AutoEncoder(hidden_nodes=128, coding_dim=10)
    Predict_head = Predict_head(coding_dim=10)
    out = AutoEncoder(Input)
    pred = Predict_head(out[0])
    print('============Auto-encoder==============')
    print(out[0].shape)
    print(out[1].shape)
    print('============Pred_head==============')
    print(pred.shape)