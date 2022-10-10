"""
Author: Ljy
Date: Aug 2022
"""

from json import decoder
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from timm.models.layers import trunc_normal_
from utils import Decoder_Block, SA_Block, CA_Block, Mlp, Position_encoder, Instance_encoder, quat2mat, Decoder_Block_tr, Decoder_Block_r

class IMT(nn.Module):
    def __init__(self, in_feature=7, out_feature=32, num_heads=8):
        super().__init__()
        self.SA1 = SA_Block(dim=out_feature)
        self.SA2 = SA_Block(dim=out_feature)
        self.SA3 = SA_Block(dim=out_feature)
        self.SA4 = SA_Block(dim=out_feature)
        self.SA5 = SA_Block(dim=out_feature)
        self.SA6 = SA_Block(dim=out_feature)
        self.SA7 = SA_Block(dim=out_feature)
        self.SA8 = SA_Block(dim=out_feature)
        self.CA1 = CA_Block(dim=out_feature)
        self.CA2 = CA_Block(dim=out_feature)
        self.CA3 = CA_Block(dim=out_feature)
        self.CA4 = CA_Block(dim=out_feature)
        self.CA5 = CA_Block(dim=out_feature)
        self.CA6 = CA_Block(dim=out_feature)
        self.CA7 = CA_Block(dim=out_feature)
        self.CA8 = CA_Block(dim=out_feature)
        self.decoder_tr = Decoder_Block_tr(in_dim=2 * out_feature, hidden_dim=[1024, 512, 128])
        self.decoder_r = Decoder_Block_r(in_dim=2 * out_feature, hidden_dim=[1024, 512, 128])
        self.pos_embedding = Position_encoder(3, out_feature)
        self.ins_encoder = Instance_encoder(in_feature, out_feature)
        self.norm1 = nn.LayerNorm(out_feature)
        self.norm2 = nn.LayerNorm(out_feature)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x1, x2):
        pos1 = x1[:,:,:3]
        pos2 = x2[:,:,:3]
        x1 = self.pos_embedding(pos1) + self.ins_encoder(x1)
        x2 = self.pos_embedding(pos2) + self.ins_encoder(x2)
        x1 = self.SA1(x1)
        x2 = self.SA1(x2)
        x1, x2 = self.CA1(x1, x2)
        x1 = self.SA2(x1)
        x2 = self.SA2(x2)
        x1, x2 = self.CA2(x1, x2)
        x1 = self.SA3(x1)
        x2 = self.SA3(x2)
        x1, x2 = self.CA3(x1, x2)
        x1 = self.SA4(x1)
        x2 = self.SA4(x2)
        x1, x2 = self.CA4(x1, x2)
        x1 = torch.max(x1, dim=1, keepdim=False)[0]
        x2 = torch.max(x2, dim=1, keepdim=False)[0]
        x = torch.cat([x1, x2], dim=1)  # B, C
        # axis = self.mlp_axis(x)
        # axis = torch.max(axis, dim=1, keepdim=False)[0]
        r = self.decoder_r(x)
        tr = self.decoder_tr(x)
        # rot = self.mlp_rot(x)
        # rot = torch.max(rot, dim=1, keepdim=False)[0]
        return r, tr


if __name__ == '__main__':
    data1 = np.ones((1, 10, 6))
    data2 = np.ones((1, 20, 6))
    data3 = np.ones((1, 30, 7))
    pytorch_device = torch.device('cuda:0')
    x1 = torch.tensor(data1, dtype=torch.float32, requires_grad=True).to(pytorch_device)
    x2 = torch.tensor(data2, dtype=torch.float32, requires_grad=True).to(pytorch_device)
    x3 = torch.tensor(data3, dtype=torch.float32, requires_grad=True).to(pytorch_device)
    imt = IMT()
    imt.to(pytorch_device)
    y = imt(x1, x2)
    optimizer = optim.Adam(imt.parameters(), lr=0.0001)
    loss = y[0] - x3[0]

    optimizer.zero_grad()
    loss.backward(loss)
    optimizer.step()