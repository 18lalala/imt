"""
Author: Ljy
Date: Aug 2022
"""

from re import M
from turtle import forward
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from timm.models.layers import trunc_normal_
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
import math

def quat2mat(quat):
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 -z2, 2*xy - 2*wz, 2*wy + 2*xz,
                            2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                            2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat

class Mlp(nn.Module):
    def __init__(self, in_feature, out_feature, hidden_feature=256, act_layer=nn.LeakyReLU, drop=0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_feature, hidden_feature)
        self.fc2 = nn.Linear(hidden_feature, out_feature)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class Position_encoder(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.mlp = Mlp(in_feature=in_feature, out_feature=out_feature)
    
    def forward(self, x):
        x = self.mlp(x)
        return x


class Instance_encoder(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.mlp = Mlp(in_feature=in_feature, out_feature=out_feature)
    
    def forward(self, x):
        x = self.mlp(x)
        return x


class Self_Attention(nn.Module):
    def __init__(
        self, dim, num_heads=8, qkv_bias=False, attn_drop=0.2, proj_drop=0.1
    ):
        super(Self_Attention, self).__init__()

        assert (
            dim % num_heads == 0
        ), "Embedding dimension should be divisible by number of heads"

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class SA_Block(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.2, proj_drop=0.1):
        super().__init__()
        self.sa = Self_Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.mlp = Mlp(in_feature=dim, out_feature=dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.sa(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CA_Block(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.2, proj_drop=0.1):
        super().__init__()
        self.ca = Cross_Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.mlp1 = Mlp(in_feature=dim, out_feature=dim)
        self.mlp2 = Mlp(in_feature=dim, out_feature=dim)
        self.norm11 = nn.LayerNorm(dim)
        self.norm12 = nn.LayerNorm(dim)
        self.norm21 = nn.LayerNorm(dim)
        self.norm22 = nn.LayerNorm(dim)

    def forward(self, x1, x2):
        x11, x22 = self.ca(self.norm11(x1), self.norm12(x2))
        x1 = x1 + x11
        x2 = x2 + x22
        x1 = x1 + self.mlp1(self.norm21(x1))
        x2 = x2 + self.mlp2(self.norm22(x2))
        return x1, x2


class Cross_Attention(nn.Module):
    def __init__(
        self, dim, num_heads=8, qkv_bias=False, attn_drop=0.2, proj_drop=0.1
    ):
        super(Cross_Attention, self).__init__()

        assert (
            dim % num_heads == 0
        ), "Embedding dimension should be divisible by number of heads"

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.act = nn.LeakyReLU()

    def forward(self, x1, x2):
        B, N, C = x1.shape
        B, M, C = x2.shape
        qkv1 = (
            self.qkv(x1)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        qkv2 = (
            self.qkv(x2)
            .reshape(B, M, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        # make torchscript happy (cannot use tensor as tuple)
        q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]
        q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]

        # attn x1
        attn1 = (q1 @ k2.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)

        x1 = (attn1 @ v2).transpose(1, 2).reshape(B, N, C)
        x1 = self.proj(x1)
        x1 = self.proj_drop(x1)

        # attn x2
        attn2 = (q2 @ k1.transpose(-2, -1)) * self.scale
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)

        x2 = (attn2 @ v1).transpose(1, 2).reshape(B, M, C)
        x2 = self.proj(x2)
        x2 = self.proj_drop(x2)

        return x1, x2


class Decoder_Block(nn.Module):
    def __init__(self, in_dim, hidden_dim, act=nn.LeakyReLU):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim[0])
        self.fc2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.fc3 = nn.Linear(hidden_dim[1], hidden_dim[2])
        self.fc_tr = nn.Linear(hidden_dim[2], 3)
        self.fc_r = nn.Linear(hidden_dim[2], 4)
        self.act = act()
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        x = self.act(x)
        tr = self.fc_tr(x)
        r = self.fc_r(x)
        r = r / torch.norm(r, p=2, dim=1, keepdim=True)
        r = quat2mat(r)
        return r, tr

class Decoder_Block_tr(nn.Module):
    def __init__(self, in_dim, hidden_dim, act=nn.LeakyReLU, drop=0.25):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim[0])
        self.fc2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.fc3 = nn.Linear(hidden_dim[1], hidden_dim[2])
        self.fc_tr = nn.Linear(hidden_dim[2], 3)
        self.act = act()
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc3(x)
        x = self.act(x)
        x = self.drop(x)
        tr = self.fc_tr(x)
        return tr


class Decoder_Block_r(nn.Module):
    def __init__(self, in_dim, hidden_dim, act=nn.LeakyReLU, drop=0.25):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim[0])
        self.fc2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.fc3 = nn.Linear(hidden_dim[1], hidden_dim[2])
        self.fc_r = nn.Linear(hidden_dim[2], 4)
        self.act = act()
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc3(x)
        x = self.act(x)
        x = self.drop(x)
        r = self.fc_r(x)
        r = r / torch.norm(r, p=2, dim=1, keepdim=True)
        r = quat2mat(r)
        return r


class get_loss(nn.Module):
    def __init__(self):
        super().__init__()

    # def forward(self, pose_gt, axis, rot, tr):
    #     axis_gt = pose_gt[:, :3].cuda()
    #     theta_gt = pose_gt[:, 3].cuda()
    #     trans_gt = pose_gt[:, 4:7].cuda()
    #     # axis_pred = pose_pred[:, :3].cuda()
    #     # theta_pred = pose_pred[:, 3].cuda()
    #     # trans_pred = pose_pred[:, 4:7].cuda()
    #     loss_ax = self.get_axis_loss(axis_gt, axis)
    #     loss_the = self.get_theta_loss(theta_gt, rot)
    #     loss_tr = self.get_trans_loss(trans_gt, tr)
    #     # loss = loss_ax + 100 * loss_the + 0.05 * loss_tr
    #     loss = loss_ax
    #     return loss, loss_ax, loss_the, loss_tr
    def forward(self, pose_gt, r_pred, tr_pred, source):
        loss_r = self.get_rotate_loss(r_pred, pose_gt[:, :9])
        loss_tr = self.get_trans_loss(pose_gt[:, 9:12], tr_pred)
        # self.get_geometric_loss(r_pred, tr_pred, pose_gt, source, epoch)
        # loss = loss_ax + 100 * loss_the + 0.05 * loss_tr
        loss = loss_tr + loss_r
        return loss, loss_r, loss_tr

    def get_axis_loss(self, gt, pred):
        target = torch.ones((gt.shape[0])).cuda()
        loss_ax = F.cosine_embedding_loss(gt, pred, target=target,reduction='mean')
        return loss_ax

    def get_theta_loss(self, gt, pred):
        l1 = F.smooth_l1_loss(torch.sin(2 * pred).reshape(-1).cuda(), torch.sin(2 * gt).reshape(-1).cuda(), reduction='mean')
        l2 = F.smooth_l1_loss(torch.cos(2 * pred).reshape(-1).cuda(), torch.cos(2 * gt).reshape(-1).cuda(), reduction='mean')
        loss_theta = l1 + l2
        return loss_theta

    def get_trans_loss(self, gt, pred):
        gt = gt
        pred = pred
        loss_tr = F.smooth_l1_loss(gt, pred, reduction='mean') / 15
        return loss_tr

    def get_geometric_loss(self, rot, tr, gt, source):

        # point = torch.ones((source.shape[0], 4)).cuda()
        point = source[:, :, :3].reshape(-1, 3)
        rot = rot[0]
        gt = gt[0]
        tr = tr[0]
        gt_pose = torch.zeros((3, 3)).cuda()
        gt_pose[0, 0:3] = gt[0:3]
        gt_pose[1, 0:3] = gt[4:7]
        gt_pose[2, 0:3] = gt[8:11]
        # gt_pose[3, 3] = 1.0
        pred_pose = torch.zeros((3, 4)).cuda()
        pred_pose[:3, :3] = rot.reshape(3, 3)
        pred_pose[:3, 3] = tr
        # print(tr.shape)
        # pred_pose[3, 3] = 1.0
        # point_pred = torch.matmul(rot, point.T).T
        # point_gt = torch.matmul(gt_pose, point.T).T
        # print(point_pred, point_gt)

    def get_rotate_loss(self, R_gt, R_pred):
        assert len(R_pred) == len(R_gt)
        trace_r1Tr2 = (R_pred.reshape(-1, 9) * R_gt.reshape(-1, 9)).sum(1)
        side = (trace_r1Tr2 - 1) / 2
        batch_loss = torch.acos(torch.clamp(side, min=-0.999, max=0.999))
        loss_r = torch.mean(batch_loss).cuda()
        # print(loss_r, side, torch.clamp(side, min=-0.999, max=0.999))
        return loss_r