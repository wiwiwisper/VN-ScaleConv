"""
Ablation 3: backbone 使用 6 层 VNLinearWithScale_PerPoint（conv1~conv6），
在原始 4 层基础上再增加 2 层（256, 512 通道），探究更深网络的影响。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vn_layers_chidu import (
    VNLinearWithScale_PerPoint,
    VNLinearLeakyReLU,
    VNMaxPool,
    VNStdFeature,
)
from models.utils.vn_dgcnn_chidu_util import get_graph_feature


class get_model(nn.Module):
    def __init__(self, args, normal_channel=False):
        super(get_model, self).__init__()
        self.args = args
        self.n_knn = 10

        self.conv1 = VNLinearWithScale_PerPoint(2, 16)
        self.pool1 = VNMaxPool(16)

        self.conv2 = VNLinearWithScale_PerPoint(16 * 2, 32)
        self.pool2 = VNMaxPool(32)

        self.conv3 = VNLinearWithScale_PerPoint(32 * 2, 64)
        self.pool3 = VNMaxPool(64)

        self.conv4 = VNLinearWithScale_PerPoint(64 * 2, 128)
        self.pool4 = VNMaxPool(128)

        # 新增 conv5, conv6
        self.conv5 = VNLinearWithScale_PerPoint(128 * 2, 256)
        self.pool5 = VNMaxPool(256)

        self.conv6 = VNLinearWithScale_PerPoint(256 * 2, 512)
        self.pool6 = VNMaxPool(512)

        # 6 层特征聚合：16+32+64+128+256+512 = 1008
        self.conv_agg = VNLinearLeakyReLU(16 + 32 + 64 + 128 + 256 + 512, 1024 // 3, dim=4, share_nonlinearity=True)

        self.std_feature = VNStdFeature(1024 // 3 * 2, dim=4, normalize_frame=False)
        self.linear1 = nn.Linear((1024 // 3) * 12, 512)

        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.1)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, x):
        batch_size = x.size(0)

        x = x.unsqueeze(1)
        x = get_graph_feature(x, k=self.n_knn)
        x = self.conv1(x)
        x1 = self.pool1(x)

        x = get_graph_feature(x1, k=self.n_knn)
        x = self.conv2(x)
        x2 = self.pool2(x)

        x = get_graph_feature(x2, k=self.n_knn)
        x = self.conv3(x)
        x3 = self.pool3(x)

        x = get_graph_feature(x3, k=self.n_knn)
        x = self.conv4(x)
        x4 = self.pool4(x)

        x = get_graph_feature(x4, k=self.n_knn)
        x = self.conv5(x)
        x5 = self.pool5(x)

        x = get_graph_feature(x5, k=self.n_knn)
        x = self.conv6(x)
        x6 = self.pool6(x)

        # 6 层多尺度特征融合
        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)
        x = self.conv_agg(x)

        num_points = x.size(-1)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), dim=1)
        x, trans = self.std_feature(x)
        x = x.view(batch_size, -1, num_points)

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), dim=1)

        x = F.leaky_relu(self.bn1(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn2(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        x = x.squeeze()

        return x, None


class get_loss(torch.nn.Module):
    def __init__(self, loss_type='mse'):
        super(get_loss, self).__init__()
        self.loss_type = loss_type

    def forward(self, pred, target, trans_feat=None):
        if self.loss_type == 'mse':
            loss = F.mse_loss(pred, target)
        elif self.loss_type == 'mae':
            loss = F.l1_loss(pred, target)
        elif self.loss_type == 'huber':
            loss = F.huber_loss(pred, target)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
        return loss
