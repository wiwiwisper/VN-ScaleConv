import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vn_layers_chidu import *
from models.utils.vn_dgcnn_chidu_util import get_graph_feature


class get_model(nn.Module):
    def __init__(self, args, normal_channel=False):
        super(get_model, self).__init__()
        self.args = args
        self.n_knn = 10

        # VN-DGCNN Encoder
        # self.conv1 = VNLinearWithScale_PerSample(2, 64 // 3)  # 替换原始VNLinearLeakyReLU
        # self.conv2 = VNLinearWithScale_PerSample(64 // 3 * 2, 64 // 3)
        # self.conv3 = VNLinearWithScale_PerSample(64 // 3 * 2, 128 // 3)
        # self.conv4 = VNLinearWithScale_PerSample(128 // 3 * 2, 256 // 3)



        #self.conv1 = VNLinearLeakyReLU(2, 64 // 3)  # 替换原始VNLinearLeakyReLU
        #self.conv2 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)
        #self.conv3 = VNLinearLeakyReLU(64 // 3 * 2, 128 // 3)
        #self.conv4 = VNLinearLeakyReLU(128 // 3 * 2, 256 // 3)

        self.conv1 = VNLinearWithScale_PerPoint(2, 64 // 3)  # 替换原始VNLinearLeakyReLU
        self.conv2 = VNLinearWithScale_PerPoint(64 // 3 * 2, 64 // 3)
        self.conv3 = VNLinearWithScale_PerPoint(64 // 3 * 2, 128 // 3)
        self.conv4 = VNLinearWithScale_PerPoint(128 // 3 * 2, 256 // 3)


        self.conv5 = VNLinearLeakyReLU(256 // 3 + 128 // 3 + 64 // 3 * 2,1024 // 3, dim=4, share_nonlinearity=True)
        # self.conv5 = VNLinearWithScale_PerSample(256 // 3 + 128 // 3 + 64 // 3 * 2,1024 // 3)
        # self.conv5 = VNLinearWithScale_PerPoint(256 // 3 + 128 // 3 + 64 // 3 * 2,1024 // 3)


        self.std_feature = VNStdFeature(1024 // 3 * 2, dim=4, normalize_frame=False)
        self.linear1 = nn.Linear((1024 // 3) * 12, 512)

        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, 1)  # 回归输出一个浮点数

        if 'max' == 'max':
            self.pool1 = VNMaxPool(64 // 3)
            self.pool2 = VNMaxPool(64 // 3)
            self.pool3 = VNMaxPool(128 // 3)
            self.pool4 = VNMaxPool(256 // 3)
        else:
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
            self.pool4 = mean_pool

    def forward(self, x):
        # 保留原始坐标并初始化尺度特征
        # x_coord = x.clone()  # [B, N, 3]
        batch_size = x.size(0)

        # 原始点云特征提取流程
        x = x.unsqueeze(1)  # [B, 1, N, 3]
        x = get_graph_feature(x, k=self.n_knn)  # [B, 2, N, k]
        x = self.conv1(x)  # [B, 64//3, N, k] (已内置尺度保留)

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

        # 多尺度特征融合
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)

        # 全局特征增强
        num_points = x.size(-1)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), dim=1)
        x, trans = self.std_feature(x)
        x = x.view(batch_size, -1, num_points)


        # 双通道池化 + 尺度拼接
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)  # [B, C]
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)  # [B, C]
        x = torch.cat((x1, x2), dim=1)  # [B, 2*C + 1]

        # 回归头（保持不变）
        x = F.leaky_relu(self.bn1(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn2(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)  # [B, 1]
        x = x.squeeze()  # [B]

        return x, None

class get_loss(torch.nn.Module):
    def __init__(self, loss_type='mse'):
        super(get_loss, self).__init__()
        self.loss_type = loss_type

    def forward(self, pred, target, trans_feat=None):
        """
        pred: [B] 预测值
        target: [B] 真实值
        """
        if self.loss_type == 'mse':
            loss = F.mse_loss(pred, target)
        elif self.loss_type == 'mae':
            loss = F.l1_loss(pred, target)
        elif self.loss_type == 'huber':
            loss = F.huber_loss(pred, target)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

        return loss
