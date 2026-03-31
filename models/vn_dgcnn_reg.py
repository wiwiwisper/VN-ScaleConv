import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vn_layers import *
from models.utils.vn_dgcnn_util import get_graph_feature


class get_model(nn.Module):
    def __init__(self, args, normal_channel=False):
        super(get_model, self).__init__()
        self.args = args
        self.n_knn = 10

        # VN-DGCNN Encoder
        self.conv1 = VNLinearLeakyReLU(2, 64 // 3)
        self.conv2 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3)
        self.conv3 = VNLinearLeakyReLU(64 // 3 * 2, 128 // 3)
        self.conv4 = VNLinearLeakyReLU(128 // 3 * 2, 256 // 3)

        self.conv5 = VNLinearLeakyReLU(256 // 3 + 128 // 3 + 64 // 3 * 2,
                                       1024 // 3, dim=4, share_nonlinearity=True)

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
        batch_size = x.size(0)
        x = x.unsqueeze(1)  # [B, 1, N, 3]
        x = get_graph_feature(x, k=self.n_knn)  # [B, 2, N, k]
        x = self.conv1(x)  # [B, 64//3, N, k]

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

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)

        num_points = x.size(-1)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), dim=1)
        x, trans = self.std_feature(x)
        x = x.view(batch_size, -1, num_points)

        # Global feature pooling
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        # Regression head
        x = F.leaky_relu(self.bn1(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn2(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)  # 输出单个浮点数 [B, 1]

        x = x.squeeze()  # [B]

        return x, None  # 返回标量和无trans_feat（可选）


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
