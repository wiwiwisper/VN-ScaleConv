
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet import PointNetEncoder,feature_transform_reguliarzer


class get_model(nn.Module):
    def __init__(self, feature_transform=True):
        super(get_model, self).__init__()
        # 输入通道为 3（xyz）
        self.encoder = PointNetEncoder(global_feat=True,
                                       feature_transform=feature_transform,
                                       channel=3)
        # 回归头
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)  # 输出面积值

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        # 编码器提取特征
        x, trans, trans_feat = self.encoder(x)

        # 回归网络
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        area_pred = self.fc3(x)

        return area_pred, trans_feat  # 返回预测值 + 特征变换矩阵（用于正则化）

class get_loss(torch.nn.Module):
    def __init__(self, reg_weight=0.001):
        super(get_loss, self).__init__()
        self.reg_weight = reg_weight

    def forward(self, pred, target, trans_feat=None):
        # 主要损失：MSE
        loss = F.mse_loss(pred.squeeze(), target)

        # 特征变换正则化项（可选）
        if trans_feat is not None:
            reg_loss = feature_transform_reguliarzer(trans_feat)
            loss += self.reg_weight * reg_loss

        return loss
