# 此模块为频域分析通道注意力代码
import torch.nn as nn
import torch
import math
import torch.nn.functional as F


def get_1d_dct(i, freq, L):
    """
    一维的离散余弦变换
    :param i: 区分
    :param freq:
    :param L:
    :return:
    """
    if freq == 0:
        return math.cos(math.pi * freq * (i + 0.5) / L) / math.sqrt(L)
    else:
        return math.cos(math.pi * freq * (i + 0.5) / L)


def get_dct_weights(width, height, channel, fidx_u=[0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 2, 3],
                    fidx_v=[0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 2, 5]):
    """
    获取离散余弦变换变量
    :param width: 输入的宽
    :param height: 输入的高
    :param channel: 输入的通道数目
    :param fidx_u: 索引1，与fidx_v组成索引，组成的索引见论文figure5
    :param fidx_v:索引2，与fidx_u组成索引，组成的索引见论文fighure5
    :return:生成的离散余弦变换变量
    """
    scale_ratio = width // 7
    fidx_u = [u * scale_ratio for u in fidx_u]
    fidx_v = [v * scale_ratio for v in fidx_v]
    dct_weights = torch.zeros(1, channel, width, height)
    c_part = channel // len(fidx_u)
    for i, (u_x, v_y) in enumerate(zip(fidx_u, fidx_v)):
        for t_x in range(height):
            for t_y in range(height):
                dct_weights[:, i * c_part:(i + 1) * c_part, t_x, t_y] = get_1d_dct(t_x, u_x, width) * get_dct_weights(
                    t_y, v_y, height)
    return dct_weights


class Fca_Module(nn.Module):
    def __init__(self, channel, reduction, width, height):
        super(Fca_Module, self).__init__()
        self.width = width
        self.height = height
        self.register_buffer("pre_computed_dct_weights", get_dct_weights(self.width, self.height, channel))
        self.attention_module = nn.Sequential()

    def forward(self, x):
        batch, channel, width, height = x.size
        attention = self.attention_module(
            torch.sum(F.adaptive_avg_pool2d(x, (self.height, self.width)) * self.pre_computed_dct_weight,
                      dim=(2, 3))).reshape(batch, channel, 1, 1)
        return x * attention.expand_as(x)
