"""
@author : MaJiaHao
@when : 2025-4-13
@homepage : https://github.com/HEX-QWQ/LucaOne
"""

import torch
from torch import nn

class SimilarityEncoding(nn.Module):
    """
    进行相似度得分嵌入
    目前的思路是定义一个可学习的向量W_sim
    将每个相似度得分s_i转换为d维向量s_i * W_sim
    这里s_i * W_sim 表示相似度得分对一个可学习方向的缩放
    """
    def __init__(self,d_model,device):
        super(SimilarityEncoding,self).__init__()
        self.W_sim = nn.Parameter(torch.randn(d_model))

    def forward(self,x):
        """
        x: 相似度得分张量，形状 (batch_size, n)
        返回: 嵌入张量，形状 (batch_size, n, d_model)
        """
        # x: (batch_size, n) -> (batch_size, n, 1)
        x = x.unsqueeze(-1)
        
        # W_sim: (d_model) -> (1, 1, d_model)
        W_sim = self.W_sim.view(1, 1, -1)
        
        # (batch_size, n, d_model)
        encoding = x * W_sim
        
        return encoding


