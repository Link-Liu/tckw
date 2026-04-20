#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
import torch
import torch.nn as nn
from agent_ppo.conf.conf import Config

def make_fc_layer(in_features, out_features, init_gain=1.0):
    """创建正交初始化的线性层（PPO 推荐实践）。"""
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data, gain=init_gain)
    nn.init.zeros_(fc.bias.data)
    return fc

class Model(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.model_name = "s_h_i_r_o_dual_branch"
        self.device = device
        
        # ============ 1. CNN 分支 (Spatial Branch) ============
        # 核心使命：吃掉 6x21x21 地图，保留空间拓扑，建立“死胡同”、“怪卡位”的空间形状感
        self.cnn_branch = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2), # 21x21 -> 10x10
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 10x10 -> 5x5
            nn.Flatten(),
            make_fc_layer(64 * 5 * 5, 256),
            nn.ReLU(),
            make_fc_layer(256, 128),
            nn.ReLU()
        )

        # ============ 2. Summary 分支 (Situation Branch) ============
        # 核心使命：吃掉那 78 维统计量和局势压力值，直接对当前局面的压力做抽象评估
        self.summary_branch = nn.Sequential(
            make_fc_layer(Config.SCALAR_LEN, 256),
            nn.LayerNorm(256), # 给标量特征一个 LayerNorm 有助于稳定 PPO
            nn.ReLU(),
            make_fc_layer(256, 128),
            nn.ReLU()
        )

        # ============ 3. Fusion & Backbone ============
        # 核心使命：把 空间图感知 和 局势数值感知 整合在一起，综合决策
        combined_dim = 128 + 128
        self.fusion_backbone = nn.Sequential(
            make_fc_layer(combined_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            make_fc_layer(256, 256),
            nn.ReLU()
        )

        # ============ 4. Heads (Actor & Critic) ============
        # Actor: 输出 16 个动作的 logits (使用较小的增益初始化，鼓励探索)
        self.actor_head = make_fc_layer(256, Config.ACTION_NUM, init_gain=0.01)
        # Critic: 输出当前局势的 Value (评价现在是顺风还是必死)
        self.critic_head = make_fc_layer(256, Config.VALUE_NUM, init_gain=1.0)

    def forward(self, obs, inference=False):
        bs = obs.size(0)
        
        # 1. 精准切割特征
        # obs 格式预期: [Batch, 2724]  (78 scalar + 2646 map)
        scalars = obs[:, :Config.SCALAR_LEN] # [Batch, 78]
        maps = obs[:, Config.SCALAR_LEN:].view(bs, 6, Config.MAP_SIDE, Config.MAP_SIDE) 
        
        # 2. 各走各的路：独立提取特征
        spatial_emb = self.cnn_branch(maps)
        summary_emb = self.summary_branch(scalars)
        
        # 3. 最高维度融合信息
        combined = torch.cat([spatial_emb, summary_emb], dim=1)
        hidden = self.fusion_backbone(combined)
        
        # 4. 吐出动作策略和局面价值预测
        logits = self.actor_head(hidden)
        value = self.critic_head(hidden)
        
        return logits, value

    def set_train_mode(self):
        self.train()
        
    def set_eval_mode(self):
        self.eval()