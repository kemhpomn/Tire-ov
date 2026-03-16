"""
模型定义模块
负责加载 DINOv3 骨干网络并提取多层特征
"""

import os
# 设置 Hugging Face 官方 URL（如果需要使用镜像，可改为 https://hf-mirror.com）
# os.environ['HF_ENDPOINT'] = 'https://huggingface.co'

import torch
import torch.nn as nn
from transformers import AutoModel
import yaml


class DINOv3FeatureExtractor(nn.Module):
    """DINOv3 特征提取器（冻结参数）"""
    
    def __init__(self, config_path='configs/config.yaml'):
        """
        Args:
            config_path: 配置文件路径
        """
        super(DINOv3FeatureExtractor, self).__init__()
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        model_name = self.config['model']['name']
        print(f"正在加载 DINOv3 模型：{model_name}")
        
        # 加载预训练的 DINOv3 模型
        self.model = AutoModel.from_pretrained(model_name)
        
        # 冻结所有参数（不进行训练）
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.eval()
        
        # 获取模型配置
        self.patch_size = self.config['model']['patch_size']
        self.extracted_layers = self.config['model']['extracted_layers']
        
        # 特征维度（DINOv3-Small 通常是 384，拼接两层后为 768）
        # 注意：实际维度需要根据具体模型确认
        self.feature_dim = 384 * len(self.extracted_layers)
        
        print(f"DINOv3 特征提取器已加载，特征维度：{self.feature_dim}")
    
    def forward(self, x):
        """
        前向传播，提取多层特征
        
        Args:
            x: 输入图像张量 [B, 3, H, W]
            
        Returns:
            features: 融合后的 Patch 特征 [B, Num_Patches, Feature_Dim]
        """
        with torch.no_grad():
            # 获取所有隐藏层状态
            outputs = self.model(x, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            
            # 提取指定层的特征
            layer_features = []
            for layer_idx in self.extracted_layers:
                # 去掉 [CLS] token，只保留图像 Patch 特征
                # hidden_states[layer_idx] shape: [B, Num_Patches+1, Dim]
                layer_feat = hidden_states[layer_idx][:, 1:, :]
                layer_features.append(layer_feat)
            
            # 在通道维度拼接多层特征
            features = torch.cat(layer_features, dim=-1)
            
            return features
    
    def get_num_patches(self, image_size=256):
        """计算图像会被分成多少个 Patch"""
        num_patches_per_dim = image_size // self.patch_size
        return num_patches_per_dim ** 2


def load_dinov3_model(config_path='configs/config.yaml', device=None):
    """
    加载 DINOv3 模型到指定设备
    
    Args:
        config_path: 配置文件路径
        device: 设备（cuda/cpu）
        
    Returns:
        model: DINOv3 特征提取器
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = DINOv3FeatureExtractor(config_path)
    model = model.to(device)
    
    return model
