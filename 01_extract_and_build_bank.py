"""
脚本 1: 提取正常图像特征并构建 FAISS 记忆库

功能：
1. 加载所有正常训练图像（4.5 万张）
2. 使用 DINOv3 提取多层特征
3. 使用核心集采样压缩特征到 1%
4. 构建 FAISS 索引并保存

预计耗时：GPU 环境下约 2-4 小时
"""

import os
import sys
import yaml
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.dataset import TireXrayNormalDataset, create_dataloader
from src.model import load_dinov3_model
from src.coreset import CoresetBuilder, save_faiss_index


def main():
    print("="*60)
    print("步骤 1: 提取正常图像特征并构建记忆库")
    print("="*60)
    
    # 加载配置
    config_path = 'configs/config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前设备：{device}")
    
    # ==================== 1. 加载数据 ====================
    print("\n[1/5] 加载正常图像数据集...")
    train_dir = config['data']['train_normal_dir']
    
    if not os.path.exists(train_dir):
        print(f"错误：训练目录不存在：{train_dir}")
        print("请先将正常图像放入该目录")
        return
    
    train_dataset = TireXrayNormalDataset(train_dir, config_path)
    print(f"共加载 {len(train_dataset)} 张正常图像")
    
    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']
    train_loader = create_dataloader(
        train_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers,
        shuffle=False
    )
    
    # ==================== 2. 加载模型 ====================
    print("\n[2/5] 加载 DINOv3 模型...")
    model = load_dinov3_model(config_path, device)
    model.eval()
    
    # ==================== 3. 提取特征 ====================
    print("\n[3/5] 批量提取特征...")
    all_features = []
    all_paths = []
    
    with torch.no_grad():
        for batch_idx, (images, paths) in enumerate(tqdm(train_loader, desc="提取特征")):
            images = images.to(device)
            
            # 提取特征 [B, Num_Patches, Dim]
            features = model(images)
            
            # 移到 CPU 并转为 numpy
            features = features.cpu().numpy()
            all_features.append(features)
            all_paths.extend(paths)
    
    # 拼接所有特征 [N_images * N_patches, Dim]
    all_features = np.concatenate(all_features, axis=0)
    all_features = all_features.reshape(-1, all_features.shape[-1])
    
    print(f"\n特征提取完成！")
    print(f"总特征数：{all_features.shape[0]:,}")
    print(f"特征维度：{all_features.shape[1]}")
    
    # ==================== 4. 核心集采样 ====================
    print("\n[4/5] 执行核心集采样...")
    coreset_builder = CoresetBuilder(config)
    
    # 可以选择不同的采样方法：'random', 'kmeans', 'kcenter'
    sampling_method = 'kmeans'  # 推荐 K-Means
    centroids, index = coreset_builder.build(all_features, method=sampling_method)
    
    print(f"\n核心集采样完成！")
    print(f"原始特征数：{all_features.shape[0]:,}")
    print(f"压缩后特征数：{centroids.shape[0]:,}")
    print(f"压缩比：{centroids.shape[0] / all_features.shape[0] * 100:.2f}%")
    
    # ==================== 5. 保存索引 ====================
    print("\n[5/5] 保存 FAISS 索引...")
    save_path = config['output']['memory_bank_path']
    
    # 确保目录存在
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    save_faiss_index(index, save_path)
    print(f"\n记忆库已保存到：{save_path}")
    
    # 保存一些统计信息
    stats_path = Path(save_path).parent / 'memory_bank_stats.txt'
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write(f"训练图像数量：{len(train_dataset)}\n")
        f.write(f"原始特征总数：{all_features.shape[0]:,}\n")
        f.write(f"核心集特征数：{centroids.shape[0]:,}\n")
        f.write(f"压缩比：{centroids.shape[0] / all_features.shape[0] * 100:.2f}%\n")
        f.write(f"特征维度：{centroids.shape[1]}\n")
        f.write(f"采样方法：{sampling_method}\n")
    
    print(f"\n统计信息已保存到：{stats_path}")
    print("\n" + "="*60)
    print("✅ 记忆库构建完成！可以运行 02_test_and_evaluate.py 进行评估")
    print("="*60)


if __name__ == '__main__':
    main()
