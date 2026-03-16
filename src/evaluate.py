"""
评估模块
负责计算异常分数、评估指标（ROC、AUC、F1-Score 等）
"""

import numpy as np
import torch
from sklearn.metrics import (
    roc_curve, 
    precision_recall_curve, 
    f1_score, 
    auc, 
    classification_report
)
import faiss


def extract_features_batch(model, dataloader, device):
    """
    批量提取特征
    
    Args:
        model: DINOv3 模型
        dataloader: 数据加载器
        device: 设备
        
    Returns:
        all_features: 所有特征列表 [N, Num_Patches, Dim]
        all_labels: 所有标签列表
        all_paths: 所有图片路径列表
    """
    model.eval()
    all_features = []
    all_labels = []
    all_paths = []
    
    with torch.no_grad():
        for batch_idx, (images, labels, paths) in enumerate(dataloader):
            images = images.to(device)
            
            # 提取特征
            features = model(images)
            
            # 移到 CPU 并转为 numpy
            features = features.cpu().numpy()
            
            all_features.append(features)
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"已处理 {batch_idx + 1}/{len(dataloader)} 批次")
    
    # 拼接所有特征
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.array(all_labels)
    
    print(f"特征提取完成：{all_features.shape}")
    return all_features, all_labels, all_paths


def calculate_anomaly_score(features, index, k=9, normalize=False, score_range=None):
    """
    计算每个 Patch 的异常分数和图像级分数
    
    Args:
        features: 单张图像的特征 [Num_Patches, Dim]
        index: FAISS 索引（正常特征记忆库）
        k: 最近邻数量
        normalize: 是否归一化分数到 [0, 1]
        score_range: 分数范围 (min_score, max_score) 用于归一化
        
    Returns:
        patch_scores: 每个 Patch 的异常分数 [Num_Patches]
        image_score: 图像级异常分数（标量）
    """
    # 确保是 float32 类型
    features_np = features.astype('float32')
    
    # 搜索 k 个最近邻
    distances, _ = index.search(features_np, k)
    
    # 计算每个 Patch 的异常分数（k 个邻居距离的平均值）
    patch_scores = np.mean(distances, axis=1)
    
    # 图像级分数：取所有 Patch 中的最大异常分数
    image_score = np.max(patch_scores)
    
    # 如果需要归一化
    if normalize and score_range is not None:
        min_score, max_score = score_range
        score_range_val = max_score - min_score
        if score_range_val > 1e-8:
            patch_scores = (patch_scores - min_score) / score_range_val
            image_score = (image_score - min_score) / score_range_val
        else:
            patch_scores = np.zeros_like(patch_scores)
            image_score = 0.0
    
    return patch_scores, image_score


def evaluate_on_testset(all_features, all_labels, index, k=9):
    """
    在测试集上评估模型性能
    
    Args:
        all_features: 所有测试图像的特征 [N, Num_Patches, Dim]
        all_labels: 真实标签 [N] (0=正常，1=异常)
        index: FAISS 索引
        k: 最近邻数量
        
    Returns:
        image_scores: 所有图像的异常分数 [N]
        best_threshold: 最佳阈值
        best_f1: 最佳 F1-Score
        metrics: 其他评估指标字典
    """
    n_images = len(all_features)
    image_scores = []
    
    print("开始计算异常分数...")
    for i in range(n_images):
        _, img_score = calculate_anomaly_score(all_features[i], index, k)
        image_scores.append(img_score)
        
        if (i + 1) % 100 == 0:
            print(f"已处理 {i + 1}/{n_images} 张图像")
    
    image_scores = np.array(image_scores)
    
    # 计算 ROC 曲线
    fpr, tpr, roc_thresholds = roc_curve(all_labels, image_scores)
    roc_auc = auc(fpr, tpr)
    
    # 计算 PR 曲线
    precision, recall, pr_thresholds = precision_recall_curve(all_labels, image_scores)
    
    # 计算 F1-Score
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = pr_thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    # 使用最佳阈值计算其他指标
    y_pred = (image_scores >= best_threshold).astype(int)
    accuracy = np.mean(y_pred == all_labels)
    
    metrics = {
        'roc_auc': roc_auc,
        'best_f1': best_f1,
        'best_threshold': best_threshold,
        'accuracy': accuracy,
        'fpr': fpr,
        'tpr': tpr,
        'precision': precision,
        'recall': recall,
        'f1_scores': f1_scores
    }
    
    print("\n" + "="*50)
    print("评估结果汇总")
    print("="*50)
    print(f"测试集样本数：{n_images}")
    print(f"正常样本数：{np.sum(all_labels == 0)}")
    print(f"异常样本数：{np.sum(all_labels == 1)}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"最佳 F1-Score: {best_f1:.4f}")
    print(f"最佳阈值：{best_threshold:.4f}")
    print(f"准确率：{accuracy:.4f}")
    print("="*50)
    
    return image_scores, best_threshold, best_f1, metrics


def print_classification_metrics(y_true, y_pred, y_scores):
    """打印详细的分类报告"""
    print("\n分类报告:")
    print(classification_report(y_true, y_pred, digits=4))
