"""
脚本 2: 测试评估并生成 ROC/PR 曲线

功能：
1. 加载已构建的 FAISS 记忆库
2. 提取测试集特征（5000 正常 + 2000 异常）
3. 计算异常分数
4. 寻找最佳 F1-Score 阈值
5. 生成 ROC、PR 曲线图

预计耗时：GPU 环境下约 30-60 分钟
"""

import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix
import seaborn as sns

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.dataset import TireXrayDataset, create_dataloader
from src.model import load_dinov3_model
from src.evaluate import extract_features_batch, evaluate_on_testset
from src.coreset import load_faiss_index


def plot_roc_and_pr_curves(metrics, save_dir):
    """绘制 ROC 和 PR 曲线"""
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # ==================== ROC 曲线 ====================
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))
    
    ax1.plot(
        metrics['fpr'], 
        metrics['tpr'],
        color='darkorange', 
        lw=2, 
        label=f'ROC Curve (AUC = {metrics["roc_auc"]:.4f})'
    )
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    ax1.legend(loc="lower right", fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    roc_path = save_dir / 'roc_curve.png'
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC 曲线已保存：{roc_path}")
    
    # ==================== PR 曲线 ====================
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
    
    ax2.plot(
        metrics['recall'], 
        metrics['precision'],
        color='blue', 
        lw=2, 
        label=f'PR Curve (Best F1 = {metrics["best_f1"]:.4f})'
    )
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curve', fontsize=14)
    ax2.legend(loc="lower left", fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    pr_path = save_dir / 'pr_curve.png'
    plt.savefig(pr_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"PR 曲线已保存：{pr_path}")
    
    # ==================== F1-Score 随阈值变化曲线 ====================
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6))
    
    # 需要重新计算 F1 对应的阈值
    thresholds = np.linspace(
        metrics['precision'].min(), 
        metrics['precision'].max(), 
        len(metrics['f1_scores'])
    )
    
    ax3.plot(thresholds, metrics['f1_scores'], color='green', lw=2)
    ax3.axvline(
        x=metrics['best_threshold'], 
        color='red', 
        linestyle='--', 
        label=f'Best Threshold = {metrics["best_threshold"]:.4f}'
    )
    ax3.set_xlabel('Threshold', fontsize=12)
    ax3.set_ylabel('F1-Score', fontsize=12)
    ax3.set_title('F1-Score vs Threshold', fontsize=14)
    ax3.legend(loc="best", fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    f1_path = save_dir / 'f1_score_curve.png'
    plt.savefig(f1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"F1-Score 曲线已保存：{f1_path}")
    
    return roc_path, pr_path, f1_path


def plot_confusion_matrix(y_true, y_pred, save_dir):
    """绘制混淆矩阵"""
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        annot_kws={'size': 14},
        cbar_kws={'label': 'Count'}
    )
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14)
    ax.set_xticklabels(['Normal', 'Anomaly'], fontsize=11)
    ax.set_yticklabels(['Normal', 'Anomaly'], fontsize=11)
    
    plt.tight_layout()
    cm_path = save_dir / 'confusion_matrix.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存：{cm_path}")


def main():
    print("="*60)
    print("步骤 2: 测试评估并生成 ROC/PR 曲线")
    print("="*60)
    
    # 加载配置
    config_path = 'configs/config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前设备：{device}")
    
    # ==================== 1. 加载 FAISS 索引 ====================
    print("\n[1/6] 加载 FAISS 记忆库...")
    memory_bank_path = config['output']['memory_bank_path']
    
    if not os.path.exists(memory_bank_path):
        print(f"错误：记忆库文件不存在：{memory_bank_path}")
        print("请先运行 01_extract_and_build_bank.py")
        return
    
    index = load_faiss_index(memory_bank_path)
    k = config['coreset']['knn_neighbors']
    
    # ==================== 2. 加载测试数据 ====================
    print("\n[2/6] 加载测试数据集...")
    test_dirs = []
    
    # 正常测试图像
    good_dir = config['data']['test_good_dir']
    if os.path.exists(good_dir):
        test_dirs.append(good_dir)
        print(f"  - 正常测试目录：{good_dir}")
    
    # 异常测试图像
    for anomaly_dir in config['data']['test_anomaly_dirs']:
        if os.path.exists(anomaly_dir):
            test_dirs.append(anomaly_dir)
            print(f"  - 异常测试目录：{anomaly_dir}")
    
    if len(test_dirs) == 0:
        print("错误：未找到任何测试数据，请检查配置文件")
        return
    
    # 创建合并的数据集（遍历所有测试目录）
    print("\n加载所有测试目录的图像...")
    all_test_images = []
    all_test_labels = []
    
    # 遍历每个测试目录
    for idx, test_dir in enumerate(test_dirs):
        # 判断是正常（0）还是异常（1）
        label = 0 if idx == 0 else 1  # 第一个目录是正常，后面是异常
        
        # 遍历目录下的所有图像
        for img_name in os.listdir(test_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                all_test_images.append(os.path.join(test_dir, img_name))
                all_test_labels.append(label)
    
    print(f"共加载 {len(all_test_images)} 张测试图像")
    print(f"  - 正常图像：{sum(1 for l in all_test_labels if l == 0)} 张")
    print(f"  - 异常图像：{sum(1 for l in all_test_labels if l == 1)} 张")
    
    # 使用 TireXrayDataset 的简化版本来加载这些图像
    from torch.utils.data import Dataset
    from torchvision import transforms
    from PIL import Image
    
    class TestDataset(Dataset):
        def __init__(self, image_paths, labels, config):
            self.image_paths = image_paths
            self.labels = labels
            self.transform = transforms.Compose([
                transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
                transforms.Normalize(
                    mean=config['data']['mean'],
                    std=config['data']['std']
                )
            ])
        
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert('L')
            image = self.transform(image)
            label = self.labels[idx]
            return image, label, img_path
    
    test_dataset = TestDataset(all_test_images, all_test_labels, config)
    
    batch_size = config['training']['batch_size']
    num_workers = 0  # 设置为 0 避免 pickle 问题（因为 TestDataset 是本地类）
    test_loader = create_dataloader(
        test_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers,
        shuffle=False
    )
    
    print(f"共加载 {len(test_dataset)} 张测试图像")
    
    # ==================== 3. 加载模型 ====================
    print("\n[3/6] 加载 DINOv3 模型...")
    model = load_dinov3_model(config_path, device)
    model.eval()
    
    # ==================== 4. 提取测试集特征 ====================
    print("\n[4/6] 批量提取测试集特征...")
    all_features, all_labels, all_paths = extract_features_batch(model, test_loader, device)
    
    # ==================== 5. 评估性能 ====================
    print("\n[5/6] 计算异常分数并评估...")
    image_scores, best_threshold, best_f1, metrics = evaluate_on_testset(
        all_features, 
        all_labels, 
        index, 
        k=k
    )
    
    # ==================== 6. 生成可视化图表 ====================
    print("\n[6/6] 生成评估图表...")
    metrics_dir = Path(config['output']['metrics_dir'])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # 绘制 ROC 和 PR 曲线
    plot_roc_and_pr_curves(metrics, metrics_dir)
    
    # 绘制混淆矩阵
    y_pred = (image_scores >= best_threshold).astype(int)
    plot_confusion_matrix(all_labels, y_pred, metrics_dir)
    
    # 保存详细结果到文本文件
    results_path = metrics_dir / 'evaluation_results.txt'
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("评估结果汇总\n")
        f.write("="*60 + "\n\n")
        f.write(f"测试集总样本数：{len(all_labels)}\n")
        f.write(f"正常样本数：{np.sum(all_labels == 0)}\n")
        f.write(f"异常样本数：{np.sum(all_labels == 1)}\n\n")
        f.write(f"ROC AUC: {metrics['roc_auc']:.4f}\n")
        f.write(f"最佳 F1-Score: {best_f1:.4f}\n")
        f.write(f"最佳阈值：{best_threshold:.4f}\n")
        f.write(f"准确率：{np.mean(y_pred == all_labels):.4f}\n\n")
        
        # 按异常类型统计（如果有）
        f.write("\n建议：可以进一步按不同缺陷类型（气泡、断丝、搭接）分别统计性能指标\n")
    
    print(f"\n评估结果已保存到：{results_path}")
    
    print("\n" + "="*60)
    print("✅ 评估完成！所有结果已保存到 outputs/metrics/ 目录")
    print("="*60)
    print("\n下一步:")
    print("1. 查看 outputs/metrics/ 中的图表")
    print("2. 运行 03_inference_demo.py 生成单张图像的缺陷热力图")


if __name__ == '__main__':
    main()
