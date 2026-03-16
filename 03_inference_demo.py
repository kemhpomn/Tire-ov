"""
脚本 3: 单张图像推理并生成缺陷热力图
"""

import os
import sys
import yaml
import torch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.model import load_dinov3_model
from src.dataset import TireXrayNormalDataset
from src.evaluate import calculate_anomaly_score
from src.visualize import save_anomaly_visualization
from src.coreset import load_faiss_index


def preprocess_single_image(image_path, config):
    from torchvision import transforms
    original_image = Image.open(image_path).convert('L')
    original_image_np = np.array(original_image)

    transform = transforms.Compose([
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        transforms.Normalize(
            mean=config['data']['mean'],
            std=config['data']['std']
        )
    ])

    image_tensor = transform(original_image).unsqueeze(0)
    return image_tensor, original_image_np


def inference_single_image(model, index, image_path, config, device, save_dir=None):
    image_tensor, original_image = preprocess_single_image(image_path, config)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        features = model(image_tensor)

    features = features.squeeze(0).cpu().numpy()

    k = config['coreset']['knn_neighbors']
    patch_scores, image_score = calculate_anomaly_score(features, index, k)

    print(f"\n图像：{Path(image_path).name}")
    print(f"  异常分数：{image_score:.4f}")
    print(
        "  (注：单图推理缺乏全局对比，热力图会自动将图内相对最异常的区域标红。若本身是正常图像，红区仅为背景噪声。建议使用批量推理获取绝对热力图。)")

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{Path(image_path).stem}_anomaly.png"

        save_anomaly_visualization(
            original_image_path=image_path,
            patch_scores=patch_scores,
            save_path=save_path,
            image_size=config['data']['image_size'],
            patch_size=config['model']['patch_size'],
            normalize=True  # 单张图只能进行局部归一化
        )

    return image_score, patch_scores


def batch_inference(model, index, image_dir, config, device, save_dir):
    image_dir = Path(image_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(list(image_dir.glob(f'*{ext}')))

    print(f"\n发现 {len(image_paths)} 张图像")
    print("\n注意：为了获得绝对准确的冷暖热力图，文件夹中最好同时包含【正常】和【异常】图像。")

    print("\n第 1 步：计算所有图像的原始分数...")
    all_raw_scores = []
    all_patch_scores_raw = []
    all_features_list = []

    for i, img_path in enumerate(image_paths):
        image_tensor, _ = preprocess_single_image(str(img_path), config)
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            features = model(image_tensor)

        features = features.squeeze(0).cpu().numpy()
        all_features_list.append(features)

        k = config['coreset']['knn_neighbors']
        patch_scores_raw, image_score_raw = calculate_anomaly_score(features, index, k)

        all_raw_scores.append(image_score_raw)
        all_patch_scores_raw.append(patch_scores_raw)

        if (i + 1) % 50 == 0:
            print(f"  已处理 {i + 1}/{len(image_paths)} 张图像")

    # 【修复重点 1】使用所有 patch 级别的最小值来界定 0 分，使用所有图片最高分的 98% 界定满分 1 分
    all_raw_scores_np = np.array(all_raw_scores)
    all_patch_scores_np = np.concatenate(all_patch_scores_raw)

    min_score = np.percentile(all_patch_scores_np, 1)  # 背景的真实底噪
    max_score = np.percentile(all_raw_scores_np, 98)  # 异常点的真实峰值

    print(f"\n全局分数范围：[{min_score:.4f}, {max_score:.4f}]")
    print(f"用于归一化到绝对的 [0, 1] 区间")

    print("\n第 2 步：生成绝对归一化热力图...")
    results = []
    score_range = (min_score, max_score)

    for i, img_path in enumerate(image_paths):
        features = all_features_list[i]

        # 使用全局范围进行归一化
        patch_scores, image_score = calculate_anomaly_score(
            features, index, k, normalize=True, score_range=score_range
        )

        patch_scores = np.clip(patch_scores, 0, 1)
        image_score = np.clip(image_score, 0, 1)

        save_path = save_dir / f"{Path(img_path).stem}_anomaly.png"

        save_anomaly_visualization(
            original_image_path=str(img_path),
            patch_scores=patch_scores,
            save_path=save_path,
            image_size=config['data']['image_size'],
            patch_size=config['model']['patch_size'],
            normalize=False  # 【修复重点 2】绝对禁止局部重归一化，保持全局尺度！
        )

        results.append({
            'path': str(img_path),
            'score': float(image_score),
            'patch_scores': patch_scores
        })

        if (i + 1) % 10 == 0:
            print(f"[{i + 1}/{len(image_paths)}] 已保存：{img_path.name} (得分: {image_score:.4f})")

    # 保存结果统计
    stats_path = save_dir / 'inference_stats.txt'
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("推理结果统计\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"分数范围：[{min_score:.4f}, {max_score:.4f}]\n")
        f.write(f"归一化后分数范围：[0, 1]\n\n")
        for result in results:
            f.write(f"{Path(result['path']).name}: {result['score']:.4f}\n")

    return results


def main():
    print("=" * 60)
    print("步骤 3: 单张图像推理并生成缺陷热力图")
    print("=" * 60)

    config_path = 'configs/config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前设备：{device}")

    memory_bank_path = config['output']['memory_bank_path']
    if not os.path.exists(memory_bank_path):
        print(f"错误：记忆库不存在，请先运行 01_extract_and_build_bank.py")
        return

    index = load_faiss_index(memory_bank_path)
    model = load_dinov3_model(config_path, device)
    model.eval()

    print("\n请选择推理模式:")
    print("1. 单张图像推理 (适合局部快速查看)")
    print("2. 批量推理整个文件夹 (适合提取论文配图，冷暖对比准)")

    choice = input("\n请输入选择 (1/2): ").strip()

    heatmaps_dir = Path(config['output']['heatmaps_dir'])
    heatmaps_dir.mkdir(parents=True, exist_ok=True)

    if choice == '1':
        image_path = input("请输入图像路径：").strip()
        if not os.path.exists(image_path):
            return
        score, _ = inference_single_image(model, index, image_path, config, device, heatmaps_dir)
        print(f"\n✅ 处理完成！热力图已保存。")

    elif choice == '2':
        image_dir = input("请输入图像文件夹路径：").strip()
        if not os.path.exists(image_dir):
            return
        results = batch_inference(model, index, image_dir, config, device, heatmaps_dir)
        print(f"\n✅ 批量处理完成！热力图已保存到：{heatmaps_dir}")


if __name__ == '__main__':
    main()