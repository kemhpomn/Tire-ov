"""
可视化模块
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path


def generate_anomaly_map(patch_scores, image_size=256, patch_size=16):
    num_patches = len(patch_scores)

    num_patches_per_dim_theory = image_size // patch_size
    expected_patches = num_patches_per_dim_theory * num_patches_per_dim_theory

    # 【修复重点 3】处理 DINO/ViT 模型中携带的 CLS Token 和 Register Tokens (例如长度为 257 或 261)
    # 因为空间 token 往往排在最后，所以我们只截取最后的 expected_patches 个元素。
    if num_patches > expected_patches:
        patch_scores = patch_scores[-expected_patches:]
        num_patches = expected_patches

    if num_patches == expected_patches:
        score_grid = patch_scores.reshape(num_patches_per_dim_theory, num_patches_per_dim_theory)
    else:
        # 兜底：寻找最接近正方形的尺寸进行 reshape
        sqrt_num = int(np.sqrt(num_patches))
        rows = cols = sqrt_num
        for i in range(sqrt_num, 0, -1):
            if num_patches % i == 0:
                rows = i
                cols = num_patches // i
                break

        if rows * cols != num_patches:
            score_grid = patch_scores[:expected_patches].reshape(
                num_patches_per_dim_theory, num_patches_per_dim_theory
            )
        else:
            score_grid = patch_scores.reshape(rows, cols)

    # 双线性插值放大到原图尺寸
    anomaly_map = cv2.resize(
        score_grid,
        (image_size, image_size),
        interpolation=cv2.INTER_LINEAR
    )

    return anomaly_map


def apply_jet_colormap(anomaly_map):
    # 确保 map 的数值在[0, 1] 之间以避免计算溢出
    anomaly_map = np.clip(anomaly_map, 0, 1)
    heatmap_uint8 = (anomaly_map * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    return heatmap


def overlay_heatmap_on_image(original_image, heatmap, alpha=0.5):
    if len(original_image.shape) == 2:
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    else:
        original_rgb = original_image

    overlay = cv2.addWeighted(original_rgb, 1 - alpha, heatmap, alpha, 0)
    return overlay


def save_anomaly_visualization(
        original_image_path,
        patch_scores,
        save_path,
        image_size=256,
        patch_size=16,
        normalize=True
):
    original_image = cv2.imread(str(original_image_path))
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image = cv2.resize(original_image, (image_size, image_size))

    anomaly_map = generate_anomaly_map(patch_scores, image_size, patch_size)

    # 如果开启局部归一化（批量推理时应当关闭它，否则正常的图也会全是红斑）
    if normalize:
        anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)

    heatmap = apply_jet_colormap(anomaly_map)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = overlay_heatmap_on_image(original_image, heatmap, alpha=0.5)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(original_image)
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis('off')

    im1 = axes[1].imshow(anomaly_map, cmap='jet', vmin=0, vmax=1)  # 强制固定比例尺0~1
    axes[1].set_title("Anomaly Map", fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1])

    axes[2].imshow(heatmap)
    axes[2].set_title("Heatmap", fontsize=12)
    axes[2].axis('off')

    axes[3].imshow(overlay)
    axes[3].set_title("Overlay", fontsize=12)
    axes[3].axis('off')

    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 生成论文用单张干净热力图
    heatmap_only_path = save_path.parent / f"{save_path.stem}_heatmap_only.png"
    fig_heatmap, ax = plt.subplots(1, 1, figsize=(5, 5))
    im = ax.imshow(anomaly_map, cmap='jet', vmin=0, vmax=1)  # 强制固定比例尺
    ax.axis('off')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(heatmap_only_path, dpi=300, bbox_inches='tight')
    plt.close()


def batch_save_heatmaps(all_paths, all_patch_scores, output_dir, **kwargs):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, (img_path, scores) in enumerate(zip(all_paths, all_patch_scores)):
        img_name = Path(img_path).stem
        save_path = output_dir / f"{i:04d}_{img_name}.png"
        save_anomaly_visualization(img_path, scores, save_path, **kwargs)