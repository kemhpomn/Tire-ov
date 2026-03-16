"""
统计数据集的均值和方差

功能：
1. 遍历指定目录下的所有图像
2. 计算整个数据集的通道均值和标准差
3. 输出统计结果并保存为配置文件

这对于论文中描述数据预处理步骤非常重要！
"""

import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import yaml


def calculate_dataset_statistics(dataset_dir, image_extensions=None):
    """
    计算数据集的均值和标准差
    
    Args:
        dataset_dir: 数据集根目录
        image_extensions: 支持的图像扩展名列表
        
    Returns:
        mean: 通道均值
        std: 通道标准差
        n_images: 图像总数
    """
    if image_extensions is None:
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    
    # 收集所有图像路径
    image_paths = []
    for ext in image_extensions:
        # 递归查找所有子文件夹
        image_paths.extend(Path(dataset_dir).rglob(f'*{ext}'))
        image_paths.extend(Path(dataset_dir).rglob(f'*{ext.upper()}'))
    
    image_paths = [str(p) for p in image_paths]
    n_images = len(image_paths)
    
    if n_images == 0:
        raise ValueError(f"在目录 {dataset_dir} 中未找到任何图像文件")
    
    print(f"发现 {n_images} 张图像")
    
    # 初始化累加器
    # 对于灰度图，我们只计算单通道的统计值
    pixel_sum = 0.0
    pixel_squared_sum = 0.0
    total_pixels = 0
    
    print("开始统计像素值...")
    
    for img_path in tqdm(image_paths, desc="处理图像"):
        try:
            # 读取图像并转为灰度
            image = Image.open(img_path).convert('L')
            image_array = np.array(image, dtype=np.float64)
            
            # 归一化到 [0, 1]
            image_array = image_array / 255.0
            
            # 累加
            pixel_sum += np.sum(image_array)
            pixel_squared_sum += np.sum(image_array ** 2)
            total_pixels += image_array.size
            
        except Exception as e:
            print(f"\n警告：无法读取图像 {img_path}: {e}")
            continue
    
    # 计算均值和方差
    mean = pixel_sum / total_pixels
    variance = (pixel_squared_sum / total_pixels) - (mean ** 2)
    std = np.sqrt(variance)
    
    print(f"\n{'='*60}")
    print("数据集统计结果")
    print(f"{'='*60}")
    print(f"图像总数：{n_images}")
    print(f"总像素数：{total_pixels:,}")
    print(f"均值 (Mean): {mean:.6f}")
    print(f"标准差 (Std): {std:.6f}")
    print(f"方差 (Variance): {variance:.6f}")
    print(f"{'='*60}")
    
    return mean, std, n_images


def update_config_file(config_path, mean, std):
    """
    更新配置文件中的均值和方差
    
    Args:
        config_path: 配置文件路径
        mean: 计算出的均值
        std: 计算出的标准差
    """
    # 读取配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 更新均值和方差（复制 3 份以匹配 RGB 三通道）
    config['data']['mean'] = [float(mean)] * 3
    config['data']['std'] = [float(std)] * 3
    
    # 保存回配置文件
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    
    print(f"\n✅ 配置文件已更新：{config_path}")
    print(f"   mean: [{mean:.6f}, {mean:.6f}, {mean:.6f}]")
    print(f"   std: [{std:.6f}, {std:.6f}, {std:.6f}]")


def save_statistics_report(save_path, mean, std, n_images, dataset_dir):
    """
    保存统计报告
    
    Args:
        save_path: 保存路径
        mean: 均值
        std: 标准差
        n_images: 图像数量
        dataset_dir: 数据集目录
    """
    report_path = Path(save_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("轮胎 X 光数据集统计报告\n")
        f.write("="*60 + "\n\n")
        f.write(f"统计时间：{Path(dataset_dir).absolute()}\n")
        f.write(f"图像总数：{n_images:,}\n\n")
        f.write("数据统计:\n")
        f.write(f"  均值 (Mean): {mean:.6f}\n")
        f.write(f"  标准差 (Std): {std:.6f}\n")
        f.write(f"  方差 (Variance): {std**2:.6f}\n\n")
        f.write("建议配置 (复制到 config.yaml):\n")
        f.write(f"  mean: [{mean:.6f}, {mean:.6f}, {mean:.6f}]\n")
        f.write(f"  std: [{std:.6f}, {std:.6f}, {std:.6f}]\n\n")
        f.write("="*60 + "\n")
        f.write("说明:\n")
        f.write("- 这些统计值基于完整的轮胎 X 光数据集计算\n")
        f.write("- 用于 DINOv3 模型的输入归一化\n")
        f.write("- 相比 ImageNet 的默认值，更能反映轮胎数据的分布特性\n")
        f.write("="*60 + "\n")
    
    print(f"\n📄 统计报告已保存：{report_path}")


def main():
    print("="*60)
    print("轮胎 X 光数据集均值方差统计工具")
    print("="*60)
    
    # 选择统计模式
    print("\n请选择统计模式:")
    print("1. 统计整个 data/ 目录（推荐：使用所有数据）")
    print("2. 仅统计 train/good/ 目录（训练集正常样本）")
    print("3. 自定义目录")
    
    choice = input("\n请输入选择 (1/2/3): ").strip()
    
    base_dir = Path(__file__).parent.parent  # 指向项目根目录
    
    if choice == '1':
        dataset_dir = base_dir / 'data'
    elif choice == '2':
        dataset_dir = base_dir / 'data' / 'train' / 'good'
    elif choice == '3':
        dataset_dir = input("请输入数据集目录路径：").strip()
        dataset_dir = Path(dataset_dir)
    else:
        print("无效的选择！")
        return
    
    # 检查目录是否存在
    if not dataset_dir.exists():
        print(f"错误：目录不存在：{dataset_dir}")
        return
    
    print(f"\n正在统计目录：{dataset_dir}")
    
    try:
        # 计算统计值
        mean, std, n_images = calculate_dataset_statistics(str(dataset_dir))
        
        # 保存统计报告
        report_path = base_dir / 'outputs' / 'dataset_statistics.txt'
        save_statistics_report(str(report_path), mean, std, n_images, str(dataset_dir))
        
        # 询问是否更新配置文件
        config_path = base_dir / 'configs' / 'config.yaml'
        if config_path.exists():
            update = input("\n是否更新 configs/config.yaml 中的均值方差？(y/n): ").strip().lower()
            if update == 'y':
                update_config_file(str(config_path), mean, std)
        else:
            print(f"\n提示：配置文件不存在：{config_path}")
        
        print("\n" + "="*60)
        print("✅ 统计完成！")
        print("="*60)
        print("\n下一步:")
        print("1. 查看 outputs/dataset_statistics.txt 查看详细报告")
        print("2. 如果满意，可以直接运行 01_extract_and_build_bank.py 开始实验")
        
    except Exception as e:
        print(f"\n❌ 发生错误：{e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
