"""
数据集统计分析工具（增强版）

功能：
1. 计算均值、方差、中位数等统计量
2. 生成像素值分布直方图
3. 生成箱线图展示数据分布
4. 保存所有统计图表

适合用于论文中的数据分布分析章节
"""

import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import yaml


class DatasetStatisticsAnalyzer:
    """数据集统计分析器"""
    
    def __init__(self, dataset_dir, output_dir='outputs/statistics'):
        """
        Args:
            dataset_dir: 数据集目录
            output_dir: 输出目录
        """
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.image_paths = []
        self.pixel_values = []
    
    def collect_images(self, image_extensions=None):
        """收集所有图像路径"""
        if image_extensions is None:
            image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
        
        for ext in image_extensions:
            self.image_paths.extend(list(self.dataset_dir.rglob(f'*{ext}')))
            self.image_paths.extend(list(self.dataset_dir.rglob(f'*{ext.upper()}')))
        
        print(f"共找到 {len(self.image_paths)} 张图像")
        return len(self.image_paths)
    
    def extract_pixel_values(self, sample_ratio=1.0):
        """
        提取像素值
        
        Args:
            sample_ratio: 采样比例（如果数据量太大，可以只采样一部分）
        """
        n_samples = int(len(self.image_paths) * sample_ratio)
        sampled_paths = np.random.choice(
            [str(p) for p in self.image_paths], 
            n_samples, 
            replace=False
        )
        
        print(f"正在从 {n_samples} 张图像中提取像素值...")
        
        all_pixels = []
        
        for img_path in tqdm(sampled_paths, desc="提取像素"):
            try:
                image = Image.open(img_path).convert('L')
                pixels = np.array(image, dtype=np.float64).flatten()
                
                # 归一化到 [0, 1]
                pixels = pixels / 255.0
                all_pixels.append(pixels)
                
            except Exception as e:
                print(f"\n警告：无法读取 {img_path}: {e}")
                continue
        
        self.pixel_values = np.concatenate(all_pixels)
        print(f"共提取了 {len(self.pixel_values):,} 个像素值")
    
    def calculate_statistics(self):
        """计算各种统计量"""
        if len(self.pixel_values) == 0:
            raise ValueError("请先提取像素值")
        
        stats = {
            'mean': np.mean(self.pixel_values),
            'std': np.std(self.pixel_values),
            'variance': np.var(self.pixel_values),
            'min': np.min(self.pixel_values),
            'max': np.max(self.pixel_values),
            'median': np.median(self.pixel_values),
            'q1': np.percentile(self.pixel_values, 25),  # 第一四分位数
            'q3': np.percentile(self.pixel_values, 75),  # 第三四分位数
            'skewness': self._calculate_skewness(),
            'kurtosis': self._calculate_kurtosis()
        }
        
        return stats
    
    def _calculate_skewness(self):
        """计算偏度"""
        n = len(self.pixel_values)
        mean = np.mean(self.pixel_values)
        std = np.std(self.pixel_values)
        return np.sum((self.pixel_values - mean) ** 3) / (n * std ** 3)
    
    def _calculate_kurtosis(self):
        """计算峰度"""
        n = len(self.pixel_values)
        mean = np.mean(self.pixel_values)
        std = np.std(self.pixel_values)
        return np.sum((self.pixel_values - mean) ** 4) / (n * std ** 4) - 3
    
    def plot_histogram(self, bins=100):
        """绘制像素值分布直方图"""
        plt.figure(figsize=(12, 6))
        
        sns.histplot(self.pixel_values, bins=bins, kde=True, color='skyblue')
        
        plt.xlabel('Normalized Pixel Value (0-1)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Pixel Value Distribution of Tire X-ray Dataset', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 添加均值和中位数线
        mean = np.mean(self.pixel_values)
        median = np.median(self.pixel_values)
        plt.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.4f}')
        plt.axvline(median, color='green', linestyle='--', label=f'Median: {median:.4f}')
        plt.legend()
        
        save_path = self.output_dir / 'pixel_distribution_histogram.png'
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 直方图已保存：{save_path}")
    
    def plot_boxplot(self):
        """绘制箱线图"""
        plt.figure(figsize=(8, 6))
        
        sns.boxplot(y=self.pixel_values, color='lightblue')
        
        plt.ylabel('Normalized Pixel Value (0-1)', fontsize=12)
        plt.title('Pixel Value Box Plot', fontsize=14)
        plt.grid(True, alpha=0.3, axis='y')
        
        # 标注关键统计量
        stats = self.calculate_statistics()
        plt.axhline(stats['mean'], color='red', linestyle='--', label=f"Mean: {stats['mean']:.4f}")
        plt.axhline(stats['median'], color='green', linestyle='-.', label=f"Median: {stats['median']:.4f}")
        plt.legend()
        
        save_path = self.output_dir / 'pixel_distribution_boxplot.png'
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 箱线图已保存：{save_path}")
    
    def plot_cumulative_distribution(self):
        """绘制累积分布函数图"""
        sorted_pixels = np.sort(self.pixel_values)
        cumulative = np.arange(1, len(sorted_pixels) + 1) / len(sorted_pixels)
        
        plt.figure(figsize=(12, 6))
        
        plt.plot(sorted_pixels, cumulative, linewidth=2, color='blue')
        
        plt.xlabel('Normalized Pixel Value (0-1)', fontsize=12)
        plt.ylabel('Cumulative Probability', fontsize=12)
        plt.title('Cumulative Distribution Function (CDF)', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 添加中位数线
        median = np.median(self.pixel_values)
        plt.axvline(median, color='red', linestyle='--', label=f'Median: {median:.4f}')
        plt.legend()
        
        save_path = self.output_dir / 'cumulative_distribution.png'
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 累积分布图已保存：{save_path}")
    
    def save_statistics_report(self, stats):
        """保存统计报告"""
        report_path = self.output_dir / 'statistics_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("轮胎 X 光数据集详细统计报告\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"数据集路径：{self.dataset_dir.absolute()}\n")
            f.write(f"图像数量：{len(self.image_paths):,}\n")
            f.write(f"像素总数：{len(self.pixel_values):,}\n\n")
            
            f.write("基本统计量:\n")
            f.write("-" * 60 + "\n")
            f.write(f"均值 (Mean):          {stats['mean']:.6f}\n")
            f.write(f"标准差 (Std):         {stats['std']:.6f}\n")
            f.write(f"方差 (Variance):      {stats['variance']:.6f}\n")
            f.write(f"最小值 (Min):         {stats['min']:.6f}\n")
            f.write(f"最大值 (Max):         {stats['max']:.6f}\n")
            f.write(f"中位数 (Median):      {stats['median']:.6f}\n")
            f.write(f"第一四分位数 (Q1):    {stats['q1']:.6f}\n")
            f.write(f"第三四分位数 (Q3):    {stats['q3']:.6f}\n\n")
            
            f.write("分布形态:\n")
            f.write("-" * 60 + "\n")
            f.write(f"偏度 (Skewness):      {stats['skewness']:.6f}\n")
            f.write(f"峰度 (Kurtosis):      {stats['kurtosis']:.6f}\n\n")
            
            f.write("建议配置 (复制到 config.yaml):\n")
            f.write("-" * 60 + "\n")
            f.write(f"mean: [{stats['mean']:.6f}, {stats['mean']:.6f}, {stats['mean']:.6f}]\n")
            f.write(f"std:  [{stats['std']:.6f}, {stats['std']:.6f}, {stats['std']:.6f}]\n\n")
            
            f.write("说明:\n")
            f.write("-" * 60 + "\n")
            f.write("- 偏度 > 0: 右偏分布（长尾在右侧）\n")
            f.write("- 偏度 < 0: 左偏分布（长尾在左侧）\n")
            f.write("- 峰度 > 0: 尖峰分布（比正态分布更陡峭）\n")
            f.write("- 峰度 < 0: 平峰分布（比正态分布更平缓）\n")
            f.write("- 轮胎 X 光图像通常呈现双峰分布（背景和轮胎结构）\n")
            f.write("="*60 + "\n")
        
        print(f"✅ 统计报告已保存：{report_path}")
    
    def run_full_analysis(self, sample_ratio=1.0):
        """运行完整分析流程"""
        print("="*60)
        print("开始数据集完整统计分析")
        print("="*60)
        
        # 1. 收集图像
        print("\n[1/4] 收集图像文件...")
        self.collect_images()
        
        if len(self.image_paths) == 0:
            print("错误：未找到任何图像文件！")
            return
        
        # 2. 提取像素值
        print("\n[2/4] 提取像素值...")
        self.extract_pixel_values(sample_ratio)
        
        # 3. 计算统计量
        print("\n[3/4] 计算统计量...")
        stats = self.calculate_statistics()
        
        print("\n" + "="*60)
        print("统计结果摘要")
        print("="*60)
        print(f"均值：   {stats['mean']:.6f}")
        print(f"标准差： {stats['std']:.6f}")
        print(f"中位数： {stats['median']:.6f}")
        print(f"偏度：   {stats['skewness']:.6f}")
        print(f"峰度：   {stats['kurtosis']:.6f}")
        print("="*60)
        
        # 4. 生成可视化图表
        print("\n[4/4] 生成可视化图表...")
        self.plot_histogram()
        self.plot_boxplot()
        self.plot_cumulative_distribution()
        
        # 5. 保存报告
        self.save_statistics_report(stats)
        
        print("\n" + "="*60)
        print("✅ 完整分析完成！")
        print(f"📊 所有结果已保存到：{self.output_dir.absolute()}")
        print("="*60)
        
        return stats


def main():
    print("="*60)
    print("轮胎 X 光数据集统计分析工具（增强版）")
    print("="*60)
    
    # 选择数据集目录
    print("\n请选择要统计的数据集:")
    print("1. 全部数据 (data/)")
    print("2. 训练集正常样本 (data/train/good/)")
    print("3. 测试集正常样本 (data/test/good/)")
    print("4. 自定义目录")
    
    choice = input("\n请输入选择 (1/2/3/4): ").strip()
    
    base_dir = Path(__file__).parent.parent
    
    if choice == '1':
        dataset_dir = base_dir / 'data'
    elif choice == '2':
        dataset_dir = base_dir / 'data' / 'train' / 'good'
    elif choice == '3':
        dataset_dir = base_dir / 'data' / 'test' / 'good'
    elif choice == '4':
        dataset_dir = input("请输入数据集目录路径：").strip()
        dataset_dir = Path(dataset_dir)
    else:
        print("无效的选择！")
        return
    
    # 检查目录是否存在
    if not dataset_dir.exists():
        print(f"❌ 错误：目录不存在：{dataset_dir}")
        return
    
    # 询问采样比例
    print(f"\n当前数据集：{dataset_dir}")
    sample_input = input("是否使用全部数据进行统计？(y/n, 默认 y): ").strip().lower()
    
    if sample_input == 'n':
        ratio_input = input("请输入采样比例 (0.1-1.0, 默认 1.0): ").strip()
        try:
            sample_ratio = float(ratio_input)
            if sample_ratio < 0.1 or sample_ratio > 1.0:
                print("采样比例应在 0.1-1.0 之间，使用默认值 1.0")
                sample_ratio = 1.0
        except:
            print("输入无效，使用默认值 1.0")
            sample_ratio = 1.0
    else:
        sample_ratio = 1.0
    
    # 创建分析器并运行
    output_dir = base_dir / 'outputs' / 'statistics'
    analyzer = DatasetStatisticsAnalyzer(str(dataset_dir), str(output_dir))
    
    try:
        stats = analyzer.run_full_analysis(sample_ratio)
        
        # 询问是否更新配置文件
        config_path = base_dir / 'configs' / 'config.yaml'
        if config_path.exists():
            update = input("\n是否更新 configs/config.yaml 中的均值方差？(y/n): ").strip().lower()
            if update == 'y':
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                config['data']['mean'] = [float(stats['mean'])] * 3
                config['data']['std'] = [float(stats['std'])] * 3
                
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
                
                print(f"✅ 配置文件已更新")
        
        print("\n下一步建议:")
        print("1. 查看 outputs/statistics/ 中的图表")
        print("2. 将这些统计图表用于论文的'数据集介绍'章节")
        print("3. 运行 python 01_extract_and_build_bank.py 开始正式实验")
        
    except Exception as e:
        print(f"\n❌ 发生错误：{e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
