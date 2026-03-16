"""
数据加载模块
负责读取轮胎 X 光图像数据集，并进行预处理：
- 灰度图转 RGB 三通道
- Resize 到 256x256
- 基于轮胎数据集统计的均值方差归一化
"""

import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import yaml
import torch


class GrayToRgb(object):
    """将灰度图转换为 RGB 三通道图像"""
    
    def __call__(self, img):
        if img.shape[0] == 1:
            return img.repeat(3, 1, 1)
        return img


class TireXrayDataset(Dataset):
    """轮胎 X 光图像数据集"""
    
    def __init__(self, root_dir, config_path='configs/config.yaml'):
        """
        Args:
            root_dir: 数据集根目录路径
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        
        # 遍历子文件夹收集图片路径
        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            
            # 判断是正常还是异常（根据文件夹名）
            label = 0 if class_name == 'good' else 1
            
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    self.image_paths.append(os.path.join(class_path, img_name))
                    self.labels.append(label)
        
        # 定义数据预处理流程
        self.transform = transforms.Compose([
            transforms.Resize((self.config['data']['image_size'], 
                             self.config['data']['image_size'])),
            transforms.ToTensor(),
            GrayToRgb(),
            transforms.Normalize(
                mean=self.config['data']['mean'],
                std=self.config['data']['std']
            )
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # 强制转为灰度图
        image = self.transform(image)
        label = self.labels[idx]
        
        return image, label, img_path


class TireXrayNormalDataset(Dataset):
    """仅包含正常图像的 dataset（用于构建记忆库）"""
    
    def __init__(self, root_dir, config_path='configs/config.yaml'):
        """
        Args:
            root_dir: 正常图像根目录
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.root_dir = root_dir
        self.image_paths = []
        
        # 只读取 good 文件夹下的图片
        good_dir = os.path.join(root_dir, 'good')
        if os.path.exists(good_dir):
            for img_name in os.listdir(good_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    self.image_paths.append(os.path.join(good_dir, img_name))
        else:
            # 如果 root_dir 本身就是 good 文件夹
            for img_name in os.listdir(root_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    self.image_paths.append(os.path.join(root_dir, img_name))
        
        self.transform = transforms.Compose([
            transforms.Resize((self.config['data']['image_size'], 
                             self.config['data']['image_size'])),
            transforms.ToTensor(),
            GrayToRgb(),
            transforms.Normalize(
                mean=self.config['data']['mean'],
                std=self.config['data']['std']
            )
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')
        image = self.transform(image)
        
        return image, img_path


def create_dataloader(dataset, batch_size=32, num_workers=4, shuffle=False):
    """创建 DataLoader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True
    )
