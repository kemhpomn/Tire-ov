"""
核心集采样模块
负责将大规模特征库压缩为具有代表性的轻量级记忆库
实现多种采样策略供对比实验：
- 随机采样（Random Subsampling）
- K-Means 聚类中心
- K-Center Greedy（可选）
"""

import numpy as np
import faiss
from tqdm import tqdm


def random_subsampling(features, ratio=0.01):
    """
    随机下采样
    
    Args:
        features: 原始特征数组 [N, D]
        ratio: 采样比例
        
    Returns:
        centroids: 采样后的特征 [N*ratio, D]
    """
    n_samples = int(len(features) * ratio)
    indices = np.random.choice(len(features), n_samples, replace=False)
    centroids = features[indices]
    
    print(f"随机采样：{len(features)} -> {len(centroids)} 个特征点")
    return centroids


def kmeans_subsampling(features, ratio=0.01, niter=20, gpu=True):
    """
    使用 K-Means 聚类提取核心集
    
    Args:
        features: 原始特征数组 [N, D]
        ratio: 采样比例
        niter: K-Means 迭代次数
        gpu: 是否使用 GPU 加速
        
    Returns:
        centroids: K-Means 聚类中心 [N*ratio, D]
    """
    n_clusters = int(len(features) * ratio)
    d = features.shape[1]
    
    print(f"开始 K-Means 聚类：{len(features)} 个样本 -> {n_clusters} 个聚类中心")
    
    # 配置 K-Means
    kmeans = faiss.Kmeans(
        d=d,
        k=n_clusters,
        niter=niter,
        verbose=True,
        gpu=gpu,
        seed=42
    )
    
    # 训练并获取聚类中心
    kmeans.train(features)
    centroids = kmeans.centroids
    
    print(f"K-Means 聚类完成，得到 {len(centroids)} 个核心特征点")
    return centroids


def kcenter_greedy_subsampling(features, ratio=0.01):
    """
    K-Center Greedy 算法（简化版本）
    这是一种更智能的核心集选择方法，但计算开销较大
    
    Args:
        features: 原始特征数组 [N, D]
        ratio: 采样比例
        
    Returns:
        centroids: 选中的核心特征点
    """
    n_samples = int(len(features) * ratio)
    n_points = len(features)
    
    print(f"开始 K-Center Greedy 采样：{n_points} -> {n_samples}")
    
    # 初始化：随机选择第一个点
    selected_indices = [np.random.randint(n_points)]
    min_distances = np.full(n_points, np.inf)
    
    # 贪心选择后续点
    for i in tqdm(range(n_samples - 1), desc="K-Center Greedy"):
        # 计算所有点到已选点的最小距离
        current_center = features[selected_indices[-1]].reshape(1, -1)
        distances = np.linalg.norm(features - current_center, axis=1)
        min_distances = np.minimum(min_distances, distances)
        
        # 选择距离最大的点作为下一个中心
        next_idx = np.argmax(min_distances)
        selected_indices.append(next_idx)
    
    centroids = features[selected_indices]
    print(f"K-Center Greedy 完成，得到 {len(centroids)} 个核心特征点")
    return centroids


def build_faiss_index(centroids, use_gpu=False):
    """
    构建 FAISS 索引用于快速最近邻搜索
    
    Args:
        centroids: 记忆库特征 [M, D]
        use_gpu: 是否使用 GPU 加速
        
    Returns:
        index: FAISS 索引对象
    """
    d = centroids.shape[1]
    
    # 创建 L2 距离的扁平索引
    index = faiss.IndexFlatL2(d)
    
    # 可选：迁移到 GPU
    if use_gpu and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    
    # 添加向量到索引
    index.add(centroids)
    
    print(f"FAISS 索引构建完成，共 {index.ntotal} 个向量")
    return index


def save_faiss_index(index, save_path):
    """
    保存 FAISS 索引到磁盘
    
    Args:
        index: FAISS 索引对象
        save_path: 保存路径
    """
    # 如果索引在 GPU 上，先迁移回 CPU
    if faiss.get_num_gpus() > 0 and hasattr(index, 'get_data'):
        index = faiss.index_gpu_to_cpu(index)
    
    faiss.write_index(index, save_path)
    print(f"FAISS 索引已保存到：{save_path}")


def load_faiss_index(load_path):
    """
    从磁盘加载 FAISS 索引
    
    Args:
        load_path: 文件路径
        
    Returns:
        index: FAISS 索引对象
    """
    index = faiss.read_index(load_path)
    print(f"FAISS 索引已从 {load_path} 加载，共 {index.ntotal} 个向量")
    return index


class CoresetBuilder:
    """核心集构建器（封装完整流程）"""
    
    def __init__(self, config):
        """
        Args:
            config: 配置字典
        """
        self.config = config
        self.ratio = config['coreset']['subsampling_ratio']
        self.niter = config['coreset']['niter']
    
    def build(self, features, method='kmeans'):
        """
        构建核心集记忆库
        
        Args:
            features: 原始特征 [N, D]
            method: 采样方法 ('random', 'kmeans', 'kcenter')
            
        Returns:
            centroids: 核心集特征
            index: FAISS 索引
        """
        print(f"\n使用 {method} 方法构建核心集...")
        
        if method == 'random':
            centroids = random_subsampling(features, self.ratio)
        elif method == 'kmeans':
            centroids = kmeans_subsampling(features, self.ratio, self.niter)
        elif method == 'kcenter':
            centroids = kcenter_greedy_subsampling(features, self.ratio)
        else:
            raise ValueError(f"不支持的采样方法：{method}")
        
        # 构建 FAISS 索引
        index = build_faiss_index(centroids, use_gpu=False)
        
        return centroids, index
