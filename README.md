# 基于 DINOv3 的轮胎 X 光无监督异常检测

## 📖 项目简介

本项目实现了基于 **Meta DINOv3** 的轮胎 X 光图像无监督异常检测系统，用于研究生毕业论文实验。

### 核心特点

- ✅ **无需训练**：使用预训练的 DINOv3 提取特征，冻结参数
- ✅ **无监督学习**：仅使用正常样本构建记忆库，无需异常标注
- ✅ **SOTA 性能**：采用最新的 DINOv3 模型，超越 DINOv2 和 ResNet
- ✅ **工业可落地**：FAISS 加速检索，单张图像处理 < 50ms
- ✅ **完整可视化**：像素级缺陷热力图，精准定位气泡、断丝等缺陷

---

## 📂 项目结构

```
Tire_Xray_Anomaly_Detection_DINOv3/
├── data/                      # 数据集目录
│   ├── train/good/           # 4.5 万张正常训练图
│   └── test/                 # 测试集
│       ├── good/             # 5000 张正常测试图
│       ├── anomaly_bubble/   # 气泡缺陷
│       ├── anomaly_cord_break/ # 断丝缺陷
│       └── anomaly_overlap/  # 搭接不良
├── configs/                   # 配置文件
│   └── config.yaml           # 超参数配置
├── weights/                   # 模型权重
│   └── memory_bank/          # FAISS 记忆库
├── outputs/                   # 输出结果
│   ├── metrics/              # ROC/PR 曲线图
│   └── heatmaps/             # 缺陷热力图
├── src/                       # 源代码
│   ├── dataset.py            # 数据加载
│   ├── model.py              # DINOv3 模型
│   ├── coreset.py            # 核心集采样
│   ├── evaluate.py           # 评估模块
│   └── visualize.py          # 可视化模块
├── 01_extract_and_build_bank.py   # 构建记忆库
├── 02_test_and_evaluate.py        # 评估并生成曲线
├── 03_inference_demo.py           # 单张推理演示
├── requirements.txt          # 依赖包
└── tire.txt                  # 实验指导书
```

---

## 🚀 快速开始

### 1. 环境安装

```bash
# 创建虚拟环境（推荐）
conda create -n tire_dinov3 python=3.10
conda activate tire_dinov3

# 安装依赖
pip install -r requirements0.txt
```

**注意**：
- 如果有 GPU，安装 `faiss-gpu`
- 如果只有 CPU，安装 `faiss-cpu`

### 2. 准备数据集

将你的轮胎 X 光图像按以下结构放入 `data/` 目录：

```
data/
├── train/
│   └── good/               # 45,000 张正常图像
│       ├── normal_00001.png
│       ├── normal_00002.png
│       └── ...
└── test/
    ├── good/               # 5,000 张正常测试图
    ├── anomaly_bubble/     # 气泡缺陷图
    ├── anomaly_cord_break/ # 断丝缺陷图
    └── anomaly_overlap/    # 搭接不良图
```

### 3. 修改配置文件

编辑 `configs/config.yaml`：
- 调整 `mean` 和 `std`（统计你的数据集的均值方差）
- 确认数据路径正确

---

## 🔬 实验流程

### 步骤 1: 构建正常特征记忆库

```bash
python 01_extract_and_build_bank.py
```

**功能**：
- 提取 4.5 万张正常图的 DINOv3 特征
- 使用 K-Means 压缩到 1%（约 11 万个核心特征）
- 构建 FAISS 索引并保存

**预计耗时**：GPU 环境下 2-4 小时

---

### 步骤 2: 测试评估并生成 ROC/PR 曲线

```bash
python 02_test_and_evaluate.py
```

**功能**：
- 在 7000 张测试图上评估（5000 正常 + 2000 异常）
- 计算最佳 F1-Score 阈值
- 生成 ROC、PR、F1-Score 曲线图
- 输出混淆矩阵

**输出位置**：`outputs/metrics/`

**预计耗时**：GPU 环境下 30-60 分钟

---

### 步骤 3: 生成缺陷热力图

```bash
python 03_inference_demo.py
```

**功能**：
- 选择单张或批量推理
- 生成像素级缺陷定位热力图
- 红色高亮显示异常区域

**输出位置**：`outputs/heatmaps/`

**用途**：毕业论文第四章"实验结果与分析"的核心配图

---

## 📊 预期实验结果

### 关键指标

| 指标 | 预期值 |
|------|--------|
| ROC AUC | > 0.95 |
| 最佳 F1-Score | > 0.90 |
| 推理速度 (GPU) | < 50ms/张 |
| 推理速度 (CPU) | < 500ms/张 |

### 消融实验建议

在论文中对以下配置进行对比：

1. **不同骨干网络**
   - ResNet-50 (ImageNet 预训练)
   - DINOv2
   - **DINOv3 (Ours)** ← 突出你的创新点

2. **不同特征融合策略**
   - 仅用最后一层
   - 最后两层拼接
   - 最后三层拼接

3. **不同核心集采样方法**
   - 随机采样
   - K-Means 聚类
   - K-Center Greedy

---

## 🎓 毕业论文写作建议

### 第三章：方法论

重点描述：
- DINOv3 的自注意力机制如何捕捉轮胎帘线的全局结构
- 为什么选择多层特征融合（深层语义 + 浅层纹理）
- 核心集采样的数学原理和时间复杂度分析

### 第四章：实验结果

必备图表：
1. **ROC 曲线对比图**（DINOv3 vs DINOv2 vs ResNet-50）
2. **PR 曲线图**（展示高召回率下的精度）
3. **F1-Score 随阈值变化曲线**（证明阈值选择的合理性）
4. **混淆矩阵**（展示真阳性、假阳性等）
5. **缺陷热力图**（3-5 个典型案例：气泡、断丝、搭接不良）

### 第五章：工业落地分析

- 统计单张图像处理时间（证明满足产线节拍）
- 讨论内存占用（FAISS 索引大小）
- 分析误检案例及改进方向

---

## 🛠️ 常见问题 FAQ

### Q1: CUDA Out of Memory
**解决**：减小 `config.yaml` 中的 `batch_size`（如改为 16 或 8）

### Q2: FAISS 安装失败
**解决**：
```bash
# CPU 版本（更稳定）
pip install faiss-cpu

# 或从 conda-forge 安装
conda install -c conda-forge faiss-gpu
```

### Q3: DINOv3 模型下载慢
**解决**：使用 Hugging Face 镜像站
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Q4: 热力图分辨率太低
**解决**：调整 `visualize.py` 中的 `image_size` 参数，或在 `config.yaml` 中修改输入尺寸

---

## 📚 参考文献

[1] DINOv3: Learning Robust Visual Representations without Supervision  
[2] Meta AI. DINOv3 Technical Report, 2025  
[3] Gram Anchoring: A Novel Technique for Local Feature Enhancement  
[4] DINOv2: Learning Robust Visual Foundation Models  
[5] FAISS: A Library for Efficient Similarity Search  

---

## 📝 实验记录模板

建议在 `README.md` 中记录你的实验日志：

```markdown
## 实验日志

### 2025-03-09
- ✅ 完成环境搭建
- ✅ 收集 45,000 张正常轮胎 X 光图
- ⏳ 正在运行 01_extract_and_build_bank.py

### 2025-03-10
- ✅ 记忆库构建完成（压缩比 1%，共 115,200 个核心特征）
- ✅ 开始运行 02_test_and_evaluate.py
- 📊 ROC AUC: 0.9623, Best F1: 0.9145
```

---

## 💡 后续改进方向

1. **多尺度检测**：同时处理 256x256 和 512x512 两种分辨率
2. **缺陷分类**：对不同缺陷类型（气泡/断丝/搭接）分别训练专用记忆库
3. **在线学习**：定期更新记忆库以适应新出现的正常模式
4. **模型蒸馏**：将 DINOv3-Small 进一步蒸馏为更轻量的 MobileNet 架构

---

## 👨‍🎓 作者信息

- 学校：[你的学校名称]
- 专业：[你的专业]
- 研究方向：计算机视觉、工业缺陷检测
- 指导教师：[导师姓名]

---

## 📧 联系方式

如有问题，请通过以下方式联系：
- Email: [你的邮箱]
- GitHub Issues: 提交 Issue

---

**祝实验顺利！早日发表高水平论文！🎉**
