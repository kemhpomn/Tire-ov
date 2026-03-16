# 快速开始指南 ⚡

## 🎯 5 分钟快速上手

### 第一步：安装依赖 (2 分钟)

```bash
# 进入项目目录
cd D:\Tire_Xray_Anomaly_Detection_DINOv3

# 创建虚拟环境
conda create -n tire_dinov3 python=3.10 -y
conda activate tire_dinov3

# 安装核心依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets faiss-gpu numpy opencv-python scikit-learn matplotlib seaborn pyyaml tqdm
```

### 第二步：准备数据 (1 分钟)

将你的轮胎 X 光图像放入以下目录：

```
data/
├── train/
│   └── good/           ← 放入 45,000 张正常图像
└── test/
    ├── good/           ← 放入 5,000 张正常测试图
    └── anomaly_*/      ← 放入 2,000 张异常图像
```

### 第三步：运行实验 (看 GPU 性能)

```bash
# 步骤 1: 构建记忆库（约 2-4 小时）
python 01_extract_and_build_bank.py

# 步骤 2: 评估并生成曲线（约 30-60 分钟）
python 02_test_and_evaluate.py

# 步骤 3: 生成热力图（即时）
python 03_inference_demo.py
```

---

## 📊 预期输出

运行完成后，查看以下结果：

```
outputs/
├── metrics/
│   ├── roc_curve.png          ← ROC 曲线（论文必备）
│   ├── pr_curve.png           ← PR 曲线
│   ├── f1_score_curve.png     ← F1-Score 曲线
│   ├── confusion_matrix.png   ← 混淆矩阵
│   └── evaluation_results.txt ← 详细指标
└── heatmaps/
    ├── 0001_xxx_anomaly.png   ← 缺陷热力图
    └── ...
```

---

## 🔧 常见问题速查

### ❌ CUDA Out of Memory
编辑 `configs/config.yaml`，减小 batch_size：
```yaml
training:
  batch_size: 16  # 改为 16 或 8
```

### ❌ FAISS 安装失败
使用 CPU 版本：
```bash
pip install faiss-cpu
```

### ❌ 模型下载太慢
在命令行设置镜像：
```bash
# Windows PowerShell
$env:HF_ENDPOINT="https://hf-mirror.com"
python 01_extract_and_build_bank.py

# Linux/Mac
export HF_ENDPOINT=https://hf-mirror.com
python 01_extract_and_build_bank.py
```

---

## 📈 查看实验结果

### 查看 GPU 使用情况
```bash
# Windows PowerShell
nvidia-smi

# 或持续监控
watch -n 1 nvidia-smi
```

### 查看处理进度
所有脚本都带有进度条显示，实时查看剩余时间。

---

## 💡 下一步建议

1. **先跑通 baseline**：用默认配置完整运行一遍
2. **调整参数**：修改 `config.yaml` 中的超参数
3. **对比实验**：尝试不同的采样方法（random/kmeans/kcenter）
4. **可视化分析**：挑选典型缺陷图像生成热力图

---

## 📞 需要帮助？

如果遇到错误：
1. 查看终端输出的完整错误信息
2. 检查 `outputs/` 目录下是否有日志文件
3. 确保数据路径正确
4. 确认显存足够（建议 ≥ 8GB）

---

**祝你实验顺利！🚀**
