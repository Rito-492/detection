# 快速使用指南

## 📦 你已获得

✅ **55230316_基线数据探索与模型构建.ipynb** (34 KB)
- 完整的 Jupyter Notebook
- 包含 EDA + 轻量化基线模型
- 符合作业要求的 .ipynb 格式

## 🚀 三步开始

### 步骤 1: 安装依赖
```bash
pip install -r requirements.txt
```

### 步骤 2: 打开 Notebook
```bash
jupyter notebook 55230316_基线数据探索与模型构建.ipynb
```

### 步骤 3: 运行所有 Cell
- 从上到下执行每个 Cell
- 注意观察输出和生成的图表
- 等待 Part 8 的训练完成（约 30-60 分钟）

## 📊 会生成什么

运行完后自动生成 6 个文件：

1. **baseline_model.pth** - 训练好的模型
2. **eda_analysis.png** - EDA 分析图表
3. **rgb_thermal_samples.png** - RGB-Thermal 样本对比
4. **training_curve.png** - 训练曲线
5. **baseline_summary.txt** - 文本总结
6. **baseline_info.json** - 模型信息

## ✅ 符合作业要求

✓ 基于真实赛题数据 (DroneVehicle)
✓ EDA 详细分析
✓ 轻量化基线模型
✓ 完整的代码和文档
✓ 为后续改进提供对比

## 📝 可直接提交

这个 Notebook 可直接作为**第二次作业的基线报告**提交！

包含的内容：
- ✓ 数据清洗与特征分析 (EDA 部分)
- ✓ 基线模型的评估指标 (训练结果部分)
- ✓ 核心代码 (可从 Notebook 复制粘贴)
- ✓ 运行结果图表 (自动生成)

## 💡 后续改进方向

基于本基线，可进行以下对比：

1. **多模态对比**: RGB only vs RGB+Thermal
2. **架构对比**: 添加 CBAM 或 FPN
3. **融合策略对比**: Early/Mid/Late Fusion
4. **超参对比**: 学习率、批大小等

## ❓ 快速问答

**Q: 需要多少时间？**
A: 依赖安装 10 分钟，Notebook 运行 30-60 分钟

**Q: 需要 GPU 吗？**
A: GPU 更快，但 CPU 也可以跑（会慢）

**Q: 如何修改参数？**
A: 在 Notebook 中找"训练参数"部分修改

**Q: 能修改模型吗？**
A: 可以！所有代码都有注释，易于修改

**Q: 如何只用 RGB？**
A: 修改 `in_channels=3` 和 `use_thermal=False`

---

**开始使用**: `python -m jupyter notebook 55230316_基线数据探索与模型构建.ipynb`
