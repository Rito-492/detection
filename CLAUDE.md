# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 项目概述

这是 Ultralytics YOLO11 项目的 Fork，用于计算机视觉目标检测任务。本项目的主要改进是在 YOLO11  backbone 和 head 中添加了 **CBAM 注意力机制**。

**学生学号：55230316**

---

## 快速开始

### 安装依赖

```bash
pip install -e '.[dev]'
```

### 训练改进的 YOLO11-CBAM 模型

```bash
python student_55230316_cbam_train.py
```

### 推理测试

```bash
python student_55230316_predict.py
```

### 模型验证

```bash
python test_cbam_model.py
```

---

## 核心架构

### 目录结构

```
ultralytics/
├── cfg/              # 配置文件和参数解析
├── data/             # 数据加载和增强
├── engine/           # 训练/验证/预测引擎
├── models/           # 模型定义（YOLO/SAM/RTDETR 等）
├── nn/               # 神经网络核心组件
│   ├── modules/      # 基础模块（conv.py, block.py, head.py）
│   └── tasks.py      # 模型解析和构建（核心！）
├── optim/            # 优化器
└── utils/            # 工具函数
```

### 关键文件说明

| 文件 | 作用 | 修改频率 |
|------|------|----------|
| `ultralytics/nn/tasks.py` | 模型解析核心，将 YAML 配置转换为 PyTorch 模型 | 高 |
| `ultralytics/nn/modules/conv.py` | 卷积层和注意力模块（含 CBAM 实现） | 中 |
| `ultralytics/nn/modules/block.py` | 网络块（C2f, C3k2, SPPF 等） | 中 |
| `ultralytics/nn/modules/head.py` | 检测头（Detect, Segment, Pose） | 中 |
| `ultralytics/cfg/models/11/*.yaml` | YOLO11 模型配置文件 | 高 |

---

## 本项目改进内容

### CBAM 注意力模块位置

在 YOLO11 的 backbone 和 head 中添加了 5 个 CBAM 模块：

| 位置 | 层索引 | 作用 |
|------|--------|------|
| Backbone P3 | Layer 3 | 增强浅层特征，提升小目标检测 |
| Backbone P4 | Layer 6 | 增强中层特征 |
| Backbone P5 | Layer 11 | 增强深层语义特征 |
| Head P3 | Layer 20 | 融合后的浅层特征增强 |
| Head P4 | Layer 24 | 融合后的中层特征增强 |

### 核心代码修改

**文件：`ultralytics/nn/tasks.py`**

1. 添加 CBAM 导入（约第 21 行）
2. 添加 CBAM 特殊处理逻辑（约第 1670 行）：
```python
elif m is CBAM:
    c2 = ch[f]
    args = [ch[f], *args]
```

**文件：`ultralytics/cfg/models/11/yolo11_cbam_55230316.yaml`**

定义改进的 YOLO11-CBAM 模型结构，包含 5 个 CBAM 模块。

---

## 常用命令

### 开发模式安装
```bash
pip install -e '.[dev]'
```

### 运行训练
```bash
python student_55230316_cbam_train.py
```

### 运行验证
```bash
yolo val model=runs/detect/yolo11_cbam_55230316/weights/best.pt data=coco8.yaml
```

### 运行推理
```bash
yolo predict model=runs/detect/yolo11_cbam_55230316/weights/best.pt source=bus.jpg
```

### 导出 ONNX
```bash
yolo export model=runs/detect/yolo11_cbam_55230316/weights/best.pt format=onnx
```

### 运行测试
```bash
pytest tests/ -v
```

---

## 模型配置格式

YOLO 模型使用 YAML 文件定义架构：

```yaml
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]      # 标准卷积
  - [-1, 1, CBAM, [7]]             # CBAM 注意力（kernel_size=7）
  - [-1, 2, C3k2, [256, False, 0.25]]

head:
  - [[20, 24, 27], 1, Detect, [nc]]  # Detect 头
```

---

## 添加新模块的通用流程

1. **在 `ultralytics/nn/modules/` 下创建新类**（如 `conv.py` 或 `block.py`）
2. **在 `ultralytics/nn/tasks.py` 中**：
   - 添加新类的导入
   - 如有必要，添加特殊处理逻辑（参考 CBAM 的处理方式）
3. **在 YAML 配置文件中使用新模块**

---

## 参考资料

- 官方文档：https://docs.ultralytics.com/
- CBAM 论文：https://arxiv.org/abs/1807.06521
- 详细库结构说明：参见 `ultralytics 库结构说明.md`
- 改进说明文档：参见 `改进说明_55230316.md`
