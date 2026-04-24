# ============================================================================
# YOLO11 目标检测训练脚本
# 学生学号：55230316
# 任务：个人任务一 - 算力环境部署与核心算法跑通
# 方向：计算机视觉 - 基于 YOLO11 的目标检测
# ============================================================================

from ultralytics import YOLO

# ============================================================================
# 一、环境配置
# ============================================================================

# 指定 GPU 设备编号
# Kaggle/Colab 等云平台通常只提供单 GPU，使用 device=0 即可
# 本地多卡环境可根据 nvidia-smi 输出选择合适的 device
DEVICE = 0  # GPU 设备编号

# 训练超参数配置
# 这些参数会影响训练速度、收敛效果和最终精度
EPOCHS = 100      # 训练轮次：数据量小可以适当增加轮次
IMG_SIZE = 640    # 输入尺寸：640 是 YOLO 系列的默认尺寸
BATCH_SIZE = 16   # 批次大小：根据 GPU 显存调整，显存小则减小 batch
LEARNING_RATE = 0.01  # 初始学习率

# 数据集选择
# DroneVehicle: 无人机视角车辆检测数据集（可见光 + 红外双模态）
DATASET = "/home/r1t0/workspace/30_projects/10_academic_labs/智能算法综合实践/YOLOv8/ultralytics/cfg/datasets/DroneVehicle.yaml"

# ============================================================================
# 二、模型加载
# ============================================================================

"""
YOLO11 模型系列说明：
- yolo11n.pt : nano 版本，最小最快，适合边缘设备
- yolo11s.pt : small 版本，速度与精度平衡
- yolo11m.pt : medium 版本，中等规模
- yolo11l.pt : large 版本，较高精度
- yolo11x.pt : extra-large 版本，最高精度但最慢

训练策略选择：
1. 从头训练 (train from scratch): 适合数据集与 COCO 差异大的场景
2. 迁移学习 (transfer learning): 加载 COCO 预训练权重，适合大多数场景
"""

# 方案 A：加载预训练权重（推荐）
# 预训练模型已经在 COCO 数据集上学习过通用特征，收敛更快
model = YOLO("yolo11n.pt")

# 方案 B：从配置文件创建新模型（从头训练）
# 取消下面这行的注释来使用此方案
# model = YOLO("yolo11n.yaml")

# ============================================================================
# 三、模型训练
# ============================================================================

print(f"学生学号：55230316 - 开始训练 YOLO11 目标检测模型")
print(f"数据集：{DATASET}")
print(f"设备：CUDA {DEVICE}")

# 开始训练
# 训练结果会保存在 runs/detect/train/ 目录下
# 包含：权重文件、训练曲线、验证结果等
results = model.train(
    # 数据相关参数
    data=DATASET,           # 数据集配置文件路径

    # 训练超参数
    epochs=EPOCHS,          # 训练轮次
    imgsz=IMG_SIZE,         # 输入图像尺寸
    batch=BATCH_SIZE,       # 每批次样本数

    # 优化器参数
    lr0=LEARNING_RATE,      # 初始学习率
    amp=False,              # 关闭自动混合精度（避免数值不稳定）

    # 设备参数
    device=DEVICE,          # GPU 设备编号

    # 预训练参数
    pretrained=True,        # 加载预训练权重（如果模型是 .pt 文件）

    # 数据增强参数（YOLO 内置多种增强策略）
    # 默认启用：Mosaic、MixUp、HSV 色彩变换、翻转等

    # 日志与保存
    project="runs/detect",  # 项目输出目录
    name="train_55230316",  # 实验名称（会创建 train_55230316 子目录）
    exist_ok=True,          # 允许覆盖已存在的实验目录

    # 其他实用参数
    workers=4,              # 数据加载线程数
    verbose=True,           # 输出详细训练日志
    save=True,              # 保存训练结果
    save_period=10,         # 每 10 个 epoch 保存一次检查点
)

# ============================================================================
# 四、模型验证
# ============================================================================

print("训练完成！开始验证模型性能...")

# 在验证集上评估模型
# 输出指标：
# - mAP50: IoU=0.5 时的平均精度
# - mAP50-95: IoU 从 0.5 到 0.95 的平均精度（更严格的指标）
# - Precision: 查准率
# - Recall: 召回率
val_results = model.val(
    data=DATASET,
    device=DEVICE,
    splits="val"  # 使用验证集进行评估
)

# 打印关键指标
print(f"\n===== 模型验证结果 (学号：55230316) =====")
print(f"mAP50-95: {val_results.box.map50-95:.4f}")
print(f"mAP50: {val_results.box.map50:.4f}")
print(f"Precision: {val_results.box.mp:.4f}")
print(f"Recall: {val_results.box.mr:.4f}")

# ============================================================================
# 五、推理测试
# ============================================================================

print("\n执行推理测试...")

# 对测试图像进行目标检测
# 可以替换为本地图像路径或 URL
test_image = "https://ultralytics.com/images/bus.jpg"
det_results = model(test_image, device=DEVICE)

# 处理检测结果
for result in det_results:
    # 获取检测框信息
    boxes = result.boxes
    if boxes is not None:
        print(f"\n检测到 {len(boxes)} 个目标:")
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            class_name = result.names[cls_id]
            print(f"  [{i}] {class_name}: {conf:.2f} (坐标：{xyxy})")

    # 保存结果图像
    result.save("inference_result_55230316.jpg")
    print("\n推理结果已保存至：inference_result_55230316.jpg")

# ============================================================================
# 六、模型导出
# ============================================================================

print("\n导出模型至 ONNX 格式...")

# 导出模型为 ONNX 格式
# ONNX 是开放的模型格式，可用于多种推理引擎
# 其他支持的格式：torchscript, onnx, openvino, engine, coreml, tflite, paddle
onnx_path = model.export(
    format="onnx",
    dynamic=False,    # 固定输入尺寸
    simplify=True     # 简化模型结构
)

print(f"ONNX 模型已保存至：{onnx_path}")

print("\n" + "="*60)
print("训练流程完成！学生学号：55230316")
print("="*60)
