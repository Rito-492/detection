# ============================================================================
# YOLO11-CBAM 改进版训练脚本
# 学生学号：55230316
# 任务：个人任务一 - 算力环境部署与核心算法跑通
# 方向：计算机视觉 - 基于改进 YOLO11 的目标检测
#
# 改进说明：
# 1. 在 backbone 和 head 的关键位置添加了 CBAM 注意力机制
# 2. CBAM = Channel Attention(通道注意力) + Spatial Attention(空间注意力)
# 3. 作用：让模型更关注重要特征区域，抑制无关特征，提升检测精度
#
# 网络改进点：
# - Backbone: 在 P3, P4, P5 特征输出后各添加一个 CBAM 模块
# - Head: 在多尺度特征融合后添加 CBAM 模块
# - 总共添加 5 个 CBAM 模块
#
# 参考文献：
# CBAM: Convolutional Block Attention Module (ECCV 2018)
# https://arxiv.org/abs/1807.06521
# ============================================================================

from ultralytics import YOLO
import torch

# ============================================================================
# 一、实验配置
# ============================================================================

# 学生信息
STUDENT_ID = "55230316"
EXPERIMENT_NAME = f"yolo11_cbam_{STUDENT_ID}"

# 设备配置
DEVICE = 0  # GPU 设备编号

# 训练参数
EPOCHS = 100          # 训练轮次
IMG_SIZE = 640        # 输入图像尺寸
BATCH_SIZE = 16       # 批次大小
LEARNING_RATE = 0.01  # 初始学习率

# 数据集
DATASET = "coco8.yaml"

# 模型配置
# 使用改进的 CBAM 模型配置文件
MODEL_CONFIG = "ultralytics/cfg/models/11/yolo11_cbam_55230316.yaml"
# 或者使用预训练权重（如果有）
# MODEL_WEIGHTS = "yolo11n.pt"

# ============================================================================
# 二、模型加载
# ============================================================================

print("=" * 60)
print(f"学生学号：{STUDENT_ID}")
print(f"实验名称：{EXPERIMENT_NAME}")
print("改进内容：在 YOLO11  backbone 和 head 中添加 CBAM 注意力机制")
print("=" * 60)

# 方案 A：从配置文件创建模型（从头训练）
print(f"\n从配置文件创建模型：{MODEL_CONFIG}")
model = YOLO(MODEL_CONFIG)

# 方案 B：加载预训练权重
# 如果需要使用预训练权重，取消下面注释
# print(f"加载预训练权重：yolo11n.pt")
# model = YOLO("yolo11n.pt")

# ============================================================================
# 三、模型训练
# ============================================================================

print(f"\n开始训练...")
print(f"数据集：{DATASET}")
print(f"设备：CUDA {DEVICE}")
print(f"轮次：{EPOCHS}, 图像尺寸：{IMG_SIZE}, 批次：{BATCH_SIZE}")
print("-" * 60)

# 开始训练
results = model.train(
    # 数据相关参数
    data=DATASET,

    # 训练超参数
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,

    # 优化器参数
    lr0=LEARNING_RATE,
    amp=False,  # 关闭自动混合精度

    # 设备参数
    device=DEVICE,

    # 预训练参数
    pretrained=True,  # 如果使用预训练权重则设为 True

    # 日志与保存
    project="runs/detect",
    name=EXPERIMENT_NAME,
    exist_ok=True,

    # 其他参数
    workers=4,
    verbose=True,
    save=True,
    save_period=10,  # 每 10 个 epoch 保存一次

    # 数据增强
    hsv_h=0.015,  # HSV-Hue 增强
    hsv_s=0.7,    # HSV-Saturation 增强
    hsv_v=0.4,    # HSV-Value 增强
    degrees=0.0,  # 旋转角度
    translate=0.1, # 平移
    scale=0.5,     # 缩放
    shear=0.0,     # 剪切
    perspective=0.0, # 透视变换
    flipud=0.0,    # 垂直翻转概率
    fliplr=0.5,    # 水平翻转概率
    mosaic=1.0,    # Mosaic 增强概率
    mixup=0.0,     # Mixup 增强概率
)

# ============================================================================
# 四、模型验证
# ============================================================================

print("\n" + "=" * 60)
print("训练完成！开始验证模型性能...")
print("=" * 60)

val_results = model.val(
    data=DATASET,
    device=DEVICE,
    splits="val"
)

# 打印关键指标
print(f"\n{'='*60}")
print(f"模型验证结果 - 学生学号：{STUDENT_ID}")
print(f"{'='*60}")
try:
    print(f"mAP50-95: {val_results.box.map50_95:.4f}")
    print(f"mAP50:    {val_results.box.map50:.4f}")
    print(f"Precision: {val_results.box.mp:.4f}")
    print(f"Recall:    {val_results.box.mr:.4f}")
except Exception:
    # 如果某些指标不可用，打印所有可用结果
    print(f"验证结果：{val_results}")
print(f"{'='*60}")

# ============================================================================
# 五、推理测试
# ============================================================================

print("\n执行推理测试...")

test_image = "https://ultralytics.com/images/bus.jpg"
det_results = model(test_image, device=DEVICE)

for result in det_results:
    boxes = result.boxes
    if boxes is not None:
        print(f"\n检测到 {len(boxes)} 个目标:")
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = result.names[cls_id]
            print(f"  [{i+1}] {class_name}: {conf:.2%}")

    # 保存结果
    output_path = result.save(f"inference_result_{STUDENT_ID}.jpg")
    print(f"\n推理结果已保存：{output_path[0] if isinstance(output_path, list) else output_path}")

# ============================================================================
# 六、模型导出
# ============================================================================

print("\n导出模型至 ONNX 格式...")

onnx_path = model.export(
    format="onnx",
    dynamic=False,
    simplify=True
)

print(f"ONNX 模型已保存至：{onnx_path}")

# ============================================================================
# 七、实验总结
# ============================================================================

print("\n" + "=" * 60)
print("训练流程完成！")
print("=" * 60)
print(f"学生学号：{STUDENT_ID}")
print(f"改进方法：CBAM 注意力机制")
print(f"改进位置：Backbone x3 + Head x2 = 共 5 个 CBAM 模块")
print(f"预期效果：提升模型对重要特征的关注，改善检测精度")
print("=" * 60)
