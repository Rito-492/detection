# ============================================================================
# YOLO11 目标检测推理脚本
# 学生学号：55230316
# 任务：个人任务一 - 算力环境部署与核心算法跑通
# 方向：计算机视觉 - 基于 YOLO11 的目标检测
# ============================================================================

from ultralytics import YOLO
import torch

# ============================================================================
# 一、设备配置
# ============================================================================

def get_device():
    """
    自动选择最佳计算设备
    优先使用 GPU，如果没有则回退到 CPU
    """
    if torch.cuda.is_available():
        device = 'cuda:0'  # 使用第一块 GPU
        print(f"✓ 使用 GPU 加速：{torch.cuda.get_device_name(0)}")
        print(f"  显存容量：{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        device = 'cpu'
        print("⚠ 未检测到 GPU，使用 CPU 推理（速度较慢）")
    return device

DEVICE = get_device()

# ============================================================================
# 二、模型加载
# ============================================================================

# 模型路径配置
# 可以使用以下任一来源的模型：
# 1. 预训练模型：yolo11n.pt, yolo11s.pt 等
# 2. 自定义训练：runs/detect/train_55230316/weights/best.pt
# MODEL_PATH = "checkpoints/yolov8n.pt"  # 使用预训练的 nano 版本
MODEL_PATH = "runs/detect/yolov8_cbam_55230316/weights/best.pt"

print(f"\n加载模型：{MODEL_PATH}")
model = YOLO(MODEL_PATH)
print(f"✓ 模型加载完成")

# ============================================================================
# 三、推理配置
# ============================================================================

# 推理参数
CONF_THRESHOLD = 0.25     # 置信度阈值：只保留置信度>0.25 的检测结果
IOU_THRESHOLD = 0.45      # NMS IoU 阈值：用于去除重复检测框
MAX_DETECTIONS = 300      # 最大检测框数量
IMAGE_SIZE = 640          # 推理时图像尺寸

# 测试图像列表
# 可以是本地路径或 URL
TEST_IMAGES = [
    "/home/tyr/datasets/coco8/images/train/000000000034.jpg",
    "test_pics/bus.jpg",
    "test_pics/zidane.jpg",
    # 添加更多本地图像路径...
    # "images/test1.jpg",
    # "images/test2.jpg",
]

# ============================================================================
# 四、执行推理
# ============================================================================

def run_inference(image_path):
    """
    对单张图像执行目标检测

    Args:
        image_path: 图像路径或 URL

    Returns:
        results: 检测结果对象
    """
    print(f"\n推理图像：{image_path}")

    # 执行推理
    results = model(
        source=image_path,
        device=DEVICE,              # 计算设备
        conf=CONF_THRESHOLD,        # 置信度阈值
        iou=IOU_THRESHOLD,          # NMS IoU 阈值
        max_det=MAX_DETECTIONS,     # 最大检测数
        imgsz=IMAGE_SIZE,           # 输入尺寸
        verbose=False               # 静默模式
    )

    return results

def process_results(results):
    """
    处理并打印检测结果

    Args:
        results: YOLO 推理结果对象
    """
    import os
    os.makedirs("inference_results_55230316", exist_ok=True)

    for result in results:
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            print("  未检测到任何目标")
            return

        print(f"  检测到 {len(boxes)} 个目标:")
        print(f"  {'类别':<15} {'置信度':<10} {'边界框坐标 (xyxy)':<30}")
        print(f"  {'-'*65}")

        # 遍历所有检测框
        for i, box in enumerate(boxes):
            # 获取类别 ID 和置信度
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # 获取边界框坐标 [x1, y1, x2, y2]
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # 获取类别名称
            class_name = result.names[cls_id]

            # 打印检测结果
            print(f"  [{i+1:2d}] {class_name:<14} {conf:>6.2%}    [{x1:5.1f}, {y1:5.1f}, {x2:5.1f}, {y2:5.1f}]")

        # 保存结果图像
        output_path = result.save(f"inference_results_55230316/result_{i}.jpg")[0]
        print(f"\n  结果已保存：{output_path}")

# ============================================================================
# 五、批量推理
# ============================================================================

def batch_inference(image_list):
    """
    对多张图像执行批量推理

    Args:
        image_list: 图像路径列表
    """
    print("\n" + "="*60)
    print("批量推理开始")
    print("="*60)

    for i, img_path in enumerate(image_list, 1):
        print(f"\n[{'*'*50}]")
        print(f"图像 {i}/{len(image_list)}")
        print(f"{'*'*50}")

        results = run_inference(img_path)
        process_results(results)

    print(f"\n{'='*60}")
    print("批量推理完成！")
    print(f"学生学号：55230316")
    print(f"{'='*60}")

# ============================================================================
# 六、主函数
# ============================================================================

if __name__ == "__main__":
    # 执行批量推理
    batch_inference(TEST_IMAGES)

    # 如果要推理单张图像，可以这样做：
    # results = run_inference("your_image.jpg")
    # process_results(results)
