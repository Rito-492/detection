# YOLOv8-CBAM 目标检测推理脚本
# 作者：汤雨润
# 学生学号：55230316

from ultralytics import YOLO

DEVICE = 7

MODEL_PATH = "runs/detect/yolov8_cbam_55230316/weights/best.pt"

print(f"\n加载模型：{MODEL_PATH}")
model = YOLO(MODEL_PATH)
print(f"模型加载完成")

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 300
IMAGE_SIZE = 640

TEST_IMAGES = [
    "test_pics/bus.jpg",
    "test_pics/zidane.jpg",
]


def run_inference(image_path):
    """对单张图像执行目标检测"""
    print(f"\n推理图像：{image_path}")

    results = model(
        source=image_path,
        device=DEVICE,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        max_det=MAX_DETECTIONS,
        imgsz=IMAGE_SIZE,
        verbose=False
    )

    return results


def process_results(results):
    """处理并打印检测结果"""
    import os
    os.makedirs("inference_results", exist_ok=True)

    for result in results:
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            print("  未检测到任何目标")
            return

        print(f"  检测到 {len(boxes)} 个目标:")
        print(f"  {'类别':<15} {'置信度':<10} {'边界框坐标 (xyxy)':<30}")
        print(f"  {'-'*65}")

        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_name = result.names[cls_id]
            print(f"  [{i+1:2d}] {class_name:<14} {conf:>6.2%}    [{x1:5.1f}, {y1:5.1f}, {x2:5.1f}, {y2:5.1f}]")

        output_path = result.save(f"inference_results/result_{i}.jpg")[0]
        print(f"\n  结果已保存：{output_path}")


def batch_inference(image_list):
    """对多张图像执行批量推理"""
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
    print(f"{'='*60}")


if __name__ == "__main__":
    batch_inference(TEST_IMAGES)
