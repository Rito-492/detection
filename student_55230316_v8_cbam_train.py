# YOLOv8-CBAM 改进版训练脚本
# 学生学号：55230316

from ultralytics import YOLO

STUDENT_ID = "55230316"
EXPERIMENT_NAME = f"yolov8_cbam_{STUDENT_ID}"

EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 16
DEVICE = 0
DATASET = "coco8.yaml"
# MODEL_CONFIG = "ultralytics/cfg/models/8/yolov8_cbam_55230316.yaml"
MODEL_CONFIG = "checkpoints\yolov8n.pt"

print("=" * 60)
print(f"学生学号：{STUDENT_ID}")
print(f"实验名称：{EXPERIMENT_NAME}")
print("改进内容：在 YOLOv8 backbone 和 head 中添加 CBAM 注意力机制")
print("=" * 60)

print(f"\n从配置文件创建模型：{MODEL_CONFIG}")
model = YOLO(MODEL_CONFIG)

print(f"\n开始训练...")
print(f"数据集：{DATASET}")
print(f"设备：CUDA {DEVICE}")

results = model.train(
    data=DATASET,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    device=DEVICE,
    pretrained=True,
    # project="runs/detect",
    name=EXPERIMENT_NAME,
    exist_ok=True,
    verbose=True,
    save=True,
    save_period=10,
)

print("\n" + "=" * 60)
print("训练完成！开始验证模型性能...")
print("=" * 60)

val_results = model.val(
    data=DATASET,
    device=DEVICE,
    split="val"
)

print(f"\n{'='*60}")
print(f"模型验证结果 - 学生学号：{STUDENT_ID}")
print(f"{'='*60}")
try:
    print(f"mAP50-95: {val_results.box.map50_95:.4f}")
    print(f"mAP50:    {val_results.box.map50:.4f}")
    print(f"Precision: {val_results.box.mp:.4f}")
    print(f"Recall:    {val_results.box.mr:.4f}")
except Exception:
    print(f"验证结果：{val_results}")
print(f"{'='*60}")

print("\n执行推理测试...")
test_image = "bus.jpg"
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

    output_path = result.save(f"inference_result_v8_{STUDENT_ID}.jpg")
    print(f"\n推理结果已保存：{output_path[0] if isinstance(output_path, list) else output_path}")

print("\n" + "=" * 60)
print("训练流程完成！")
print("=" * 60)
print(f"学生学号：{STUDENT_ID}")
print(f"改进方法：CBAM 注意力机制")
print(f"改进位置：Backbone x3 + Head x2 = 共 5 个 CBAM 模块")
print("=" * 60)
