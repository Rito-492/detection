from ultralytics import YOLO

# Load a pretrained YOLO model (recommended for training)
model = YOLO("runs/detect/train3/weights/best.pt")

# Perform object detection on an image using the model
results = model("/home/tyr/datasets/coco8/images/val/000000000036.jpg", conf=0.001)

for result in results:
    print(result.boxes)
    result.save()