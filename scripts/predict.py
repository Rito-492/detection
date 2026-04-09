from ultralytics import YOLO

device = 'cuda:2'

# Load a pretrained YOLO model (recommended for training)
model = YOLO("runs/detect/train3/weights/yolo11n.pt")

# Perform object detection on an image using the model
results = model("YOLOO/bus.jpg", device=device)

for result in results:
    print(result.boxes)
    result.save()