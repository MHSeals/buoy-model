from ultralytics import YOLO

# Load your trained model (e.g., 'yolov8n.pt' or 'best.pt')
model = YOLO("runs/detect/v20/weights/best.pt")

# Export to ONNX format
model.export(format="onnx")