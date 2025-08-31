from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11m.pt")

# Export the model to TensorRT format with batch size 30 and 1280x720 input size
model.export(format="engine", imgsz=(720, 1280), batch=30, half=True)  # creates TensorRT engine for batch size 30

# Load the exported TensorRT model
tensorrt_model = YOLO("yolo11m.engine")

# Run inference
results = tensorrt_model("https://ultralytics.com/images/bus.jpg")