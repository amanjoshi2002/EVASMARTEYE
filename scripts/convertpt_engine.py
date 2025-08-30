# from ultralytics import YOLO

# model = YOLO("yolo11x.pt")
# model.export(format="engine")  # This creates a TensorRT .engine files



from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11m.pt")

# Export the model to TensorRT format with 1920x1080 input size
model.export(format="engine", imgsz=(720, 1280))  # creates FP16 TensorRT engine for 1280x720

# Load the exported TensorRT model
tensorrt_model = YOLO("yolo11m.engine")

# Run inference
results = tensorrt_model("https://ultralytics.com/images/bus.jpg")