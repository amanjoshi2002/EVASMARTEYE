import ffmpeg
import numpy as np
import queue
import threading
import time  # <-- Add this import
import json  # <-- Add this import
import os
import cv2
from ultralytics import YOLO

model = None  # Global model

def ffmpeg_frame_reader(rtsp_url, width=1280, height=736, fps=1):
    """
    Generator that yields frames from an RTSP stream using ffmpeg at the desired FPS and resolution.
    """
    process = (
        ffmpeg
        .input(rtsp_url, rtsp_transport='tcp')
        .output(
            'pipe:',
            format='rawvideo',
            pix_fmt='rgb24',
            vf=f'fps={fps},scale={width}:{height}'
        )
        .global_args('-hwaccel', 'cuda')
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )
    frame_size = width * height * 3
    while True:
        in_bytes = process.stdout.read(frame_size)
        if not in_bytes or len(in_bytes) < frame_size:
            break
        frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
        yield frame
        time.sleep(1)  # 1 FPS

def producer(rtsp_url, frame_queue):
    for frame in ffmpeg_frame_reader(rtsp_url):
        if not frame_queue.full():
            frame_queue.put(frame)

def consumer(frame_queue, cam_name):
    global model
    frame_count = 0
    start_time = time.time()
    save_dir = f"processed_images/{cam_name}"
    os.makedirs(save_dir, exist_ok=True)
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        results = model(frame, imgsz=(736, 1280))
        frame_count += 1
        print(f"[{cam_name}] Inference done")
        # Draw labels on the frame
        labeled_img = results[0].plot()  # This returns an image with labels drawn
        # Save the labeled image
        img_path = os.path.join(save_dir, f"{int(time.time())}_{frame_count}.jpg")
        cv2.imwrite(img_path, cv2.cvtColor(labeled_img, cv2.COLOR_RGB2BGR))
        frame_queue.task_done()
        # Report every minute in JSON and store in file
        if time.time() - start_time >= 60:
            report = {
                "camera": cam_name,
                "frames_processed_last_minute": frame_count,
                "timestamp": int(time.time())
            }
            with open("frame_log.json", "a") as f:
                f.write(json.dumps(report) + "\n")
            print(json.dumps(report))
            frame_count = 0
            start_time = time.time()

def run_detection(cameras):
    global model
    model = YOLO("yolo11m16.engine")  # Load once globally

    for cam in cameras:
        rtsp_url = f"rtsp://admin:private123@{cam.ip}:554/cam/realmonitor?channel=1&subtype=0"
        frame_queue = queue.Queue(maxsize=10)
        threading.Thread(target=producer, args=(rtsp_url, frame_queue), daemon=True).start()
        threading.Thread(target=consumer, args=(frame_queue, cam.name), daemon=True).start()