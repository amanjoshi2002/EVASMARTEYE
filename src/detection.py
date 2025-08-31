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
        .global_args('-fflags', '+discardcorrupt+nobuffer')  # Add this
        .global_args('-flags', '+low_delay')                 # Add this
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )
    frame_size = width * height * 3
    while True:
        in_bytes = process.stdout.read(frame_size)
        if not in_bytes or len(in_bytes) < frame_size:
            break
        frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
        yield frame
        # Remove this line: time.sleep(1)  # Let FFmpeg handle timing


def producer(rtsp_url, frame_queue):
    for frame in ffmpeg_frame_reader(rtsp_url):
        if not frame_queue.full():
            frame_queue.put(frame)

def consumer(frame_queue, cam_name):
    """
    Consumer function that processes frames with duplicate detection.
    """
    global model
    frame_count = 0
    duplicate_count = 0
    total_frames = 0
    start_time = time.time()
    save_dir = f"processed_images/{cam_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    last_frame_hash = None
    last_timestamp = 0
    min_time_diff = 0.8  # Minimum 0.8 seconds between frames
    
    print(f"[{cam_name}] Consumer started")
    
    while True:
        try:
            frame = frame_queue.get(timeout=30)
            if frame is None:
                break
            
            current_timestamp = time.time()
            total_frames += 1
            
            # Time-based filtering FIRST
            if current_timestamp - last_timestamp < min_time_diff:
                duplicate_count += 1
                frame_queue.task_done()
                print(f"[{cam_name}] Skipped frame due to time constraint (Total duplicates: {duplicate_count})")
                continue
            
            # Hash-based duplicate detection SECOND
            frame_sample = frame[::20, ::20].tobytes()  # Even more aggressive sampling
            frame_hash = hash(frame_sample)
            
            if frame_hash == last_frame_hash:
                duplicate_count += 1
                frame_queue.task_done()
                print(f"[{cam_name}] Skipped duplicate hash frame (Total duplicates: {duplicate_count})")
                continue
            
            # Update tracking variables
            last_frame_hash = frame_hash
            last_timestamp = current_timestamp
            
            # Process the frame
            results = model(frame, imgsz=(736, 1280))
            frame_count += 1
            
            print(f"[{cam_name}] Processed frame {frame_count} at {current_timestamp:.2f} (Skipped {duplicate_count} duplicates)")
            
            # Draw labels and save
            labeled_img = results[0].plot()
            timestamp = int(time.time())
            img_path = os.path.join(save_dir, f"{timestamp}_{frame_count}.jpg")
            cv2.imwrite(img_path, cv2.cvtColor(labeled_img, cv2.COLOR_RGB2BGR))
            
            frame_queue.task_done()
            
            # Rest of your reporting code...
            current_time = time.time()
            if current_time - start_time >= 60:
                report = {
                    "camera": cam_name,
                    "frames_processed_last_minute": frame_count,
                    "duplicates_skipped_last_minute": duplicate_count,
                    "total_frames_received": total_frames,
                    "duplicate_rate_percent": round((duplicate_count / max(total_frames, 1)) * 100, 2),
                    "timestamp": int(current_time)
                }
                
                log_entry = json.dumps(report) + "\n"
                with open("frame_log.json", "a") as f:
                    f.write(log_entry)
                
                print(f"[{cam_name}] Report: {json.dumps(report)}")
                
                frame_count = 0
                duplicate_count = 0
                total_frames = 0
                start_time = current_time
                
        except queue.Empty:
            print(f"[{cam_name}] No frames received for 30 seconds...")
            continue
        except Exception as e:
            print(f"[{cam_name}] Consumer error: {e}")
            break
    
    print(f"[{cam_name}] Consumer stopped")

def run_detection(cameras):
    global model
    model = YOLO("yolo11m16.engine")  # Load once globally

    for cam in cameras:
        rtsp_url = f"rtsp://admin:private123@{cam.ip}:554/cam/realmonitor?channel=1&subtype=0"
        frame_queue = queue.Queue(maxsize=10)
        threading.Thread(target=producer, args=(rtsp_url, frame_queue), daemon=True).start()
        threading.Thread(target=consumer, args=(frame_queue, cam.name), daemon=True).start()