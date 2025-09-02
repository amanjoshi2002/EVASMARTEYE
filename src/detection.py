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

def ffmpeg_frame_reader(rtsp_url, width=1280, height=736, fps=1, cam_name="unknown"):
    log_dir = f"logs/{cam_name}"
    os.makedirs(log_dir, exist_ok=True)
    ffmpeg_log_path = os.path.join(log_dir, "ffmpeg_stderr.log")

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
        .global_args('-fflags', '+discardcorrupt+nobuffer')
        .global_args('-flags', '+low_delay')
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )
    frame_size = width * height * 3

    # Thread to log FFmpeg stderr in real time
    def log_ffmpeg_stderr(stderr, log_path):
        with open(log_path, "ab") as log_file:
            while True:
                line = stderr.readline()
                if not line:
                    break
                log_file.write(line)
                log_file.flush()

    stderr_thread = threading.Thread(target=log_ffmpeg_stderr, args=(process.stderr, ffmpeg_log_path), daemon=True)
    stderr_thread.start()

    try:
        while True:
            in_bytes = process.stdout.read(frame_size)
            if not in_bytes or len(in_bytes) < frame_size:
                break
            frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
            yield frame
    finally:
        process.stdout.close()
        process.stderr.close()
        process.wait()


def producer(rtsp_url, frame_queue, stats, cam_name, lock):
    while True:
        try:
            for frame in ffmpeg_frame_reader(rtsp_url, cam_name=cam_name):
                if not frame_queue.full():
                    frame_queue.put(frame)
                    with lock:
                        stats[cam_name]["frames_received"] += 1
            print(f"[{cam_name}] FFmpeg stopped producing frames, restarting in 5 seconds...")
            time.sleep(5)
        except Exception as e:
            print(f"[{cam_name}] Producer error: {e}, restarting in 5 seconds...")
            time.sleep(5)

def consumer(frame_queue, cam_name, stats, lock):
    """
    Consumer function that processes frames with duplicate detection.
    """
    global model
    frame_count = 0
    duplicate_count = 0
    total_frames = 0
    total_processing_time = 0.0  # Track total processing time for averaging
    start_time = time.time()

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
            frame_sample = frame[::20, ::20].tobytes()
            frame_hash = hash(frame_sample)

            if frame_hash == last_frame_hash:
                duplicate_count += 1
                frame_queue.task_done()
                print(f"[{cam_name}] Skipped duplicate hash frame (Total duplicates: {duplicate_count})")
                continue

            # Update tracking variables
            last_frame_hash = frame_hash
            last_timestamp = current_timestamp

            # Measure processing time
            process_start = time.time()
            results = model(frame, imgsz=(736, 1280))
            process_end = time.time()
            processing_time = process_end - process_start
            total_processing_time += processing_time

            frame_count += 1

            print(f"[{cam_name}] Processed frame {frame_count} at {current_timestamp:.2f} (Skipped {duplicate_count} duplicates, Processing time: {processing_time:.3f}s)")

            frame_queue.task_done()

            # Reporting every minute (removed log writing)
            current_time = time.time()
            if current_time - start_time >= 60:
                with lock:
                    frames_received = stats[cam_name]["frames_received"]
                    stats[cam_name]["frames_received"] = 0  # reset for next minute

                avg_processing_time = (
                    round(total_processing_time / frame_count, 4) if frame_count > 0 else 0.0
                )

                report = {
                    "camera": cam_name,
                    "frames_received_last_minute": frames_received,
                    "frames_processed_last_minute": frame_count,
                    "duplicates_skipped_last_minute": duplicate_count,
                    "total_frames_received_by_consumer": total_frames,
                    "duplicate_rate_percent": round((duplicate_count / max(total_frames, 1)) * 100, 2),
                    "avg_processing_time_sec": avg_processing_time,
                    "timestamp": int(current_time)
                }

                print(f"[{cam_name}] Report: {report}")

                frame_count = 0
                duplicate_count = 0
                total_processing_time = 0.0
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
    model = YOLO("best.engine")  # Load once globally

    stats = {}
    lock = threading.Lock()
    for cam in cameras:
        stats[cam.name] = {"frames_received": 0}
        rtsp_url = f"rtsp://admin:private123@{cam.ip}:554/cam/realmonitor?channel=1&subtype=0"
        frame_queue = queue.Queue(maxsize=10)
        threading.Thread(target=producer, args=(rtsp_url, frame_queue, stats, cam.name, lock), daemon=True).start()
        threading.Thread(target=consumer, args=(frame_queue, cam.name, stats, lock), daemon=True).start()