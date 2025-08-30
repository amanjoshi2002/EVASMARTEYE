import urllib.request
import cv2

video_url = "https://filesamples.com/samples/video/mp4/sample_1920x1080.mp4"
local_path = "sample_1920x1080.mp4"
urllib.request.urlretrieve(video_url, local_path)

cap = cv2.VideoCapture(local_path)