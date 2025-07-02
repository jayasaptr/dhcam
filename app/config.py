import os

# Path model YOLO
MODEL_PATH = "best.pt"

# Folder output
OUTPUT_FOLDER = os.path.join("static", "no_helmet_detected")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Post API target
POST_URL = "http://10.24.240.141:8002/api/offenses"

# Deteksi
CONFIDENCE_THRESHOLD = 0.2
IOU_THRESHOLD = 0.2

# Delay POST per orang
POST_DELAY_MINUTES = 5

# RTSP Configuration
RTSP_SETTINGS = {
    'enable_hevc': True,
    'use_ffmpeg_cmd': False,
    'max_retries': 5,
    'retry_delay': 1,  # Reduced delay
    'timeout_ms': 3000,  # Reduced timeout
    'buffer_size': 1,  # Minimal buffer
    'low_latency': True
}
