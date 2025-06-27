import os

# Path model YOLO
MODEL_PATH = "best.pt"

# Folder output
OUTPUT_FOLDER = os.path.join("static", "no_hardhat_detected")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Post API target
POST_URL = "http://127.0.0.1:8000/api/offenses"

# Deteksi
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.3

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
