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
    'retry_delay': 2,
    'timeout_ms': 5000,
    'buffer_size': 10,
    'low_latency': True
}

# GPU Configuration
GPU_SETTINGS = {
    'enable_gpu': True,
    'device': 'cuda:0',  # Use first CUDA device
    'use_cuda_opencv': True,
    'gpu_memory_fraction': 0.8,  # Use 80% of GPU memory
    'mixed_precision': True,  # Enable mixed precision for faster inference
    'batch_size': 1,  # Batch size for inference
    'half_precision': True,  # Use FP16 for faster inference
}
