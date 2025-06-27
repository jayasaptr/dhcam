import cv2
import time
from app.config import RTSP_SETTINGS, GPU_SETTINGS

# Check if OpenCV has CUDA support
try:
    cv2_build_info = cv2.getBuildInformation()
    OPENCV_CUDA_AVAILABLE = 'CUDA' in cv2_build_info and 'YES' in cv2_build_info
    if OPENCV_CUDA_AVAILABLE and GPU_SETTINGS['use_cuda_opencv']:
        print("✅ OpenCV CUDA acceleration enabled")
    else:
        print("ℹ️ OpenCV CUDA not available or disabled")
except:
    OPENCV_CUDA_AVAILABLE = False
    print("ℹ️ Could not detect OpenCV CUDA support")

frame_global = None
last_frame_time = 0

def setup_rtsp_capture(rtsp_url):
    """Setup RTSP capture with optimized settings for minimal latency"""
    # Use CUDA backend if available
    if OPENCV_CUDA_AVAILABLE and GPU_SETTINGS['use_cuda_opencv']:
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        # Try to use GPU backend
        cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
    else:
        cap = cv2.VideoCapture()
    
    # Ultra low latency settings - set before opening
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, RTSP_SETTINGS['timeout_ms'])
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000)  # Reduce read timeout
    
    # Force TCP for more reliable connection
    rtsp_url_tcp = rtsp_url + "?tcp"
    
    # Try to open with retries
    for attempt in range(RTSP_SETTINGS['max_retries']):
        print(f"🔄 Attempting RTSP connection (attempt {attempt + 1}/{RTSP_SETTINGS['max_retries']})...")
        
        if cap.open(rtsp_url_tcp):
            print("✅ RTSP connection successful!")
            
            # Ultra low latency settings after connection
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Disable frame dropping and use latest frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Get and display stream properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            print(f"📹 Stream properties: {width}x{height} @ {fps} FPS")
            print("⚡ Low latency mode enabled")
            
            return cap
        else:
            print(f"❌ RTSP connection failed (attempt {attempt + 1})")
            if attempt < RTSP_SETTINGS['max_retries'] - 1:
                time.sleep(RTSP_SETTINGS['retry_delay'])
    
    print("🛑 Failed to connect to RTSP stream after all retries")
    return None

frame_global = None
last_frame_time = 0

def set_frame(frame):
    global frame_global, last_frame_time
    frame_global = frame
    last_frame_time = time.time()

def generate_frames():
    global frame_global, last_frame_time
    last_yield_time = 0
    min_interval = 1/30  # Max 30 FPS for web streaming
    
    while True:
        if frame_global is None:
            time.sleep(0.01)  # Small sleep to prevent CPU spinning
            continue
            
        current_time = time.time()
        
        # Throttle frame rate for web streaming to reduce bandwidth and delay
        if current_time - last_yield_time < min_interval:
            time.sleep(0.01)
            continue
            
        try:
            # Use lower quality to reduce encoding time and bandwidth
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 70]  # Reduced quality for speed
            _, buffer = cv2.imencode('.jpg', frame_global, encode_params)
            frame = buffer.tobytes()
            last_yield_time = current_time
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"⚠️ Frame encoding error: {e}")
            time.sleep(0.01)
