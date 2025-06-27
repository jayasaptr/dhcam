import threading
import cv2
from flask import Flask, Response
from flask_cors import CORS
from app.camera import generate_frames, setup_rtsp_capture
from app.detection import run_detection
import os
from urllib.parse import quote

app = Flask(__name__)
CORS(app)

username = "admin"
password = "Network2011#"
encoded_password = quote(password)

# RTSP URL configuration
# rtsp_url = f"rtsp://{username}:{encoded_password}@10.24.240.67:554/Streaming/Channels/101"
# rtsp_url = "rtsp://admin:Network2011*@10.24.240.171:554/Streaming/Channels/101"
rtsp_url = 'rtsp://admin:Network2011*@10.24.240.191:554/Streaming/Channels/101'

# Setup RTSP capture with optimized settings
cap = setup_rtsp_capture(rtsp_url)
if cap is None:
    print("🛑 Failed to initialize RTSP capture. Exiting...")
    exit(1)

# Clear initial buffer to reduce startup delay
print("🔄 Clearing initial buffer...")
for _ in range(10):
    cap.read()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25  # Default to 25 FPS if unable to get from stream
print(f"📹 Stream info: {frame_width}x{frame_height} @ {fps} FPS")

output_video = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_flask():
    print("✅ Flask running on http://localhost:5000/video_feed")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

if __name__ == "__main__":
    try:
        flask_thread = threading.Thread(target=run_flask)
        flask_thread.daemon = True
        flask_thread.start()
        run_detection(cap, output_video, rtsp_url)
    except KeyboardInterrupt:
        print("\n🛑 Dihentikan oleh user.")
    finally:
        cap.release()
        output_video.release()
        cv2.destroyAllWindows()
