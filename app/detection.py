import cv2, time
from ultralytics import YOLO
from datetime import datetime, timedelta
from app.camera import set_frame, setup_rtsp_capture
from app.utils import save_and_post_image
from app.config import MODEL_PATH, IOU_THRESHOLD, POST_DELAY_MINUTES, RTSP_SETTINGS

CONFIDENCE_THRESHOLD = 0.25 
MARGIN_TOLERANCE = 10       

model = YOLO(MODEL_PATH)
cocoClassNames = ['boots', 'glasses', 'gloves', 'helmet', 'no-boots', 'no-gloves', 'no-helmet', 'no-vest', 'person', 'vest']

image_count = 0
person_last_posted = {}

CLASS_COLORS = {
    'glasses': (255, 255, 0),     # cyan
    'helmet': (0, 255, 0),        # green
    'no-gloves': (255, 0, 255),   # magenta
    'no-helmet': (0, 0, 255),     # red
    'person': (255, 255, 0),      # yellow
}

def run_detection(cap, video_writer, rtsp_url=None):
    global image_count
    ptime = 0
    reconnect_attempts = 0
    max_reconnect_attempts = 10
    frame_skip_count = 0
    max_frame_skip = 2
    detection_skip = 0
    detection_interval = 2
    last_no_helmet_boxes = []

    while True:
        for _ in range(max_frame_skip + 1):
            ret, frame = cap.read()
            if not ret:
                break

        if not ret:
            print("⚠️ Failed to read frame from stream")
            if rtsp_url and reconnect_attempts < max_reconnect_attempts:
                print(f"🔄 Attempting to reconnect to RTSP stream... (attempt {reconnect_attempts + 1})")
                cap.release()
                time.sleep(RTSP_SETTINGS['retry_delay'])
                new_cap = setup_rtsp_capture(rtsp_url)
                if new_cap is not None:
                    cap = new_cap
                    reconnect_attempts = 0
                    print("✅ RTSP reconnection successful!")
                    continue
                else:
                    reconnect_attempts += 1
                    continue
            else:
                print("🛑 Max reconnection attempts reached or no RTSP URL provided")
                break

        reconnect_attempts = 0
        frame_skip_count += 1
        if frame_skip_count % 30 == 0:
            for _ in range(3):
                cap.read()

        detection_skip += 1
        process_detection = detection_skip % detection_interval == 0
        now = datetime.now()
        no_helmet_boxes = []

        if process_detection:
            results = model.track(frame, persist=True, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)

            if results and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else []

                for i, box in enumerate(boxes):
                    class_id = int(box.cls[0])
                    class_name = cocoClassNames[class_id]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    track_id = ids[i] if i < len(ids) else None

                    # Simpan bounding box untuk no-helmet untuk pengiriman
                    if class_name == "no-helmet":
                        no_helmet_boxes.append((x1, y1, x2, y2, track_id))

                    # Tampilkan semua class yang ada di CLASS_COLORS
                    if class_name in CLASS_COLORS:
                        label = f"{class_name} (ID:{track_id})" if track_id is not None else class_name
                        color = CLASS_COLORS[class_name]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            no_helmet_boxes = last_no_helmet_boxes

        last_no_helmet_boxes = no_helmet_boxes

        # Logic simpan gambar full-frame jika no-helmet terdeteksi
        for (x1, y1, x2, y2, track_id) in no_helmet_boxes:
            if track_id is None:
                continue

            key = f"no_helmet_{track_id}"
            if (
                key not in person_last_posted or
                now - person_last_posted[key]["time"] >= timedelta(minutes=POST_DELAY_MINUTES)
            ):
                success, _ = save_and_post_image(frame, None, None, None, None, key, "helmet")
                if success:
                    person_last_posted[key] = {
                        "time": now,
                        "item": "helmet"
                    }
                    image_count += 1
                    print(f"📸 Gambar full disimpan untuk ID {track_id}")
            else:
                print(f"⏳ Skip post: {key} masih dalam 5 menit")

        ctime = time.time()
        fps_calc = 1 / (ctime - ptime) if ptime else 0
        ptime = ctime
        cv2.putText(frame, f"FPS: {int(fps_calc)}", (30, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        video_writer.write(frame)
        set_frame(frame.copy())
