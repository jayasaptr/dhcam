import cv2, time
from ultralytics import YOLO
from datetime import datetime, timedelta
from app.camera import set_frame, setup_rtsp_capture
from app.utils import save_and_post_image
from app.config import MODEL_PATH, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, POST_DELAY_MINUTES, RTSP_SETTINGS

# Inisialisasi model dan class label
model = YOLO(MODEL_PATH)
cocoClassNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person',
                  'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

image_count = 0
person_last_posted = {}

def run_detection(cap, video_writer, rtsp_url=None):
    global image_count
    ptime = 0
    reconnect_attempts = 0
    max_reconnect_attempts = 10
    frame_skip_count = 0
    max_frame_skip = 2  # Skip frames to reduce latency
    detection_skip = 0
    detection_interval = 2  # Process every 2nd frame for detection to reduce CPU load
    last_person_boxes = []
    last_no_helmet_boxes = []

    while True:
        # Read multiple frames to get the latest one (reduce buffering delay)
        for _ in range(max_frame_skip + 1):
            ret, frame = cap.read()
            if not ret:
                break
        
        # Handle RTSP disconnection
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

        # Reset reconnect attempts on successful frame read
        reconnect_attempts = 0
        
        # Clear buffer periodically to prevent accumulation
        frame_skip_count += 1
        if frame_skip_count % 30 == 0:  # Every 30 frames
            # Flush buffer by reading additional frames
            for _ in range(3):
                cap.read()

        # Skip detection processing for some frames to reduce delay
        detection_skip += 1
        process_detection = detection_skip % detection_interval == 0

        if process_detection:
            results = model.track(frame, persist=True, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)

            person_boxes = []
            no_helmet_boxes = []

            if results and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else []

                for i, box in enumerate(boxes):
                    class_id = int(box.cls[0])
                    class_name = cocoClassNames[class_id]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    track_id = ids[i] if i < len(ids) else None

                    if class_name == "Person":
                        person_boxes.append((x1, y1, x2, y2, track_id))
                    elif class_name == "NO-Hardhat":
                        no_helmet_boxes.append((x1, y1, x2, y2))

            # Store results for next frame if detection is skipped
            last_person_boxes = person_boxes
            last_no_helmet_boxes = no_helmet_boxes
        else:
            # Use previous detection results to maintain smooth visualization
            person_boxes = last_person_boxes
            no_helmet_boxes = last_no_helmet_boxes

        now = datetime.now()

        for (x1p, y1p, x2p, y2p, track_id) in person_boxes:
            if track_id is None:
                continue

            center_x = (x1p + x2p) // 2
            center_y = (y1p + y2p) // 2
            has_no_helmet = any(
                x1h >= x1p and x2h <= x2p and y1h >= y1p and y2h <= y2p
                for (x1h, y1h, x2h, y2h) in no_helmet_boxes
            )

            key = f"person_{track_id}"

            if has_no_helmet and process_detection:  # Only save images when actually processing detection
                # Cek apakah sudah dikirim dalam 5 menit dengan pelanggaran yang sama
                if (
                    key not in person_last_posted or
                    person_last_posted[key].get("item") != "helmet" or
                    now - person_last_posted[key]["time"] >= timedelta(minutes=POST_DELAY_MINUTES)
                ):
                    success, _ = save_and_post_image(frame, x1p, y1p, x2p, y2p, key, "helmet")
                    if success:
                        person_last_posted[key] = {
                            "center": (center_x, center_y),
                            "time": now,
                            "item": "helmet"
                        }
                        image_count += 1
                else:
                    print(f"⏳ Skip post: {key} masih dalam 5 menit & masih pelanggaran helmet.")
            elif not has_no_helmet and process_detection:
                # Reset data jika orang tersebut sudah memakai helm
                if key in person_last_posted:
                    print(f"🧼 {key} sekarang pakai helm — reset tracking.")
                    del person_last_posted[key]

            label = f"ID:{track_id} - No Hardhat" if has_no_helmet else f"ID:{track_id} - Person"
            color = (0, 0, 255) if has_no_helmet else (255, 255, 0)
            cv2.rectangle(frame, (x1p, y1p), (x2p, y2p), color, 2)
            cv2.putText(frame, label, (x1p, y1p - 5), 0, 0.5, color, 2)

        # Gambar box untuk NO-Hardhat
        for (x1h, y1h, x2h, y2h) in no_helmet_boxes:
            cv2.rectangle(frame, (x1h, y1h), (x2h, y2h), (0, 0, 255), 2)
            cv2.putText(frame, "NO-Hardhat", (x1h, y1h - 5), 0, 0.5, (0, 0, 255), 2)

        # Tampilkan FPS
        ctime = time.time()
        fps_calc = 1 / (ctime - ptime) if ptime else 0
        ptime = ctime
        cv2.putText(frame, f"FPS: {int(fps_calc)}", (30, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        video_writer.write(frame)
        set_frame(frame.copy())
