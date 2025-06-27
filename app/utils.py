import cv2
import io
import requests
from datetime import datetime, timedelta
from app.config import POST_URL

# Dictionary untuk menyimpan file terakhir dan waktu post-nya
last_posted_images = {}

def save_and_post_image(frame, x1, y1, x2, y2, image_key, item):
    now = datetime.now()
    cropped = frame[y1:y2, x1:x2]

    # Cek apakah gambar ini sudah di-post dalam 5 menit terakhir
    if image_key in last_posted_images:
        last_time = last_posted_images[image_key]
        if now - last_time < timedelta(minutes=5):
            print(f"⏳ Lewatkan post {image_key}, masih dalam 5 menit terakhir.")
            return False, None

    # Encode gambar ke memory (tanpa simpan ke disk)
    success, buffer = cv2.imencode('.jpg', cropped)
    if not success:
        print("❌ Gagal encode gambar.")
        return False, None

    img_bytes = io.BytesIO(buffer)

    try:
        files = {'picture': ('image.jpg', img_bytes, 'image/jpeg')}
        data = {
            "area": "Entery Gate",
            "item": item if item else "Unknown",
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M")
        }
        response = requests.post(POST_URL, data=data, files=files)
        if response.status_code in [200, 201]:
            last_posted_images[image_key] = now  # Simpan waktu post terakhir
            print(f"✅ POST berhasil untuk {image_key}")
            return True, None
        else:
            print(f"⚠️ Gagal POST: {response.status_code} - {response.text}")
            return False, None
    except Exception as e:
        print(f"❌ Error post: {e}")
        return False, None
