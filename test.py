import cv2
import time
import firebase_admin
import numpy as np
import cloudinary.uploader
import requests
from inference import InferencePipeline
from firebase_admin import credentials, firestore

# Initialize Firebase
cred = credentials.Certificate("finalproject-f1a6e-firebase-adminsdk-rb9vv-5f590cfe25.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Cloudinary configuration
cloudinary.config(
    cloud_name="deigyww0d",
    api_key="841958412982462",
    api_secret="BbCCAm4-Ry6EMj-g1bWVKlLJDFQ",
    secure=True
)


def upload_image(image_path):
    upload_result = cloudinary.uploader.upload(image_path, public_id="test")
    imageURL = upload_result["secure_url"]
    return imageURL


def call_ocr_api(image_path):
    url_ocr = "https://api.aiforthai.in.th/panyapradit-ocr"
    files = {'file': open(image_path, 'rb')}
    headers = {'Apikey': "QQAfpfak9Ot0HLeGklytNd5EJl9f4jaE"}
    response = requests.post(url_ocr, files=files, headers=headers)
    return response.content


def save_to_firestore(data):
    db.collection('detections').add(data)


def custom_on_prediction(results, frame):
    predictions = results.get('predictions', [])
    detected_classes = set()

    for prediction in predictions:
        label = prediction['class']
        print(f"Detected: {label}")
        detected_classes.add(label)

    target_classes = {"without-helmet", "motorcycle", "license-plate"}
    if target_classes.issubset(detected_classes):
        if not hasattr(custom_on_prediction, 'last_capture_time') or \
                time.time() - custom_on_prediction.last_capture_time >= 8:

            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"captured_image_{timestamp}.jpg"
            if isinstance(frame, np.ndarray):
                frame_to_save = frame
            else:
                frame_to_save = cv2.cvtColor(np.array(frame.image), cv2.COLOR_BGR2RGB)

            cv2.imwrite(filename, frame_to_save)
            print(f"Image saved as {filename}")

            # Upload the image
            image_url = upload_image(filename)
            print(f"Image uploaded to {image_url}")

            # Call the OCR API
            ocr_results = call_ocr_api(filename)
            print(f"OCR Results: {ocr_results}")

            data = {
                'imgbb_url': image_url,
                'recognition': ocr_results
            }
            save_to_firestore(data)

            custom_on_prediction.last_capture_time = time.time()


pipeline = InferencePipeline.init(
    model_id="final-9tgod/1",
    video_reference="img_4652.mp4",
    on_prediction=custom_on_prediction,
    api_key="rVXyPZdX0nQ5LXH7m0fr",
)

pipeline.start()
pipeline.join()
