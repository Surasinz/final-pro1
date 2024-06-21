import cv2
import time
import firebase_admin
import numpy as np
import requests
from inference import InferencePipeline
from firebase_admin import credentials, firestore
from inference.core.interfaces.stream.sinks import render_boxes

cred = credentials.Certificate('finalproject-f1a6e-firebase-adminsdk-rb9vv-5f590cfe25.json')
firebase_admin.initialize_app(cred)
db = firestore.client()


def custom_on_prediction(results, frame):
    predictions = results.get('predictions', [])
    detected_classes = set()

    for prediction in predictions:
        label = prediction['class']
        print(f"ตรวจพบ: {label}")
        detected_classes.add(label)

    target_classes = {"without-helmet", "motorcycle", "license-plate"}
    if target_classes.issubset(detected_classes):
        if not hasattr(custom_on_prediction, 'last_capture_time') or \
                time.time() - custom_on_prediction.last_capture_time >= 7:

            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"captured_image_{timestamp}.jpg"
            if hasattr(frame, 'image'):
                frame = frame.image
            if not isinstance(frame, np.ndarray):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imwrite(filename, frame)
            print(f"บันทึกภาพเป็น {filename}")

            # อัปโหลดไป ImgBB
            imgbb_response = handler({'imgbb': {'$auth': {'api_key': 'YOUR_IMGBB_API_KEY'}}}, filename)

            if imgbb_response['status_code'] == 200:
                imgbb_url = imgbb_response['data']['url']

                # เรียก API OCR จาก AI for Thai
                ocr_response = call_ocr_api(filename)

                # อัปโหลดข้อมูลไปยัง Firebase
                upload_to_firebase({'recognition': ocr_response, 'imgbb_url': imgbb_url})
            else:
                print("ImgBB upload failed!")

            custom_on_prediction.last_capture_time = time.time()

    render_boxes(predictions, frame)


def handler(pd, image_path):
    with open(image_path, 'rb') as f:
        files = {
            'image': f,
            'key': (None, pd['imgbb']['$auth']['api_key']),
            'name': (None, '')
        }
        r = requests.post('https://api.imgbb.com/1/upload', files=files)
        return r.json()


def call_ocr_api(image_path):
    url_ocr = "https://api.aiforthai.in.th/panyapradit-ocr"
    files = {'file': open(image_path, 'rb')}
    headers = {'Apikey': "QQAfpfak9Ot0HLeGklytNd5EJl9f4jaE"}
    response = requests.post(url_ocr, files=files, headers=headers)
    return response.json()


def upload_to_firebase(info):
    doc_ref = db.collection('responses').document()
    doc_ref.set({
        'recognition': info['recognition'],
        'imgbb_url': info['imgbb_url'],
        'timestamp': firestore.SERVER_TIMESTAMP
    })
    print("Uploaded response to Firebase")


pipeline = InferencePipeline.init(
    model_id="plan-d/3",
    video_reference="img_4652.mp4",
    on_prediction=custom_on_prediction,
    api_key="ygAKWxk6wdtHW8QKwFed",
)

pipeline.start()
pipeline.join()
