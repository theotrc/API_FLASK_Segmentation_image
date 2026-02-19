from flask import request, jsonify
from App import app
from logging import FileHandler, WARNING
from App import dpl_model
import numpy as np
import cv2
import base64




file_handler = FileHandler('errorlog.txt')
file_handler.setLevel(WARNING)
app.logger.addHandler(file_handler)

@app.route("/")
def home():
    return "api segmentation image pour voiture autonome"

@app.route("/predict", methods=["GET", "POST"])
def predict():

    img = None
    if request.method == 'POST':
        print("POST request received")
        print("Request content type:", request.content_type)
        img = request.files.get("image", None)
        print("Received image:", img)
        if img:
            IMAGE_SIZE = 512
            print("image filename:", img.filename)
            npimg = np.frombuffer(img.read(), np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
            
            img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            img_normalized = np.float32(img_resized)  
            img_batch = np.expand_dims(img_normalized, axis=0)
            
            
            prediction = dpl_model.predict(img_batch)
            prediction = np.squeeze(prediction)
            prediction = prediction.reshape(IMAGE_SIZE, IMAGE_SIZE, 8)
            prediction = cv2.resize(prediction, (img.shape[1], img.shape[0]))
            mask_prediction = np.argmax(prediction, axis=-1)
            
            
            _, buffer = cv2.imencode(".png", mask_prediction)
            mask_base64 = base64.b64encode(buffer).decode("utf-8")
                        
            return jsonify({'message': f'Image received and saved successfully.',
                            "mask":mask_base64}), 200
    # else:
    #     text = request.args.get('text')

    if not img:
        return jsonify({'error': 'No img provided. Send JSON {"text": "..."} or form/query param "text".'}), 400

