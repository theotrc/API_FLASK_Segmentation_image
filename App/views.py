from flask import request, jsonify
from App import app
from logging import FileHandler, WARNING
from App import unet
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
            print("image filename:", img.filename)
            npimg = np.frombuffer(img.read(), np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
            print("Image shape:", img.shape)
            img_resized = cv2.resize(img, (160, 160))
            img_normalized = img_resized / 255.0  # Normalize to [0, 1]
            img_batch = np.expand_dims(img_normalized, axis=0)
            print("Resized image shape:", img_resized.shape)
            
            prediction = unet.predict(img_batch)
            mask = np.argmax(prediction, axis=-1)[0]
            mask = (mask * 255).astype(np.uint8)
            
            mask_resized = cv2.resize(
                mask,
                (img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            
            _, buffer = cv2.imencode(".png", mask_resized)
            mask_base64 = base64.b64encode(buffer).decode("utf-8")
                        
            return jsonify({'message': f'Image received and saved successfully.',
                            "mask":mask_base64}), 200
    # else:
    #     text = request.args.get('text')

    if not img:
        return jsonify({'error': 'No img provided. Send JSON {"text": "..."} or form/query param "text".'}), 400

