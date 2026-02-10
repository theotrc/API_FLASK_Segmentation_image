from flask import request, jsonify
from App import app
from logging import FileHandler, WARNING
from App import unet


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
            print("image:", img.read())
            prediction = unet.predict(img.read())
            return jsonify({'message': f'Image received and saved successfully.'}), 200
    # else:
    #     text = request.args.get('text')

    if not img:
        return jsonify({'error': 'No img provided. Send JSON {"text": "..."} or form/query param "text".'}), 400

    # # allow a single string or a list of strings
    # if isinstance(text, str):
    #     texts = [text]
    # elif isinstance(text, (list, tuple)):
    #     texts = list(text)
    # else:
    #     return jsonify({'error': 'Invalid text format; must be string or list of strings.'}), 400

