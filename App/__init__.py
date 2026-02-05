import os
from flask import Flask
from App.utils import download_model



app = Flask(__name__)

model_path = download_model()

print("Model downloaded to:", model_path)


# During testing we avoid importing heavy TensorFlow packages and loading
# the model at import time. CI/tests should set `TESTING=1` to use a stub.


from App import views



