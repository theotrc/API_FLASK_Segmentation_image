from flask import Flask
import keras
from App.utils import download_model




app = Flask(__name__)

model_path = download_model()

print("Model downloaded to:", model_path)

dpl_model = keras.saving.load_model(model_path,
                                       compile=False)
# dpl_model = keras.saving.load_model("App/models/DeepLabV3PlusDiceV2.keras",
#                                        compile=False)



from App import views



