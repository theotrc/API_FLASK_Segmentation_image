from flask import Flask
import keras
from App.utils import MeanIoUSoftmax



app = Flask(__name__)

unet = keras.models.load_model("App/models/saved_model2.keras",
                                       custom_objects={"MeanIoUSoftmax": MeanIoUSoftmax})



from App import views



