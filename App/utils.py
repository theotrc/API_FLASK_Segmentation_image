import os
import json
from google.cloud import storage
from dotenv import load_dotenv
from keras.saving import register_keras_serializable
import tensorflow as tf





@register_keras_serializable()
class MeanIoUSoftmax(tf.keras.metrics.MeanIoU):
    def __init__(self, num_classes=8, name="mean_iou", **kwargs):
        super().__init__(num_classes=num_classes, name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

    
    
    
    
        
    
    
    



    
    
    
    
