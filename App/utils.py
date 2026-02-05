## transform text
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import json
from google.cloud import storage
from dotenv import load_dotenv



def download_model():
    load_dotenv(override=True)
    
    creds = json.loads(os.environ["GCP_SECRET"])
    creds_path = "/tmp/gcp_key.json"
    
    with open(creds_path, "w") as f:
        json.dump(creds, f)
        
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path

    model_folder = os.environ["MODEL_FOLDER"]
    client = storage.Client()
    bucket = client.bucket(os.environ["GCS_BUCKET"])
    
    print("BUCKET NAME:", bucket)
    bloblist = bucket.list_blobs()

    for blob in bloblist:
        print("BLOB NAME:", blob.name)

        local_path = os.path.join("/tmp", blob.name)
        
        # créer les dossiers parents
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        blob.download_to_filename(local_path)
        
    return os.path.join("/tmp", model_folder)
    
    
    
    
        
    
    
    



    
    
    
    
