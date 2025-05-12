import os
import gdown
from tensorflow.keras.models import load_model

# Path where you want to save the model
model_path = "model.h5"

# Google Drive file ID from your link
gdrive_id = "1tNTnMZh1ReAqkqeWMWufqIkfoH49p-hg"  # Replace with your actual ID

# Google Drive download URL
url = f"https://drive.google.com/uc?id={gdrive_id}"

# Function to download the model if not already present
def download_model():
    if not os.path.exists(model_path):
        print("Model not found locally. Downloading model from Google Drive...")
        try:
            gdown.download(url, model_path, quiet=False, fuzzy=True)
            print("Model downloaded successfully.")
        except Exception as e:
            print(f"Error downloading the model: {e}")
            raise

# Ensure the model is downloaded before loading it
download_model()

# Load the model
model = load_model(model_path)
print("Model loaded successfully.")
