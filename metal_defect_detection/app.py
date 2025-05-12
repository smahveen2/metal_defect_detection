from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load your .h5 model
model = load_model('static/model.h5')  # Changed to .h5

# Class names - update with your actual classes
CLASS_NAMES = ['crack', 'scratch', 'corrosion', 'normal']  # Example

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    # Save temporarily
    img_path = 'static/temp/temp.jpg'
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    file.save(img_path)
    
    # Preprocess image
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    
    return jsonify({
        'prediction': CLASS_NAMES[predicted_class],
        'confidence': float(np.max(predictions[0])),
        'image_url': img_path  # Return path to display the image
    })

if __name__ == '__main__':
    app.run(debug=True)
