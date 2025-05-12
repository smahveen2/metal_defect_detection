from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image  # Added import
import numpy as np
import os

app = Flask(__name__)

# Load your .h5 model
model = load_model('static/model.h5')  # Verify this path exists

# Class names - must match your model's training classes
CLASS_NAMES = ['crack', 'scratch', 'corrosion', 'normal']  # Update if different

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("\n--- New Prediction Request ---")
        
        # 1. Check file
        if 'file' not in request.files:
            print("Error: No file uploaded")
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        if file.filename == '':
            print("Error: Empty filename")
            return jsonify({'error': 'No file selected'}), 400
        
        # 2. Process image
        print(f"Processing: {file.filename}")
        img = Image.open(file.stream).convert('RGB')
        img = img.resize((256, 256))
        img_array = np.array(img) / 255.0
        print(f"Image shape: {img_array.shape}")  # Should be (256,256,3)
        
        # 3. Predict
        img_array = np.expand_dims(img_array, axis=0)
        print(f"Model input shape: {img_array.shape}")  # Should be (1,256,256,3)
        
        # Verify model and input match
        print(f"Model expects: {model.input_shape}")
        print(f"Actual input: {img_array.shape}")
        
        pred = model.predict(img_array)
        print("Raw predictions:", pred)
        
        class_idx = np.argmax(pred[0])
        confidence = float(np.max(pred[0]))
        predicted_class = CLASS_NAMES[class_idx]
        print(f"Predicted: {predicted_class} ({confidence:.2%})")
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence
        })
        
    except Exception as e:
        print(f"Critical Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
