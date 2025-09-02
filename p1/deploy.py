#!/usr/bin/env python3
"""
Deployment Script for ModelGardener
Generated on: 2025-09-02 15:47:55
Configuration: config.yaml
"""

import os
import sys
import yaml
import tensorflow as tf
import keras
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import io
import base64
from pathlib import Path

# No custom functions

app = Flask(__name__)

# Global variables
model = None
class_labels = []
target_size = (224, 224)

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def preprocess_image(image_data, target_size=(224, 224)):
    """Preprocess image for prediction."""
    try:
        # If image_data is base64 string, decode it
        if isinstance(image_data, str):
            image_data = base64.b64decode(image_data)
        
        # Load image
        img = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize
        img = img.resize(target_size)
        
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint."""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get image from request
        if 'image' not in request.files:
            # Try to get base64 encoded image from JSON
            data = request.get_json()
            if data and 'image' in data:
                image_data = data['image']
            else:
                return jsonify({'error': 'No image provided'}), 400
        else:
            image_file = request.files['image']
            image_data = image_file.read()
        
        # Preprocess image
        img_array = preprocess_image(image_data, target_size)
        if img_array is None:
            return jsonify({'error': 'Failed to preprocess image'}), 400
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = class_labels[predicted_class_idx] if class_labels else f"class_{predicted_class_idx}"
        
        # Get top 5 predictions
        top_5_indices = np.argsort(predictions[0])[::-1][:5]
        top_5_predictions = [
            {
                'class': class_labels[idx] if class_labels else f"class_{idx}",
                'probability': float(predictions[0][idx])
            }
            for idx in top_5_indices
        ]
        
        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence,
            'top_predictions': top_5_predictions
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get available classes."""
    return jsonify({'classes': class_labels})

def load_model_and_classes():
    """Load model and class labels."""
    global model, class_labels, target_size
    
    # Configuration
    config_file = "config.yaml"
    model_dir = "./logs"
    img_height = 32
    img_width = 32
    target_size = (img_height, img_width)
    
    # Load configuration if available
    if os.path.exists(config_file):
        try:
            config = load_config(config_file)
            print(f"✅ Loaded configuration from {config_file}")
        except Exception as e:
            print(f"⚠️  Could not load configuration: {e}")
            config = {}
    else:
        print(f"⚠️  Configuration file {config_file} not found, using defaults")
        config = {}
    
        # No custom functions to load
    custom_functions = None
    
    # Find model file
    model_files = []
    if os.path.exists(model_dir):
        for file in os.listdir(model_dir):
            if file.endswith('.h5') and ('best' in file or 'final' in file):
                model_files.append(os.path.join(model_dir, file))
    
    if not model_files:
        print(f"❌ No model files found in {model_dir}")
        return False
    
    # Use the best model if available
    model_file = None
    for file in model_files:
        if 'best' in os.path.basename(file):
            model_file = file
            break
    if not model_file:
        model_file = model_files[0]
    
    print(f"📥 Loading model from: {model_file}")
    
    # Load model
    try:
        model = keras.models.load_model(model_file)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Try to get class labels from training directory structure
    train_dir = "./data"
    if os.path.exists(train_dir):
        class_labels = [d for d in os.listdir(train_dir) 
                      if os.path.isdir(os.path.join(train_dir, d))]
        class_labels.sort()
        print(f"📋 Detected class labels: {class_labels}")
    
    # Default class labels if none found
    if not class_labels:
        num_classes = 10
        class_labels = [f"class_{i}" for i in range(num_classes)]
        print(f"⚠️  Using default class labels: {class_labels}")
    
    return True

@app.route('/', methods=['GET'])
def home():
    """Home page with API documentation."""
    docs = """
    <h1>ModelGardener Model API</h1>
    <p>Generated on: 2025-09-02 15:47:55</p>
    
    <h2>Endpoints:</h2>
    <ul>
        <li><strong>GET /health</strong> - Health check</li>
        <li><strong>POST /predict</strong> - Make prediction on uploaded image</li>
        <li><strong>GET /classes</strong> - Get available classes</li>
    </ul>
    
    <h2>Usage Examples:</h2>
    
    <h3>Python requests:</h3>
    <pre>
import requests

# Predict with file upload
with open('image.jpg', 'rb') as f:
    response = requests.post('http://localhost:5000/predict', files={'image': f})
    result = response.json()
    print(result)

# Predict with base64 encoded image
import base64
with open('image.jpg', 'rb') as f:
    image_b64 = base64.b64encode(f.read()).decode()

response = requests.post('http://localhost:5000/predict', 
                        json={'image': image_b64})
result = response.json()
print(result)
    </pre>
    
    <h3>cURL:</h3>
    <pre>
# Upload file
curl -X POST -F "image=@image.jpg" http://localhost:5000/predict

# Health check
curl http://localhost:5000/health

# Get classes
curl http://localhost:5000/classes
    </pre>
    """
    return docs

def main():
    """Main deployment function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy ModelGardener model as REST API')
    parser.add_argument('--host', default='0.0.0.0', help='Host address (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='Port number (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    print("🚀 Starting ModelGardener Model API...")
    print("=" * 50)
    
    # Load model and classes
    if not load_model_and_classes():
        print("❌ Failed to load model")
        sys.exit(1)
    
    print(f"🌐 Starting server on {args.host}:{args.port}")
    print("📖 API documentation available at: http://localhost:{}/".format(args.port))
    print("💡 Use Ctrl+C to stop the server")
    
    # Start Flask app
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
