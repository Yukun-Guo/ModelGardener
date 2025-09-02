#!/usr/bin/env python3
"""
Prediction Script for ModelGardener
Generated on: 2025-09-02 12:04:32
Configuration: config.yaml
"""

import os
import sys
import yaml
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import argparse
from pathlib import Path

# No custom functions

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess a single image for prediction."""
    try:
        # Load image
        img = Image.open(image_path)
        
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
        print(f"Error preprocessing image {image_path}: {e}")
        return None

def predict_single_image(model, image_path, class_labels, target_size=(224, 224)):
    """Predict class for a single image."""
    
    # Preprocess image
    img_array = preprocess_image(image_path, target_size)
    if img_array is None:
        return None, None, None
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx])
    predicted_class = class_labels[predicted_class_idx]
    
    return predicted_class, confidence, predictions[0]

def predict_batch(model, image_dir, class_labels, target_size=(224, 224)):
    """Predict classes for all images in a directory."""
    
    results = []
    
    # Supported image extensions
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    # Find all images
    image_files = []
    for ext in extensions:
        image_files.extend(Path(image_dir).glob(f'*{ext}'))
        image_files.extend(Path(image_dir).glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return results
    
    print(f"Found {len(image_files)} images")
    
    # Process each image
    for image_file in image_files:
        predicted_class, confidence, probabilities = predict_single_image(
            model, str(image_file), class_labels, target_size
        )
        
        if predicted_class is not None:
            result = {
                'image_path': str(image_file),
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': probabilities.tolist()
            }
            results.append(result)
            print(f"📷 {image_file.name}: {predicted_class} ({confidence:.3f})")
        else:
            print(f"❌ Failed to process: {image_file.name}")
    
    return results

def main():
    """Main prediction function."""
    
    parser = argparse.ArgumentParser(description='Make predictions using trained ModelGardener model')
    parser.add_argument('--input', '-i', required=True, 
                       help='Input image file or directory containing images')
    parser.add_argument('--model', '-m', 
                       help='Path to model file (if not specified, will search in model directory)')
    parser.add_argument('--output', '-o', 
                       help='Output file to save results (JSON format)')
    parser.add_argument('--top-k', '-k', type=int, default=5,
                       help='Show top K predictions (default: 5)')
    
    args = parser.parse_args()
    
    # Configuration
    config_file = "config.yaml"
    model_dir = "./logs"
    img_height = 224
    img_width = 224
    target_size = (img_height, img_width)
    
    # Load configuration if available
    class_labels = []
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
    
    # Find model file
    model_file = args.model
    if not model_file:
        model_files = []
        if os.path.exists(model_dir):
            for file in os.listdir(model_dir):
                if file.endswith('.h5') and ('best' in file or 'final' in file):
                    model_files.append(os.path.join(model_dir, file))
        
        if not model_files:
            print(f"❌ No model files found in {model_dir}")
            return
        
        # Use the best model if available
        for file in model_files:
            if 'best' in os.path.basename(file):
                model_file = file
                break
        if not model_file:
            model_file = model_files[0]
    
    if not os.path.exists(model_file):
        print(f"❌ Model file not found: {model_file}")
        return
    
    print(f"📥 Loading model from: {model_file}")
    
    # Load model
    try:
        model = keras.models.load_model(model_file)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Try to get class labels from training directory structure
    if not class_labels:
        train_dir = ""
        if os.path.exists(train_dir):
            class_labels = [d for d in os.listdir(train_dir) 
                          if os.path.isdir(os.path.join(train_dir, d))]
            class_labels.sort()
            print(f"📋 Detected class labels from training data: {class_labels}")
    
    # Default class labels if none found
    if not class_labels:
        num_classes = 1000
        class_labels = [f"class_{i}" for i in range(num_classes)]
        print(f"⚠️  Using default class labels: {class_labels}")
    
    # Check input
    input_path = args.input
    if not os.path.exists(input_path):
        print(f"❌ Input path not found: {input_path}")
        return
    
    # Make predictions
    results = []
    
    if os.path.isfile(input_path):
        # Single image prediction
        print(f"🔍 Predicting single image: {input_path}")
        predicted_class, confidence, probabilities = predict_single_image(
            model, input_path, class_labels, target_size
        )
        
        if predicted_class is not None:
            print(f"\n🎯 Prediction: {predicted_class}")
            print(f"🎲 Confidence: {confidence:.4f}")
            
            # Show top-k predictions
            top_k_indices = np.argsort(probabilities)[::-1][:args.top_k]
            print(f"\n📊 Top-{args.top_k} predictions:")
            for i, idx in enumerate(top_k_indices, 1):
                print(f"  {i}. {class_labels[idx]}: {probabilities[idx]:.4f}")
            
            results.append({
                'image_path': input_path,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'top_predictions': [
                    {'class': class_labels[idx], 'probability': float(probabilities[idx])}
                    for idx in top_k_indices
                ]
            })
    
    elif os.path.isdir(input_path):
        # Batch prediction
        print(f"📁 Predicting batch of images in: {input_path}")
        results = predict_batch(model, input_path, class_labels, target_size)
    
    else:
        print(f"❌ Invalid input path: {input_path}")
        return
    
    # Save results if requested
    if args.output and results:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to: {args.output}")
    
    print("✅ Prediction completed!")

if __name__ == "__main__":
    main()
