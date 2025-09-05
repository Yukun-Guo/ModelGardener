#!/usr/bin/env python3
"""
Evaluation Script for ModelGardener
Generated on: 2025-09-04 17:04:12
Configuration: config.yaml
"""

import os
import sys
import yaml
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from pathlib import Path

# No custom functions

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_test_generator(test_dir, batch_size=32, img_height=224, img_width=224):
    """Create test data generator."""
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False  # Important for evaluation
    )
    
    return test_generator

def evaluate_model():
    """Main evaluation function."""
    
    # Configuration
    config_file = "config.yaml"
    test_dir = "./data/test"
    model_dir = "./logs"
    batch_size = 32
    img_height = 32
    img_width = 32
    
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
        return
    
    # Use the best model if available, otherwise use the first one
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
        return
    
    # Create test generator
    if not os.path.exists(test_dir):
        print(f"❌ Test directory not found: {test_dir}")
        return
    
    print("📁 Creating test data generator...")
    test_gen = create_test_generator(test_dir, batch_size, img_height, img_width)
    
    # Evaluate model
    print("📊 Evaluating model...")
    
    # Get predictions
    predictions = model.predict(test_gen, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Get true labels
    true_classes = test_gen.classes
    class_labels = list(test_gen.class_indices.keys())
    
    # Calculate accuracy
    accuracy = np.mean(predicted_classes == true_classes)
    print(f"🎯 Test Accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\n📋 Classification Report:")
    report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    print(report)
    
    # Confusion matrix
    print("🔄 Generating confusion matrix...")
    cm = confusion_matrix(true_classes, predicted_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Save confusion matrix
    cm_path = os.path.join(model_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"💾 Confusion matrix saved to: {cm_path}")
    plt.show()
    
    # Save evaluation results
    results = {
        'accuracy': float(accuracy),
        'classification_report': report,
        'model_file': model_file,
        'test_directory': test_dir,
        'num_samples': len(true_classes),
        'num_classes': len(class_labels),
        'class_labels': class_labels
    }
    
    results_path = os.path.join(model_dir, 'evaluation_results.yaml')
    with open(results_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    print(f"💾 Evaluation results saved to: {results_path}")
    
    print("✅ Evaluation completed!")

if __name__ == "__main__":
    evaluate_model()
