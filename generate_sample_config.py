#!/usr/bin/env python3
"""
Generate a sample configuration file with the new simplified structure
"""

import sys
import os
import json
from pathlib import Path

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from cli_config import ModelConfigCLI

def generate_sample_config():
    """Generate a sample configuration file with the new structure."""
    cli = ModelConfigCLI()
    config = cli.create_default_config()
    
    # Customize some values for demonstration
    config['configuration']['data']['train_dir'] = './example_data/train'
    config['configuration']['data']['val_dir'] = './example_data/val'
    config['configuration']['data']['data_loader']['parameters']['batch_size'] = 32
    
    # Enable some augmentation
    config['configuration']['data']['augmentation']['Horizontal Flip']['enabled'] = True
    config['configuration']['data']['augmentation']['Rotation']['enabled'] = True
    config['configuration']['data']['augmentation']['Rotation']['angle_range'] = 10.0
    
    # Enable some callbacks
    config['configuration']['model']['callbacks']['Early Stopping']['enabled'] = True
    config['configuration']['model']['callbacks']['Early Stopping']['patience'] = 10
    config['configuration']['model']['callbacks']['Model Checkpoint']['enabled'] = True
    
    # Enable cross-validation
    config['configuration']['training']['cross_validation']['enabled'] = True
    config['configuration']['training']['cross_validation']['k_folds'] = 5
    config['configuration']['training']['epochs'] = 50
    
    # Model configuration
    config['configuration']['model']['model_family'] = 'resnet'
    config['configuration']['model']['model_name'] = 'ResNet-50'
    config['configuration']['model']['model_parameters']['classes'] = 10
    
    return config

def main():
    print("📄 Generating Sample Configuration with Simplified Structure")
    print("=" * 60)
    
    config = generate_sample_config()
    
    # Save as JSON
    output_path = Path(__file__).parent / 'sample_simplified_config.json'
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Sample configuration saved to: {output_path}")
    
    # Show the structure
    print("\n📋 New Configuration Structure:")
    print("configuration/")
    print("  ├── task_type: image_classification")
    print("  ├── data/")
    print("  │   ├── train_dir, val_dir")
    print("  │   ├── data_loader/ (batch_size, etc.)")
    print("  │   ├── preprocessing/ (resizing, normalization)")
    print("  │   └── augmentation/ ← MOVED FROM ADVANCED")
    print("  ├── model/")
    print("  │   ├── model_family, model_name, model_parameters")
    print("  │   ├── optimizer/")
    print("  │   ├── loss_functions/")
    print("  │   ├── metrics/")
    print("  │   └── callbacks/ ← MOVED FROM ADVANCED")
    print("  ├── training/")
    print("  │   ├── epochs, learning_rate, etc.")
    print("  │   ├── cross_validation/ ← MOVED FROM ADVANCED")
    print("  │   └── training_loop/")
    print("  └── runtime/")
    print("      └── model_dir, distribution_strategy, etc.")
    
    print(f"\n🗑️  Advanced section: REMOVED (simplified configuration)")
    
    print(f"\n📁 Custom functions path: All custom functions should be placed in a 'src' directory")
    print(f"📁 Generated scripts: train.py, evaluation.py, prediction.py, deploy.py will be generated")
    
    # Show sample content
    print(f"\n📄 Sample configuration content (basic level):")
    basic_config = config['configuration']
    
    print("• Data augmentation enabled:")
    aug_config = basic_config['data']['augmentation']
    for name, settings in aug_config.items():
        if settings.get('enabled', False):
            print(f"  - {name}: probability={settings.get('probability', 'N/A')}")
    
    print("• Callbacks enabled:")
    callback_config = basic_config['model']['callbacks']
    for name, settings in callback_config.items():
        if settings.get('enabled', False):
            print(f"  - {name}: {settings}")
    
    print("• Cross-validation:")
    cv_config = basic_config['training']['cross_validation']
    if cv_config.get('enabled', False):
        print(f"  - K-folds: {cv_config.get('k_folds', 'N/A')}")
        print(f"  - Stratified: {cv_config.get('stratified', 'N/A')}")

if __name__ == "__main__":
    main()
