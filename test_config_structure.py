#!/usr/bin/env python3
"""
Test script to verify the restructured configuration with loss functions and metrics under model section.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from main_window import MainWindow

def test_configuration_structure():
    """Test that the configuration structure is correctly organized."""
    
    # Create a MainWindow instance to access the configuration
    main_win = MainWindow(experiment_name="test_config")
    
    # Get the basic configuration
    basic_config = main_win.create_comprehensive_config()['basic']
    
    print("=== Configuration Structure Test ===")
    
    # Test model section structure
    if 'model' in basic_config:
        model_section = basic_config['model']
        print(f"✓ Model section found")
        
        # Check if optimizer is in model section
        if 'optimizer' in model_section:
            print(f"✓ Optimizer found in model section")
        else:
            print(f"✗ Optimizer NOT found in model section")
        
        # Check if loss_functions is in model section
        if 'loss_functions' in model_section:
            print(f"✓ Loss functions found in model section")
            print(f"  - Type: {model_section['loss_functions']['type']}")
        else:
            print(f"✗ Loss functions NOT found in model section")
        
        # Check if metrics is in model section
        if 'metrics' in model_section:
            print(f"✓ Metrics found in model section")
            print(f"  - Type: {model_section['metrics']['type']}")
        else:
            print(f"✗ Metrics NOT found in model section")
            
        print(f"Model section keys: {list(model_section.keys())}")
    else:
        print(f"✗ Model section NOT found")
    
    # Test training section structure
    if 'training' in basic_config:
        training_section = basic_config['training']
        print(f"✓ Training section found")
        
        # Check that loss_functions is NOT in training section
        if 'loss_functions' not in training_section:
            print(f"✓ Loss functions correctly removed from training section")
        else:
            print(f"✗ Loss functions still in training section (should be moved)")
        
        # Check that metrics is NOT in training section
        if 'metrics' not in training_section:
            print(f"✓ Metrics correctly removed from training section")
        else:
            print(f"✗ Metrics still in training section (should be moved)")
            
        print(f"Training section keys: {list(training_section.keys())}")
    else:
        print(f"✗ Training section NOT found")
    
    print("\n=== Structure Summary ===")
    print("New logical organization:")
    print("  model/")
    print("    ├── backbone_type, model_id, dropout_rate, activation")
    print("    ├── optimizer/")
    print("    ├── loss_functions/")
    print("    └── metrics/")
    print("  training/")
    print("    └── epochs, learning_rate, momentum, etc.")

if __name__ == '__main__':
    test_configuration_structure()
