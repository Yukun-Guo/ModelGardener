#!/usr/bin/env python3
"""
Test script for the new preprocessing and callbacks parameter groups.

This script demonstrates the new functionality added to the Model Gardener
for custom preprocessing and callbacks loading.
"""

import sys
import os

# Add current directory to path to import main module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_preprocessing_and_callbacks():
    """Test the new preprocessing and callbacks groups."""
    try:
        from main import (
            create_comprehensive_config, 
            dict_to_params, 
            PreprocessingGroup, 
            CallbacksGroup,
            extract_preprocessing_config,
            extract_callbacks_config
        )
        
        print("✓ Successfully imported new preprocessing and callbacks classes")
        
        # Test configuration creation
        config = create_comprehensive_config()
        
        # Check that new sections exist
        assert 'preprocessing' in config['advanced'], "Preprocessing section missing from advanced config"
        assert 'callbacks' in config['advanced'], "Callbacks section missing from advanced config"
        
        print("✓ New sections found in configuration structure")
        
        # Test parameter tree creation
        params = dict_to_params(config)
        
        print("✓ Successfully created parameter tree with new groups")
        
        # Test group instantiation
        preprocessing_group = PreprocessingGroup(name='test_preprocessing')
        callbacks_group = CallbacksGroup(name='test_callbacks')
        
        print("✓ Successfully instantiated new parameter groups")
        
        # Check preset methods
        preprocessing_methods = [child.name() for child in preprocessing_group.children()]
        callbacks_methods = [child.name() for child in callbacks_group.children()]
        
        expected_preprocessing = ['Resizing', 'Normalization', 'Load Custom Preprocessing']
        expected_callbacks = [
            'Early Stopping', 'Learning Rate Scheduler', 'Model Checkpoint', 
            'CSV Logger', 'TensorBoard', 'Load Custom Callbacks'
        ]
        
        for method in expected_preprocessing:
            assert method in preprocessing_methods, f"Missing preprocessing method: {method}"
        
        for method in expected_callbacks:
            assert method in callbacks_methods, f"Missing callback method: {method}"
        
        print("✓ All expected preset methods found")
        
        # Test configuration extraction
        prep_config = extract_preprocessing_config(preprocessing_group)
        callbacks_config = extract_callbacks_config(callbacks_group)
        
        print("✓ Successfully extracted configurations from groups")
        
        print("\n🎉 All tests passed! The new preprocessing and callbacks functionality is working correctly.")
        
        # Print summary of features
        print("\n📋 New Features Summary:")
        print("1. PreprocessingGroup with preset methods:")
        for method in expected_preprocessing[:-1]:  # Exclude button
            print(f"   - {method}")
        
        print("\n2. CallbacksGroup with preset methods:")
        for method in expected_callbacks[:-1]:  # Exclude button
            print(f"   - {method}")
            
        print("\n3. Custom loading capabilities:")
        print("   - Load custom preprocessing functions from Python files")
        print("   - Load custom callback functions/classes from Python files")
        print("   - AST parsing for automatic parameter detection")
        print("   - Support for both 2D and 3D data processing")
        
        print("\n4. Example files created:")
        print("   - example_custom_preprocessing.py (7 example functions)")
        print("   - example_custom_callbacks.py (6 example callbacks)")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_preprocessing_and_callbacks()
    sys.exit(0 if success else 1)
