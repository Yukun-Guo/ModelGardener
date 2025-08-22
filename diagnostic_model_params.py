#!/usr/bin/env python3
"""
Simple diagnostic tool to verify model parameters are working
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_model_parameters_display():
    """Simple test to verify model parameters system"""
    print("🔍 MODEL PARAMETERS DIAGNOSTIC")
    print("=" * 50)
    
    # Test 1: Check if ModelGroup works in isolation
    print("\n1. Testing ModelGroup in isolation...")
    try:
        from model_group import ModelGroup
        model_group = ModelGroup(name='test', model_name='ResNet50', task_type='image_classification')
        print(f"   ✓ ModelGroup creates {len(model_group.children())} parameters")
        
        # Check specific parameters
        expected_params = ['input_shape', 'include_top', 'weights', 'pooling', 'classes', 'classifier_activation']
        found_params = [child.name() for child in model_group.children() if child.name() != 'load_custom_model']
        
        print(f"   Expected: {len(expected_params)} keras.applications params")
        print(f"   Found: {len(found_params)} params")
        print(f"   Match: {set(expected_params) <= set(found_params)}")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 2: Check parameter type registration
    print("\n2. Testing parameter type registration...")
    try:
        from pyqtgraph.parametertree import parameterTypes as pTypes
        from model_group import ModelGroup
        
        # Register the type (like main_window.py does)
        pTypes.registerParameterType('model_group', ModelGroup, override=True)
        print("   ✓ model_group type registered successfully")
        
        # Try to create a parameter with this type
        from pyqtgraph.parametertree import Parameter
        test_param = Parameter.create(
            name='test_model_params',
            type='model_group',
            model_name='ResNet50',
            task_type='image_classification'
        )
        print(f"   ✓ Created model_group parameter with {len(test_param.children())} children")
        
        # List the children
        for child in test_param.children():
            if child.name() != 'load_custom_model':
                try:
                    value = child.value()
                    print(f"     - {child.name()}: {value}")
                except:
                    print(f"     - {child.name()}: (group)")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Verify the parameter update mechanism
    print("\n3. Testing parameter update mechanism...")
    try:
        from model_group import ModelGroup
        model_group = ModelGroup(name='test', model_name='ResNet50', task_type='image_classification')
        
        print(f"   Initial: {len(model_group.children())} children (ResNet50)")
        
        # Update to MobileNet (should have more parameters)
        model_group.update_model_selection('MobileNet', 'image_classification')
        print(f"   After update: {len(model_group.children())} children (MobileNet)")
        
        # Check for MobileNet-specific parameter
        mobilenet_specific = ['alpha', 'depth_multiplier', 'dropout']
        found_specific = [child.name() for child in model_group.children() 
                         if child.name() in mobilenet_specific]
        print(f"   MobileNet-specific params found: {found_specific}")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 CONCLUSION:")
    print("If all tests pass, the ModelGroup is working correctly.")
    print("The issue might be in the GUI integration or parameter tree refresh.")
    print("Try changing the model selection in the GUI to see if parameters update.")
    print("=" * 50)

if __name__ == "__main__":
    test_model_parameters_display()
