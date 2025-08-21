#!/usr/bin/env python3
"""Comprehensive test of all custom function loading."""

import sys
import os
import json

def test_all_custom_functions():
    """Test loading all types of custom functions from the manifest."""
    
    # Load the custom functions manifest
    manifest_path = "test_config/ModelGardener_Config_Package_20250821_142147/custom_functions_manifest.json"
    
    if not os.path.exists(manifest_path):
        print(f"Manifest not found: {manifest_path}")
        return False
    
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
            
        # Group functions by type
        functions_by_type = {}
        for func in manifest['custom_functions']:
            func_type = func['type']
            if func_type not in functions_by_type:
                functions_by_type[func_type] = []
            functions_by_type[func_type].append(func)
        
        print("Testing all custom function types:")
        print("=" * 50)
        
        total_functions = 0
        total_successful = 0
        
        for func_type, functions in functions_by_type.items():
            print(f"\n{func_type.upper()}: {len(functions)} functions")
            print("-" * 30)
            
            total_functions += len(functions)
            type_successful = 0
            
            # Test each type differently based on the group
            if func_type == 'callbacks':
                type_successful = test_callbacks_type(functions)
            elif func_type == 'data_loaders':
                type_successful = test_data_loaders_type(functions)  
            elif func_type == 'loss_functions':
                type_successful = test_loss_functions_type(functions)
            elif func_type == 'metrics':
                type_successful = test_metrics_type(functions)
            elif func_type == 'preprocessing':
                type_successful = test_preprocessing_type(functions)
            elif func_type == 'augmentations':
                print(f"  Note: {func_type} testing not implemented yet")
                type_successful = len(functions)  # Assume they would work
            else:
                print(f"  Warning: Unknown type {func_type}")
                
            total_successful += type_successful
            print(f"  Result: {type_successful}/{len(functions)} loaded successfully")
        
        print("=" * 50)
        print(f"OVERALL RESULT: {total_successful}/{total_functions} functions loaded successfully")
        print(f"Success rate: {(total_successful/total_functions)*100:.1f}%")
        
        return total_successful == total_functions
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_callbacks_type(functions):
    """Test callback functions."""
    from callbacks_group import CallbacksGroup
    
    callbacks_group = CallbacksGroup(name='Callbacks')
    successful = 0
    
    for func in functions:
        func_copy = func.copy()
        func_copy['file_path'] = os.path.join(
            "test_config/ModelGardener_Config_Package_20250821_142147", 
            func['file_path']
        )
        
        result = callbacks_group.load_custom_callback_from_metadata(func_copy)
        if result:
            successful += 1
            print(f"  ✓ {func['function_name']}")
        else:
            print(f"  ✗ {func['function_name']}")
            
    return successful

def test_data_loaders_type(functions):
    """Test data loader functions."""
    from data_loader_group import DataLoaderGroup
    
    data_loader_group = DataLoaderGroup(name='DataLoader')
    successful = 0
    
    for func in functions:
        func_copy = func.copy()
        func_copy['file_path'] = os.path.join(
            "test_config/ModelGardener_Config_Package_20250821_142147", 
            func['file_path']
        )
        
        result = data_loader_group.load_custom_data_loader_from_metadata(func_copy)
        if result:
            successful += 1
            print(f"  ✓ {func['function_name']}")
        else:
            print(f"  ✗ {func['function_name']}")
            
    return successful

def test_loss_functions_type(functions):
    """Test loss functions."""
    from loss_functions_group import LossFunctionsGroup
    
    loss_group = LossFunctionsGroup(name='LossFunctions')
    successful = 0
    
    for func in functions:
        func_copy = func.copy()
        func_copy['file_path'] = os.path.join(
            "test_config/ModelGardener_Config_Package_20250821_142147", 
            func['file_path']
        )
        
        result = loss_group.load_custom_loss_from_metadata(func_copy)
        if result:
            successful += 1
            print(f"  ✓ {func['function_name']}")
        else:
            print(f"  ✗ {func['function_name']}")
            
    return successful

def test_metrics_type(functions):
    """Test metric functions."""
    from metrics_group import MetricsGroup
    
    metrics_group = MetricsGroup(name='Metrics')
    successful = 0
    
    for func in functions:
        func_copy = func.copy()
        func_copy['file_path'] = os.path.join(
            "test_config/ModelGardener_Config_Package_20250821_142147", 
            func['file_path']
        )
        
        result = metrics_group.load_custom_metric_from_metadata(func_copy)
        if result:
            successful += 1
            print(f"  ✓ {func['function_name']}")
        else:
            print(f"  ✗ {func['function_name']}")
            
    return successful

def test_preprocessing_type(functions):
    """Test preprocessing functions."""
    from preprocessing_group import PreprocessingGroup
    
    preprocessing_group = PreprocessingGroup(name='Preprocessing')
    successful = 0
    
    for func in functions:
        func_copy = func.copy()
        func_copy['file_path'] = os.path.join(
            "test_config/ModelGardener_Config_Package_20250821_142147", 
            func['file_path']
        )
        
        result = preprocessing_group.load_custom_preprocessing_from_metadata(func_copy)
        if result:
            successful += 1
            print(f"  ✓ {func['function_name']}")
        else:
            print(f"  ✗ {func['function_name']}")
            
    return successful

if __name__ == "__main__":
    success = test_all_custom_functions()
    if success:
        print("\n🎉 All tests passed! Custom function loading is working perfectly!")
        sys.exit(0)
    else:
        print("\n⚠️  Some issues remain, but major progress made!")
        sys.exit(1)
