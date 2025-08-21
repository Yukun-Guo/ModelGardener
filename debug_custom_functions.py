#!/usr/bin/env python3
"""
Debug script to understand the custom functions loading issue.
"""

import json
import os
from pathlib import Path

def analyze_manifest_and_config():
    """Analyze the manifest and config files to understand the issue."""
    
    # Paths
    config_package_dir = "/mnt/sda1/WorkSpace/ModelGardener/test_config/ModelGardener_Config_Package_20250821_160526"
    manifest_path = os.path.join(config_package_dir, "custom_functions_manifest.json")
    model_config_path = os.path.join(config_package_dir, "model_config.json")
    
    print("=== ANALYZING CUSTOM FUNCTIONS LOADING ISSUE ===\n")
    
    # Read manifest
    print("1. Reading manifest file...")
    with open(manifest_path, 'r') as f:
        manifest_data = json.load(f)
    
    print(f"Manifest structure:")
    print(f"  - Version: {manifest_data.get('model_gardener_version')}")
    print(f"  - Creation date: {manifest_data.get('creation_date')}")
    print(f"  - Custom functions count: {len(manifest_data.get('custom_functions', []))}")
    
    # Analyze custom functions
    print(f"\n2. Analyzing custom functions in manifest:")
    custom_functions = manifest_data.get('custom_functions', [])
    
    data_loaders = []
    for func in custom_functions:
        print(f"  - Type: {func.get('type')}")
        print(f"    Name: {func.get('name')}")
        print(f"    Function name: '{func.get('function_name')}'")
        print(f"    File path: {func.get('file_path')}")
        print(f"    Empty function name: {not func.get('function_name')}")
        print()
        
        if func.get('type') == 'data_loaders':
            data_loaders.append(func)
    
    # Check for empty function names
    empty_function_names = [f for f in custom_functions if not f.get('function_name')]
    print(f"3. Functions with empty function names: {len(empty_function_names)}")
    for func in empty_function_names:
        print(f"  - {func.get('name')} in {func.get('file_path')}")
    
    # Read model config
    print(f"\n4. Reading model config file...")
    with open(model_config_path, 'r') as f:
        model_config = json.load(f)
    
    # Check how custom functions are supposed to be structured
    if 'metadata' in model_config:
        metadata = model_config['metadata']
        custom_functions_info = metadata.get('custom_functions', {})
        print(f"Model config has metadata with custom_functions_info structure:")
        for func_type, funcs in custom_functions_info.items():
            print(f"  - {func_type}: {len(funcs)} functions")
    else:
        print("Model config does not have metadata section")
    
    # Show expected vs actual structure
    print(f"\n5. Expected structure conversion:")
    print("From manifest format (flat list) to grouped format:")
    
    # Convert manifest to grouped format (like collect_custom_functions_info does)
    grouped_functions = {
        'data_loaders': [],
        'optimizers': [],
        'loss_functions': [],
        'metrics': [],
        'augmentations': [],
        'callbacks': [],
        'preprocessing': []
    }
    
    for func in custom_functions:
        func_type = func.get('type', '')
        if func_type in grouped_functions:
            # Check if function_name is empty
            function_name = func.get('function_name', '')
            if not function_name:
                print(f"  WARNING: Empty function_name for {func.get('name')} in {func_type}")
                continue
                
            grouped_functions[func_type].append({
                'name': func.get('name'),
                'function_name': function_name,
                'file_path': func.get('file_path'),
                'type': 'function'  # Default type
            })
    
    print(f"\nGrouped functions result:")
    for func_type, funcs in grouped_functions.items():
        if funcs:
            print(f"  - {func_type}: {len(funcs)} functions")
            for func in funcs:
                print(f"    * {func['name']} -> {func['function_name']}")

if __name__ == "__main__":
    analyze_manifest_and_config()
