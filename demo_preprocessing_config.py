#!/usr/bin/env python3
"""
Demo script to showcase the new preprocessing configuration functionality
without requiring interactive terminal input.
"""

import sys
import os
import json

# Add the ModelGardener directory to Python path
sys.path.insert(0, '/mnt/sda1/WorkSpace/ModelGardener')

from cli_config import ModelConfigCLI

def demo_preprocessing_analysis():
    """Demonstrate custom preprocessing function analysis."""
    print("🌱 ModelGardener Preprocessing Configuration Demo")
    print("=" * 55)
    
    # Create instance of the CLI config
    config_cli = ModelConfigCLI()
    
    # Demonstrate custom preprocessing analysis
    print("\n🔍 Custom Preprocessing Function Analysis")
    print("-" * 45)
    
    example_file = "./example_funcs/example_custom_preprocessing.py"
    if os.path.exists(example_file):
        success, analysis = config_cli.analyze_custom_preprocessing_file(example_file)
        
        if success:
            print(f"✅ Successfully analyzed: {example_file}")
            print(f"📊 Found {len(analysis)} preprocessing functions:")
            
            for func_name, func_info in analysis.items():
                print(f"\n  📝 Function: {func_name}")
                print(f"     Description: {func_info.get('description', 'No description')}")
                print(f"     Parameters: {list(func_info.get('parameters', {}).keys())}")
                
                # Show parameter details
                params = func_info.get('parameters', {})
                for param_name, param_info in params.items():
                    param_type = param_info.get('type', 'str')
                    default_val = param_info.get('default', 'None')
                    print(f"       • {param_name} ({param_type}): default={default_val}")
        else:
            print("❌ No preprocessing functions found")
    else:
        print(f"❌ File not found: {example_file}")
    
    # Demonstrate default preprocessing configuration structure
    print("\n🔧 Default Preprocessing Configuration")
    print("-" * 40)
    
    # Create a mock configuration
    mock_config = {}
    
    # Show what the configure_preprocessing method would return
    # (without interactive input)
    default_preprocessing = {
        "Resizing": {
            "enabled": False,
            "target_size": {"width": 224, "height": 224, "depth": 1},  # depth=1 for 2D data
            "interpolation": "bilinear",
            "preserve_aspect_ratio": True,
            "data_format": "2D"
        },
        "Normalization": {
            "enabled": True,
            "method": "zero-center",
            "min_value": 0.0,
            "max_value": 1.0,
            "mean": {"r": 0.485, "g": 0.456, "b": 0.406},
            "std": {"r": 0.229, "g": 0.224, "b": 0.225},
            "axis": -1,
            "epsilon": 1e-07
        }
    }
    
    print("📋 Structure that would be created:")
    print(json.dumps(default_preprocessing, indent=2))
    
    # Demonstrate the preprocessing configuration flow
    print("\n🛠️  Preprocessing Configuration Steps")
    print("-" * 38)
    
    print("Step 1: Resizing Strategy Selection")
    print("  • Options: None, scaling, crop-padding")
    print("  • Methods for scaling: nearest, bilinear, bicubic, area, lanczos")
    print("  • Methods for crop-padding: central_cropping, random_cropping")
    print("  • Data format: 2D (images) or 3D (volumes/sequences)")
    print("  • Target size: width, height, depth (for 3D data only)")
    
    print("\nStep 2: Normalization Configuration")
    print("  • Methods: zero-center, min-max, unit-norm, standard, robust")
    print("  • Parameters: mean/std for zero-center, min/max for min-max")
    print("  • Common: axis, epsilon")
    
    print("\nStep 3: Custom Preprocessing")
    print("  • Load from Python files")
    print("  • Automatic function detection")
    print("  • Parameter configuration")
    print("  • Multiple function support")
    
    print("\n✅ Preprocessing configuration is ready for interactive use!")
    print("\n💡 To test interactively, run:")
    print("   python modelgardener_cli.py create test_project -i")
    print("   (then follow the prompts including the new preprocessing section)")

if __name__ == "__main__":
    demo_preprocessing_analysis()
