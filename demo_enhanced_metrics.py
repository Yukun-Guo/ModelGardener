#!/usr/bin/env python3
"""
Demonstration script showing the enhanced metrics configuration workflow.
This shows the step-by-step process similar to loss functions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cli_config import ModelConfigCLI
import json

def demo_enhanced_metrics_workflow():
    """Demonstrate the enhanced metrics configuration workflow."""
    print("🚀 Enhanced Metrics Configuration Workflow Demo")
    print("=" * 55)
    
    # Create CLI instance
    cli = ModelConfigCLI()
    
    # Example 1: Single Output Model Configuration
    print("\n📋 Example 1: Single Output Model (ResNet-50)")
    print("-" * 45)
    
    # Simulate single output configuration
    single_output_config = {
        'configuration': {
            'model': {
                'model_family': 'keras_applications',
                'model_name': 'ResNet50'
            }
        }
    }
    
    # Simulate loss functions config for single output
    loss_functions_config_single = {
        'Model Output Configuration': {
            'num_outputs': 1,
            'output_names': 'main_output',
            'loss_strategy': 'single_loss_all_outputs'
        },
        'Loss Selection': {
            'selected_loss': 'Categorical Crossentropy'
        }
    }
    
    print("🔍 Step 1: Reusing model output analysis from loss functions...")
    print(f"   ✅ Detected {loss_functions_config_single['Model Output Configuration']['num_outputs']} output")
    print(f"   ✅ Output names: {loss_functions_config_single['Model Output Configuration']['output_names']}")
    
    print("\n📝 Step 2: Model Output Information")
    print(f"   Detected outputs: {loss_functions_config_single['Model Output Configuration']['num_outputs']}")
    print(f"   Output names: [{loss_functions_config_single['Model Output Configuration']['output_names']}]")
    
    print("\n⚙️  Step 3: Metrics Strategy Selection")
    print("   Single output detected - using 'shared_metrics_all_outputs' strategy")
    
    print("\n🎯 Step 4: Metrics Configuration")
    print("   Strategy: shared_metrics_all_outputs")
    print("   [User would select metrics: Accuracy, Precision, Recall]")
    
    # Expected output for single model
    expected_single_metrics = {
        'Model Output Configuration': {
            'num_outputs': 1,
            'output_names': 'main_output',
            'metrics_strategy': 'shared_metrics_all_outputs'
        },
        'Metrics Selection': {
            'selected_metrics': 'Accuracy,Precision,Recall',
            'custom_metrics_configs': {}
        }
    }
    
    print("\n📄 Expected Configuration Output:")
    print(json.dumps(expected_single_metrics, indent=2))
    
    # Example 2: Multi-Output Model Configuration
    print("\n\n📋 Example 2: Multi-Output Custom Model")
    print("-" * 45)
    
    # Simulate multiple output configuration
    multiple_output_config = {
        'configuration': {
            'model': {
                'model_family': 'custom',
                'model_name': 'CustomMultiOutputModel'
            }
        }
    }
    
    # Simulate loss functions config for multiple outputs
    loss_functions_config_multi = {
        'Model Output Configuration': {
            'num_outputs': 2,
            'output_names': 'main_output,aux_output_1',
            'loss_strategy': 'different_loss_each_output'
        },
        'Loss Selection': {
            'main_output': {
                'selected_loss': 'Categorical Crossentropy'
            },
            'aux_output_1': {
                'selected_loss': 'Binary Crossentropy'
            }
        }
    }
    
    print("🔍 Step 1: Reusing model output analysis from loss functions...")
    print(f"   ✅ Detected {loss_functions_config_multi['Model Output Configuration']['num_outputs']} outputs")
    print(f"   ✅ Output names: {loss_functions_config_multi['Model Output Configuration']['output_names']}")
    
    print("\n📝 Step 2: Model Output Information") 
    print(f"   Detected outputs: {loss_functions_config_multi['Model Output Configuration']['num_outputs']}")
    output_names = loss_functions_config_multi['Model Output Configuration']['output_names'].split(',')
    print(f"   Output names: {output_names}")
    
    print("\n⚙️  Step 3: Metrics Strategy Selection")
    print("   Multiple outputs detected (2) - please select strategy:")
    print("   [User chooses between shared or different metrics strategies]")
    print("   Selected: different_metrics_per_output")
    
    print("\n🎯 Step 4: Metrics Configuration")
    print("   Strategy: different_metrics_per_output")
    print("   Configuring metrics for 'main_output': [User selects: Accuracy, Top K Categorical Accuracy]")
    print("   Configuring metrics for 'aux_output_1': [User selects: AUC, Precision, Recall]")
    
    # Expected output for multi-output model
    expected_multi_metrics = {
        'Model Output Configuration': {
            'num_outputs': 2,
            'output_names': 'main_output,aux_output_1',
            'metrics_strategy': 'different_metrics_per_output'
        },
        'Metrics Selection': {
            'main_output': {
                'selected_metrics': 'Accuracy,Top K Categorical Accuracy',
                'custom_metrics_configs': {}
            },
            'aux_output_1': {
                'selected_metrics': 'AUC,Precision,Recall',
                'custom_metrics_configs': {}
            }
        }
    }
    
    print("\n📄 Expected Configuration Output:")
    print(json.dumps(expected_multi_metrics, indent=2))
    
    # Example 3: Custom Metrics Integration
    print("\n\n📋 Example 3: Custom Metrics Integration")
    print("-" * 45)
    
    print("🔧 Custom Metrics Workflow:")
    print("   1. User selects 'Load Custom Metrics Functions'")
    print("   2. System prompts for Python file path")
    print("   3. File analysis finds custom metrics functions")
    print("   4. User selects specific custom metrics to load")
    print("   5. Custom metrics appear in subsequent output selections")
    
    # Check if example custom metrics exist
    custom_metrics_path = "./example_funcs/example_custom_metrics.py"
    if os.path.exists(custom_metrics_path):
        print(f"\n📁 Analyzing: {custom_metrics_path}")
        try:
            success, metrics_info = cli.analyze_custom_metrics_file(custom_metrics_path)
            if success:
                print("   ✅ Custom metrics found:")
                for name, info in metrics_info.items():
                    print(f"      - {name} ({info['type']}): {info.get('signature', 'N/A')}")
                    
                # Expected output with custom metrics
                expected_custom_metrics = {
                    'Model Output Configuration': {
                        'num_outputs': 1,
                        'output_names': 'main_output',
                        'metrics_strategy': 'shared_metrics_all_outputs'
                    },
                    'Metrics Selection': {
                        'selected_metrics': 'Accuracy,balanced_accuracy',
                        'custom_metrics_configs': {
                            'balanced_accuracy': {
                                'custom_metrics_path': custom_metrics_path,
                                'parameters': {
                                    'threshold': 0.5
                                }
                            }
                        }
                    }
                }
                
                print("\n📄 Example Configuration with Custom Metrics:")
                print(json.dumps(expected_custom_metrics, indent=2))
            else:
                print("   ⚠️  No valid metrics found in example file")
        except Exception as e:
            print(f"   ❌ Error analyzing custom metrics: {e}")
    else:
        print(f"   ⚠️  Example file not found: {custom_metrics_path}")
    
    # Summary of improvements
    print("\n\n🎯 Key Improvements Summary")
    print("-" * 30)
    print("✅ Automatic model output analysis reuse from loss functions")
    print("✅ Intelligent metrics strategy selection for multi-output models")
    print("✅ Custom metrics functions support with analysis and loading")
    print("✅ Per-output metrics configuration for complex models")
    print("✅ Consistent workflow with loss functions configuration")
    print("✅ Comprehensive parameter handling for custom metrics")
    
    print("\n📊 Configuration Structure Consistency")
    print("-" * 40)
    print("Loss Functions Config:")
    print("  Model Output Configuration -> num_outputs, output_names, loss_strategy")
    print("  Loss Selection -> per-output or shared loss configurations")
    print("\nMetrics Config (NEW):")
    print("  Model Output Configuration -> num_outputs, output_names, metrics_strategy") 
    print("  Metrics Selection -> per-output or shared metrics configurations")
    print("\n🔄 Both follow the exact same pattern for consistency!")
    
    print("\n🚀 Enhanced Metrics Configuration Demo Complete!")
    print("=" * 55)

if __name__ == "__main__":
    demo_enhanced_metrics_workflow()
