#!/usr/bin/env python3
"""
Inspect keras.applications models to extract their actual parameters.
"""

import sys
import os
import inspect
import tensorflow as tf
from tensorflow import keras

def inspect_keras_applications():
    """Inspect keras.applications models to understand their parameters."""
    print("Inspecting keras.applications models...")
    
    # Common models in keras.applications
    models_to_inspect = {
        'ResNet': [
            keras.applications.ResNet50,
            keras.applications.ResNet101,
            keras.applications.ResNet152,
        ],
        'EfficientNet': [
            keras.applications.EfficientNetB0,
            keras.applications.EfficientNetB1,
            keras.applications.EfficientNetB3,
            keras.applications.EfficientNetB7,
        ],
        'MobileNet': [
            keras.applications.MobileNet,
            keras.applications.MobileNetV2,
            keras.applications.MobileNetV3Small,
            keras.applications.MobileNetV3Large,
        ],
        'DenseNet': [
            keras.applications.DenseNet121,
            keras.applications.DenseNet169,
            keras.applications.DenseNet201,
        ],
        'VGG': [
            keras.applications.VGG16,
            keras.applications.VGG19,
        ],
        'Inception': [
            keras.applications.InceptionV3,
            keras.applications.InceptionResNetV2,
        ],
        'Xception': [
            keras.applications.Xception,
        ]
    }
    
    model_signatures = {}
    
    for family_name, models in models_to_inspect.items():
        print(f"\n=== {family_name} Family ===")
        family_signatures = {}
        
        for model_class in models:
            try:
                model_name = model_class.__name__
                signature = inspect.signature(model_class)
                
                print(f"{model_name}:")
                parameters = {}
                
                for param_name, param in signature.parameters.items():
                    param_info = {
                        'type': str(param.annotation) if param.annotation != param.empty else 'Any',
                        'default': param.default if param.default != param.empty else None,
                        'required': param.default == param.empty
                    }
                    parameters[param_name] = param_info
                    
                    # Print parameter info
                    default_str = f" = {param.default}" if param.default != param.empty else " (required)"
                    annotation_str = f": {param.annotation}" if param.annotation != param.empty else ""
                    print(f"  - {param_name}{annotation_str}{default_str}")
                
                family_signatures[model_name] = parameters
                
            except Exception as e:
                print(f"  Error inspecting {model_class.__name__}: {e}")
        
        model_signatures[family_name] = family_signatures
    
    return model_signatures

def analyze_common_parameters(signatures):
    """Analyze common parameters across models."""
    print("\n" + "="*60)
    print("COMMON PARAMETERS ANALYSIS")
    print("="*60)
    
    all_params = {}
    param_frequency = {}
    
    for family, models in signatures.items():
        for model, params in models.items():
            for param_name, param_info in params.items():
                if param_name not in all_params:
                    all_params[param_name] = []
                    param_frequency[param_name] = 0
                
                all_params[param_name].append({
                    'family': family,
                    'model': model,
                    'info': param_info
                })
                param_frequency[param_name] += 1
    
    # Sort by frequency
    sorted_params = sorted(param_frequency.items(), key=lambda x: x[1], reverse=True)
    
    print(f"Most common parameters (found in multiple models):")
    for param_name, frequency in sorted_params[:15]:
        if frequency > 1:
            print(f"  {param_name}: used in {frequency} models")
            
            # Show different default values
            defaults = set()
            for usage in all_params[param_name]:
                default = usage['info']['default']
                if default is not None:
                    defaults.add(str(default))
            
            if defaults:
                print(f"    Common defaults: {', '.join(sorted(defaults))}")

def generate_refined_parameters():
    """Generate refined parameter configurations based on keras.applications."""
    print("\n" + "="*60)  
    print("REFINED PARAMETER CONFIGURATIONS")
    print("="*60)
    
    # Based on actual keras.applications signatures
    refined_configs = {
        'image_classification': {
            'resnet': {
                'common_params': {
                    'input_shape': {
                        'type': 'group',
                        'children': [
                            {'name': 'height', 'type': 'int', 'value': 224, 'limits': [32, 1024]},
                            {'name': 'width', 'type': 'int', 'value': 224, 'limits': [32, 1024]},
                            {'name': 'channels', 'type': 'int', 'value': 3, 'limits': [1, 4]}
                        ]
                    },
                    'include_top': {'type': 'bool', 'value': True},
                    'weights': {
                        'type': 'list', 
                        'value': 'imagenet',
                        'limits': ['imagenet', 'None', 'ssl', 'swsl']
                    },
                    'input_tensor': {'type': 'str', 'value': '', 'readonly': True},
                    'pooling': {
                        'type': 'list',
                        'value': None,
                        'limits': [None, 'avg', 'max']
                    },
                    'classes': {'type': 'int', 'value': 1000, 'limits': [1, 100000]},
                    'classifier_activation': {
                        'type': 'list',
                        'value': 'softmax',
                        'limits': ['softmax', 'sigmoid', 'linear', None]
                    }
                }
            },
            'efficientnet': {
                'common_params': {
                    'include_top': {'type': 'bool', 'value': True},
                    'weights': {
                        'type': 'list',
                        'value': 'imagenet', 
                        'limits': ['imagenet', 'noisy-student', None]
                    },
                    'input_tensor': {'type': 'str', 'value': ''},
                    'input_shape': {
                        'type': 'group',
                        'children': [
                            {'name': 'height', 'type': 'int', 'value': 224, 'limits': [32, 1024]},
                            {'name': 'width', 'type': 'int', 'value': 224, 'limits': [32, 1024]},
                            {'name': 'channels', 'type': 'int', 'value': 3, 'limits': [1, 4]}
                        ]
                    },
                    'pooling': {
                        'type': 'list',
                        'value': None,
                        'limits': [None, 'avg', 'max']
                    },
                    'classes': {'type': 'int', 'value': 1000, 'limits': [1, 100000]},
                    'classifier_activation': {
                        'type': 'list',
                        'value': 'softmax',
                        'limits': ['softmax', 'sigmoid', None]
                    },
                    'drop_connect_rate': {
                        'type': 'float',
                        'value': 0.2,
                        'limits': [0.0, 0.8],
                        'step': 0.05
                    }
                }
            },
            'mobilenet': {
                'v1_params': {
                    'input_shape': {
                        'type': 'group', 
                        'children': [
                            {'name': 'height', 'type': 'int', 'value': 224, 'limits': [32, 1024]},
                            {'name': 'width', 'type': 'int', 'value': 224, 'limits': [32, 1024]},
                            {'name': 'channels', 'type': 'int', 'value': 3, 'limits': [1, 4]}
                        ]
                    },
                    'alpha': {
                        'type': 'float',
                        'value': 1.0,
                        'limits': [0.25, 1.0, 1.3, 1.4],
                        'step': 0.05
                    },
                    'depth_multiplier': {
                        'type': 'int',
                        'value': 1,
                        'limits': [1, 2, 3, 4]
                    },
                    'dropout': {
                        'type': 'float',
                        'value': 0.001,
                        'limits': [0.0, 0.9],
                        'step': 0.001
                    },
                    'include_top': {'type': 'bool', 'value': True},
                    'weights': {
                        'type': 'list',
                        'value': 'imagenet',
                        'limits': ['imagenet', None]
                    },
                    'pooling': {
                        'type': 'list',
                        'value': None,
                        'limits': [None, 'avg', 'max']
                    },
                    'classes': {'type': 'int', 'value': 1000, 'limits': [1, 100000]}
                },
                'v2_params': {
                    'alpha': {
                        'type': 'float', 
                        'value': 1.0,
                        'limits': [0.35, 0.5, 0.75, 1.0, 1.3, 1.4]
                    }
                },
                'v3_params': {
                    'alpha': {
                        'type': 'float',
                        'value': 1.0, 
                        'limits': [0.75, 1.0]
                    },
                    'minimalistic': {'type': 'bool', 'value': False}
                }
            }
        }
    }
    
    return refined_configs

if __name__ == "__main__":
    print("🔍 Analyzing keras.applications Models")
    print("="*50)
    
    try:
        # Inspect actual model signatures
        signatures = inspect_keras_applications()
        
        # Analyze common parameters
        analyze_common_parameters(signatures)
        
        # Generate refined configurations
        refined = generate_refined_parameters()
        
        print(f"\n✅ Analysis complete!")
        print(f"📊 Found signatures for {sum(len(models) for models in signatures.values())} models")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
