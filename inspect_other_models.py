#!/usr/bin/env python3
"""
Inspect detection and segmentation models from various sources.
"""

import tensorflow as tf
import sys

def inspect_detection_models():
    """Inspect object detection models."""
    print("=== Object Detection Models ===")
    
    # TensorFlow Model Garden / TensorFlow Hub common parameters
    detection_configs = {
        'yolo': {
            'common_params': [
                'input_size', 'num_classes', 'anchors', 'anchor_masks',
                'iou_threshold', 'confidence_threshold', 'max_boxes_per_class',
                'max_total_size', 'use_mixed_precision'
            ],
            'yolo_specific': {
                'YOLOv3': {'default_input_size': 416, 'num_anchors': 9},
                'YOLOv4': {'default_input_size': 512, 'num_anchors': 9, 'use_csp': True},
                'YOLOv5': {'default_input_size': 640, 'focus_layer': True},
                'YOLOv8': {'default_input_size': 640, 'use_ultralytics_format': True}
            }
        },
        'faster_rcnn': {
            'backbone_options': ['resnet50', 'resnet101', 'mobilenet_v2'],
            'common_params': [
                'num_classes', 'backbone', 'backbone_checkpoint',
                'include_mask', 'image_size', 'anchor_scale',
                'aspect_ratios', 'scales_per_octave', 'num_scales'
            ]
        },
        'ssd': {
            'backbone_options': ['mobilenet_v1', 'mobilenet_v2', 'resnet50'],
            'common_params': [
                'num_classes', 'backbone', 'image_size',
                'aspect_ratios', 'scales', 'clip_boxes'
            ]
        },
        'retinanet': {
            'common_params': [
                'num_classes', 'backbone', 'image_size',
                'anchor_scale', 'aspect_ratios', 'num_scales',
                'focal_loss_alpha', 'focal_loss_gamma'
            ]
        },
        'efficientdet': {
            'variants': ['D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7'],
            'common_params': [
                'num_classes', 'image_size', 'backbone_checkpoint',
                'mixed_precision', 'model_name'
            ]
        }
    }
    
    for model_family, config in detection_configs.items():
        print(f"\n{model_family.upper()}:")
        if 'common_params' in config:
            print(f"  Common parameters: {', '.join(config['common_params'])}")
        if 'backbone_options' in config:
            print(f"  Backbone options: {', '.join(config['backbone_options'])}")
        if 'variants' in config:
            print(f"  Variants: {', '.join(config['variants'])}")
    
    return detection_configs

def inspect_segmentation_models():
    """Inspect semantic segmentation models."""
    print("\n=== Semantic Segmentation Models ===")
    
    segmentation_configs = {
        'unet': {
            'variants': ['U-Net', 'U-Net++', 'U-Net-3+', 'Attention-U-Net'],
            'common_params': [
                'input_shape', 'num_classes', 'filters', 'num_layers',
                'dropout_rate', 'batch_normalization', 'activation',
                'kernel_initializer', 'use_attention', 'deep_supervision'
            ]
        },
        'deeplabv3': {
            'variants': ['DeepLabV3', 'DeepLabV3+'],
            'backbone_options': ['resnet50', 'resnet101', 'mobilenet_v2', 'xception'],
            'common_params': [
                'num_classes', 'backbone', 'output_stride', 'atrous_rates',
                'aspp_dropout', 'decoder_channels', 'decoder_dropout',
                'upsampling', 'activation'
            ]
        },
        'pspnet': {
            'backbone_options': ['resnet50', 'resnet101', 'mobilenet_v2'],
            'common_params': [
                'num_classes', 'backbone', 'pyramid_bins',
                'dropout_rate', 'activation', 'aux_loss'
            ]
        },
        'fcn': {
            'variants': ['FCN-8s', 'FCN-16s', 'FCN-32s'],
            'backbone_options': ['vgg16', 'resnet50', 'resnet101'],
            'common_params': [
                'num_classes', 'backbone', 'skip_connections',
                'dropout_rate', 'activation'
            ]
        },
        'segnet': {
            'backbone_options': ['vgg16', 'resnet'],
            'common_params': [
                'num_classes', 'backbone', 'encoder_weights',
                'decoder_use_batchnorm', 'activation'
            ]
        }
    }
    
    for model_family, config in segmentation_configs.items():
        print(f"\n{model_family.upper()}:")
        if 'common_params' in config:
            print(f"  Common parameters: {', '.join(config['common_params'])}")
        if 'backbone_options' in config:
            print(f"  Backbone options: {', '.join(config['backbone_options'])}")
        if 'variants' in config:
            print(f"  Variants: {', '.join(config['variants'])}")
    
    return segmentation_configs

def inspect_advanced_classification_models():
    """Inspect advanced classification models like Vision Transformers."""
    print("\n=== Advanced Classification Models ===")
    
    advanced_configs = {
        'vision_transformer': {
            'variants': ['ViT-Base-16', 'ViT-Large-16', 'ViT-Huge-14'],
            'common_params': [
                'image_size', 'patch_size', 'num_layers', 'num_heads',
                'hidden_size', 'mlp_dim', 'dropout_rate', 'attention_dropout_rate',
                'num_classes', 'representation_size', 'classifier_activation'
            ]
        },
        'convnext': {
            'variants': ['ConvNeXt-T', 'ConvNeXt-S', 'ConvNeXt-B', 'ConvNeXt-L'],
            'common_params': [
                'input_shape', 'include_top', 'weights', 'classes',
                'classifier_activation', 'drop_path_rate'
            ]
        },
        'regnet': {
            'variants': ['RegNetX', 'RegNetY'],
            'common_params': [
                'input_shape', 'include_top', 'weights', 'classes',
                'width_coefficient', 'depth_coefficient'
            ]
        }
    }
    
    for model_family, config in advanced_configs.items():
        print(f"\n{model_family.upper()}:")
        if 'common_params' in config:
            print(f"  Common parameters: {', '.join(config['common_params'])}")
        if 'variants' in config:
            print(f"  Variants: {', '.join(config['variants'])}")
    
    return advanced_configs

if __name__ == "__main__":
    print("🔍 Analyzing Detection and Segmentation Models")
    print("="*50)
    
    detection_configs = inspect_detection_models()
    segmentation_configs = inspect_segmentation_models()  
    advanced_configs = inspect_advanced_classification_models()
    
    print(f"\n✅ Model analysis complete!")
    print(f"📊 Detection families: {len(detection_configs)}")
    print(f"📊 Segmentation families: {len(segmentation_configs)}")
    print(f"📊 Advanced classification families: {len(advanced_configs)}")
