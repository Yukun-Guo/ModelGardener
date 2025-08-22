# Parameter Limits Fix Summary

## Problem Description

The user encountered a "ValueError: too many values to unpack (expected 2)" error when switching to ViT-Base-16 model in the ModelGardener application. This error occurred in the PyQt parameter tree widget creation system.

## Root Cause

The error was caused by parameter definitions in `model_group.py` that used `'limits'` arrays with more than 2 values for numeric parameter types (`'type': 'int'` or `'type': 'float'`). PyQt's parameter system expects exactly `[min, max]` format for numeric parameters with limits, but some parameters were defined with discrete value lists like `[8, 16, 32]`.

## Error Location

The error occurred in pyqtgraph's parameter creation code when it tried to unpack limits arrays:
```python
defs['min'], defs['max'] = opts['limits']  # Fails when limits has >2 values
```

## Parameters Fixed

### Vision Transformer (ViT) Parameters
- `patch_size`: `'limits': [8, 16, 32]` → `'values': [8, 16, 32]` + `'type': 'list'`
- `num_layers`: `'limits': [6, 12, 24]` → `'values': [6, 12, 24]` + `'type': 'list'`
- `num_heads`: `'limits': [4, 8, 12, 16]` → `'values': [4, 8, 12, 16]` + `'type': 'list'`
- `hidden_size`: `'limits': [256, 384, 768, 1024]` → `'values': [256, 384, 768, 1024]` + `'type': 'list'`
- `mlp_dim`: `'limits': [1024, 3072, 4096]` → `'values': [1024, 3072, 4096]` + `'type': 'list'`
- `representation_size`: `'limits': [0, 768, 1024]` → `'values': [0, 768, 1024]` + `'type': 'list'`

### MobileNet Parameters
- `alpha`: `'limits': [0.25, 0.35, 0.5, 0.75, 1.0, 1.3, 1.4]` → `'values': [0.25, 0.35, 0.5, 0.75, 1.0, 1.3, 1.4]` + `'type': 'list'`
- `depth_multiplier`: `'limits': [1, 2, 3, 4]` → `'values': [1, 2, 3, 4]` + `'type': 'list'`

### RegNet Parameters
- `width_coefficient`: `'limits': [0.5, 1.0, 1.5, 2.0]` → `'values': [0.5, 1.0, 1.5, 2.0]` + `'type': 'list'`
- `depth_coefficient`: `'limits': [0.5, 1.0, 1.5, 2.0]` → `'values': [0.5, 1.0, 1.5, 2.0]` + `'type': 'list'`

### Segmentation Model Parameters
- U-Net `filters`: `'limits': [16, 32, 64, 128, 256]` → `'values': [16, 32, 64, 128, 256]` + `'type': 'list'`
- U-Net `num_layers`: `'limits': [3, 4, 5, 6]` → `'values': [3, 4, 5, 6]` + `'type': 'list'`
- DeepLabV3 `output_stride`: `'limits': [8, 16, 32]` → `'values': [8, 16, 32]` + `'type': 'list'`
- DeepLabV3 `decoder_channels`: `'limits': [64, 128, 256, 512]` → `'values': [64, 128, 256, 512]` + `'type': 'list'`

### List Parameter Corrections
Also fixed list parameters that were incorrectly using `'limits'` instead of `'values'`:
- `weights`: `'limits': ['imagenet', 'None']` → `'values': ['imagenet', 'None']`
- `pooling`: `'limits': ['None', 'avg', 'max']` → `'values': ['None', 'avg', 'max']`  
- `classifier_activation`: `'limits': ['softmax', 'sigmoid', 'linear', 'None']` → `'values': ['softmax', 'sigmoid', 'linear', 'None']`
- Various backbone selection parameters

## Solution Summary

1. **Parameter Type Change**: Changed parameter type from `'int'`/`'float'` to `'list'` for discrete value parameters
2. **Property Name Change**: Changed `'limits'` to `'values'` for discrete value lists
3. **Preserved Range Parameters**: Kept `'limits': [min, max]` format for true range parameters (e.g., dropout rates)

## Verification

Created comprehensive test (`test_parameter_limits_simple.py`) that verified:
- ✅ All ViT parameters can be created without errors
- ✅ Other potentially problematic models (MobileNet, EfficientNet, U-Net, etc.) work correctly
- ✅ Parameter definitions use correct `'values'` vs `'limits'` format
- ✅ Parameter tree creation succeeds with 15 parameters for ViT-Base-16

## Files Modified

- `model_group.py`: Fixed parameter definitions in multiple methods:
  - `_get_vit_specific_params()`
  - `_get_mobilenet_specific_params()`
  - `_get_regnet_specific_params()`
  - `_get_unet_specific_params()`
  - `_get_deeplab_specific_params()`
  - `_get_pspnet_specific_params()`
  - `_get_fcn_segnet_specific_params()`
  - Common parameters in `_get_model_parameters()`

## Impact

This fix resolves the "too many values to unpack" error that was preventing users from:
- Selecting ViT (Vision Transformer) models
- Using models with discrete parameter choices
- Properly configuring advanced model parameters

The fix maintains all functionality while ensuring PyQt parameter tree compatibility.
