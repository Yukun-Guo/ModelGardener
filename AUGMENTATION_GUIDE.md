# Augmentation Configuration Guide

## Overview
The refactored augmentation system provides a flexible, user-friendly interface for configuring data augmentation methods. It includes preset augmentation methods and allows adding custom augmentation configurations.

## Features

### Preset Augmentation Methods
The system includes the following preset augmentation methods, each with enable/disable checkboxes and relevant parameters:

1. **Horizontal Flip**
   - Enable/disable checkbox
   - Probability control (0.0-1.0)

2. **Vertical Flip**
   - Enable/disable checkbox
   - Probability control (0.0-1.0)

3. **Rotation**
   - Enable/disable checkbox
   - Angle range control (0-180 degrees)
   - Probability control (0.0-1.0)

4. **Gaussian Noise**
   - Enable/disable checkbox
   - Variance limit control
   - Probability control (0.0-1.0)

5. **Brightness Adjustment**
   - Enable/disable checkbox
   - Brightness limit control (±)
   - Probability control (0.0-1.0)

6. **Contrast Adjustment**
   - Enable/disable checkbox
   - Contrast limit control (±)
   - Probability control (0.0-1.0)

7. **Color Jittering**
   - Enable/disable checkbox
   - Hue shift limit (0-50)
   - Saturation shift limit (0-100)
   - Value shift limit (0-100)
   - Probability control (0.0-1.0)

8. **Random Cropping**
   - Enable/disable checkbox
   - Crop area range (min/max as fraction of original)
   - Aspect ratio range (min/max)
   - Probability control (0.0-1.0)

### Custom Augmentation Methods
At the end of the augmentation section, there's a button to "Add Custom Augmentation" with the following options:

- **Custom Rotation**: Specify exact min/max angle ranges
- **Custom Noise**: Choose noise type (gaussian, uniform, salt_pepper) with intensity control
- **Custom Blur**: Select blur type (gaussian, motion, median) with blur limit
- **Custom Distortion**: Apply distortion effects (elastic, perspective, barrel)
- **Custom Filter**: Apply filters (sharpen, emboss, edge_enhance)

## Implementation Details

### AugmentationGroup Class
The system is built using the `AugmentationGroup` class, which extends PyQtGraph's `GroupParameter` class, similar to the `ScalableGroup` from parametertree.py.

Key features:
- Preset methods are automatically added when the group is created
- Custom methods can be added via the "Add Custom Augmentation" button
- All custom methods are removable and renamable
- Each method has appropriate parameter limits and tooltips

### Configuration Structure
The configuration is stored as a nested dictionary:

```python
{
    'Horizontal Flip': {
        'enabled': True,
        'probability': 0.5
    },
    'Custom Rotation 1': {
        'enabled': True,
        'min_angle': -30.0,
        'max_angle': 30.0,
        'probability': 0.3
    }
}
```

### TensorFlow Integration
The system maps the new augmentation configuration to TensorFlow Models format:
- Horizontal Flip → `aug_rand_hflip`
- Random Cropping → `aug_crop` + `crop_area_range`
- Color Jittering → `color_jitter` (normalized hue shift)

### Albumentations Integration
For preview functionality, the system builds Albumentations pipelines from the configuration:
- Each enabled method is converted to the appropriate Albumentations transform
- Custom methods are handled with additional logic
- Maintains backward compatibility with legacy configurations

## Usage

1. **Enable/Disable Methods**: Use checkboxes to enable or disable specific augmentation methods

2. **Adjust Parameters**: Each method has relevant parameters with appropriate limits and tooltips

3. **Add Custom Methods**: Click "Add Custom Augmentation" to add specialized augmentation methods

4. **Remove Custom Methods**: Custom methods can be removed using the context menu

5. **Preview Results**: Use the "Preview Data" button to see how augmentations affect sample images

## Benefits

- **User-Friendly**: Clear organization with enable/disable controls
- **Flexible**: Supports both preset and custom augmentation methods
- **Extensible**: Easy to add new augmentation types
- **Backwards Compatible**: Works with existing configurations
- **Visual**: Real-time preview of augmentation effects
- **Well-Documented**: Comprehensive tooltips for all parameters

The new system provides much better control over data augmentation while maintaining the simplicity of the checkbox-based approach for common use cases.
