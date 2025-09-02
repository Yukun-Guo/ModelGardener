# CIFAR-10 NPZ Dataset Integration

This document describes the integration of the CIFAR-10 NPZ dataset into ModelGardener, replacing the previous sample_xx.jpg image examples.

## Overview

The ModelGardener framework now uses a real CIFAR-10 subset dataset instead of synthetic sample images. This provides:

- **Real image data**: 1000 actual CIFAR-10 images (100 per class, 10 classes)
- **Proper train/validation split**: Automatic 80/20 split with stratification
- **Efficient loading**: NPZ format for fast loading and preprocessing
- **One-hot encoded labels**: Ready for categorical classification
- **Normalized data**: Images automatically normalized to [0, 1] range

## Dataset Details

- **File**: `example_data/cifar10.npz`
- **Total samples**: 1000 images
- **Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Image size**: 32x32x3 (RGB)
- **Split**: 800 training + 200 validation samples
- **Format**: NumPy arrays in NPZ format

## Files Changed/Added

### 1. Dataset Generation
- `test_generate_subset.py` - Generates the CIFAR-10 subset from the full dataset

### 2. Custom Data Loaders
- `cifar10_npz_data_loader.py` - Standalone CIFAR-10 data loader (reference implementation)
- `example_funcs/example_custom_data_loaders.py` - Updated with CIFAR-10 loaders:
  - `load_cifar10_npz_data()` - Function-based loader
  - `CIFAR10NPZDataLoader` - Class-based loader  
  - `simple_cifar10_loader()` - Simplified interface

### 3. Configuration Files
- `cifar10_config.json` - Complete configuration for CIFAR-10 training
  - Uses custom data loader
  - Optimized for 32x32 input size
  - 10 classes output
  - Appropriate augmentation settings

### 4. Test Scripts
- `test_cifar10_loader.py` - Tests the standalone data loader
- `test_updated_loaders.py` - Tests the integrated loaders
- `test_cifar10_integration.py` - Complete integration test

## Usage

### 1. Generate the Dataset

```bash
python test_generate_subset.py
```

This creates `example_data/cifar10.npz` with the CIFAR-10 subset.

### 2. Use with Custom Data Loaders

#### Function-based approach:
```python
from example_funcs.example_custom_data_loaders import load_cifar10_npz_data

# Training data
train_ds = load_cifar10_npz_data(
    data_dir="example_data",
    batch_size=32,
    split='train'
)

# Validation data
val_ds = load_cifar10_npz_data(
    data_dir="example_data",
    batch_size=32,
    split='val'
)
```

#### Class-based approach:
```python
from example_funcs.example_custom_data_loaders import CIFAR10NPZDataLoader

loader = CIFAR10NPZDataLoader(
    data_dir="example_data",
    batch_size=32,
    validation_split=0.2
)

train_ds = loader.get_dataset('train')
val_ds = loader.get_dataset('val')
```

### 3. Use with ModelGardener GUI

1. Load the `cifar10_config.json` configuration file
2. The custom data loader will automatically load the CIFAR-10 dataset
3. Train your model with real image data

## Configuration Details

The `cifar10_config.json` file includes:

- **Custom data loader**: `Custom_load_cifar10_npz_data`
- **Input shape**: 32x32x3 (matching CIFAR-10)
- **Output classes**: 10 (for CIFAR-10 classes)
- **Data augmentation**: Horizontal flip, rotation, brightness adjustment
- **Optimized settings**: For small 32x32 images

## Benefits of the New Dataset

1. **Real data**: Train on actual images instead of synthetic examples
2. **Proper evaluation**: Realistic performance metrics
3. **Standard benchmark**: CIFAR-10 is a well-known computer vision benchmark
4. **Efficient**: NPZ format loads faster than individual image files
5. **Balanced**: Equal samples per class (100 each)
6. **Pre-processed**: Normalized and one-hot encoded
7. **Split ready**: Automatic train/validation splitting

## Migration Notes

### From sample_xx.jpg to CIFAR-10 NPZ:

1. **Data format**: Changed from individual JPG files to NPZ format
2. **Image size**: Changed from 224x224 to 32x32 (CIFAR-10 native size)
3. **Classes**: Fixed to 10 classes (CIFAR-10 classes)
4. **Loading method**: Custom data loader instead of directory-based loading
5. **Preprocessing**: Built into the data loader

### Configuration updates:
- Model input shape: 224x224x3 → 32x32x3
- Number of classes: Variable → 10
- Data loader: Default → Custom_load_cifar10_npz_data
- Preprocessing: Disabled (handled by loader)

## Testing

Run the integration test to verify everything works:

```bash
python test_cifar10_integration.py
```

This will test:
- Dataset loading
- Custom data loaders
- Model compatibility
- Configuration file
- Performance benchmarks

## Troubleshooting

### Common Issues:

1. **Dataset not found**: Run `test_generate_subset.py` first
2. **Import errors**: Ensure you're in the ModelGardener directory
3. **Shape mismatches**: Check model input shape is (32, 32, 3)
4. **Memory issues**: Reduce batch size if needed

### Verify Installation:
```bash
# Check dataset exists
ls -la example_data/cifar10.npz

# Test loaders
python test_cifar10_integration.py

# Verify configuration
cat cifar10_config.json | grep -A5 "data_loader"
```

## Next Steps

With the CIFAR-10 NPZ dataset integrated, you can:

1. Train real image classification models
2. Experiment with different architectures optimized for 32x32 images
3. Compare performance with standard CIFAR-10 benchmarks
4. Use the custom data loader pattern for other datasets

The framework is now ready for serious computer vision experiments with real image data!
