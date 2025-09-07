# `preview` Command

Preview data samples with preprocessing and augmentation visualization to understand your data pipeline.

## Synopsis

```bash
mg preview --config CONFIG [options]
```

## Description

The `preview` command allows you to visualize your data samples before training, showing both original and processed versions. This helps you understand how preprocessing and augmentation transforms affect your data, making it easier to debug and optimize your data pipeline.

## Arguments

**Required:**
- `--config`, `-c`: Configuration file path

## Options

### Data Options

| Option | Short | Type | Description | Default |
|--------|-------|------|-------------|---------|
| `--num-samples` | `-n` | `int` | Number of samples to preview | 8 |
| `--split` | `-s` | `str` | Data split to preview (`train`, `val`, `test`) | `train` |

### Output Options

| Option | Short | Type | Description | Default |
|--------|-------|------|-------------|---------|
| `--save` |  | `flag` | Save plot to file instead of displaying | Disabled |
| `--output` | `-o` | `str` | Output file path for saved plot | Auto-generated |

## Usage Examples

### Basic Preview

```bash
# Preview 8 training samples (default)
mg preview --config config.yaml

# Preview validation data
mg preview --config config.yaml --split val

# Preview more samples
mg preview --config config.yaml --num-samples 16
```

### Customized Preview

```bash
# Preview specific number of samples from test set
mg preview --config config.yaml --split test --num-samples 12

# Save preview to file
mg preview --config config.yaml --save --output my_data_preview.png

# Preview with custom output file
mg preview --config config.yaml --num-samples 20 --save --output detailed_preview.png
```

## Features

### Visualization Details

The preview command creates a comparison visualization showing:

- **Original Images**: Raw data as loaded from your dataset
- **Processed Images**: After applying preprocessing and augmentation
- **Class Labels**: Displayed with each sample
- **Side-by-side Comparison**: Easy to see the effect of transformations

### Supported Data Loaders

- **CIFAR-10 NPZ**: Direct preview from `.npz` files
- **Directory Structure**: Standard image classification folder structure
- **Custom Data Loaders**: Any configured data loader in your config

### Preprocessing Support

Shows effects of configured preprocessing:
- Resizing
- Normalization (min-max, zero-center)
- Custom preprocessing functions

### Augmentation Support

Visualizes augmentation effects (training split only):
- Horizontal/Vertical flipping
- Rotation
- Brightness adjustment
- Contrast adjustment
- Gaussian noise
- Custom augmentation functions

## Output Formats

The preview generates plots in common image formats:
- PNG (recommended for high quality)
- JPEG
- PDF (for vector graphics)

## Configuration Requirements

Your `config.yaml` should include:

```yaml
configuration:
  data:
    data_loader:
      selected_data_loader: "Custom_load_cifar10_npz_data"  # or other
      parameters:
        npz_file_path: "./data/cifar10.npz"
    
    preprocessing:
      Resizing:
        enabled: true
        target_size:
          height: 224
          width: 224
      
      Normalization:
        enabled: true
        method: "zero-center"
    
    augmentation:
      "Horizontal Flip":
        enabled: true
        probability: 0.5
      
      "Rotation":
        enabled: true
        probability: 0.3
        angle_range: 15.0
```

## Examples by Data Type

### CIFAR-10 NPZ Files

```bash
# Basic CIFAR-10 preview
mg preview --config config.yaml

# Compare training vs validation augmentation
mg preview --config config.yaml --split train --num-samples 8
mg preview --config config.yaml --split val --num-samples 8
```

### Directory-based Images

```bash
# Preview images from directory structure
mg preview --config config.yaml --split train

# Save comparison for documentation
mg preview --config config.yaml --save --output dataset_overview.png
```

## Tips and Best Practices

### Debugging Data Pipeline

1. **Check Data Loading**: Verify your data loads correctly
2. **Validate Preprocessing**: Ensure preprocessing doesn't distort images
3. **Optimize Augmentation**: Adjust augmentation parameters based on visual feedback
4. **Class Balance**: Check if all classes are represented

### Performance Considerations

- Start with fewer samples (`--num-samples 4`) for quick checks
- Use `--save` to avoid blocking the terminal
- Preview different splits to ensure consistency

### Common Issues

**No Images Found**
```bash
# Check your data paths in config.yaml
mg check config.yaml --verbose
```

**Preprocessing Errors**
```bash
# Preview with minimal samples to isolate issues
mg preview --config config.yaml --num-samples 2
```

**Augmentation Too Strong**
```bash
# Compare train vs val to see augmentation effects
mg preview --config config.yaml --split train --save --output train_preview.png
mg preview --config config.yaml --split val --save --output val_preview.png
```

## Integration with Workflow

### Before Training

```bash
# 1. Create project
mg create my_project --interactive

# 2. Preview data to validate setup
mg preview --config config.yaml

# 3. Adjust config if needed
mg config config.yaml --interactive

# 4. Preview again to confirm changes
mg preview --config config.yaml --save --output final_preview.png

# 5. Start training
mg train --config config.yaml
```

### During Development

```bash
# Quick data check
mg preview --config config.yaml --num-samples 4

# Detailed analysis
mg preview --config config.yaml --num-samples 16 --save
```

## See Also

- [`config`](config.md) - Modify data preprocessing and augmentation settings
- [`check`](check.md) - Validate configuration before preview
- [`train`](train.md) - Train models after validating data pipeline
- [Data Configuration Tutorial](../tutorials/data-configuration.md)
