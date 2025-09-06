# Data Preview Command

The `preview` command in ModelGardener CLI allows you to visualize data samples from your configured datasets before training, showing both original images and the processed versions after applying configured preprocessing and augmentation. This helps verify your data pipeline and understand exactly what your model will see during training.

## Usage

```bash
mg preview --config <config_file> [options]
```

## Options

- `--config`, `-c`: Configuration file (required)
- `--num-samples`, `-n`: Number of samples to preview (default: 8)
- `--split`, `-s`: Data split to preview - train, val, or test (default: train)
- `--save`: Save plot to file instead of displaying
- `--output`, `-o`: Output file path for saved plot

## Key Features

### ✨ **Original vs Processed Comparison**
- Shows original images alongside processed versions
- Applies exact preprocessing and augmentation from your config
- Different processing for train vs validation splits

### 🔧 **Preprocessing Visualization**
- **Resizing**: Shows effect of target size changes
- **Normalization**: Visualizes min-max, zero-center, or custom normalization
- **Custom Preprocessing**: Displays results of custom preprocessing functions

### 🎨 **Augmentation Preview**
- **Training Split**: Shows augmented images (random effects applied)
- **Validation Split**: Shows only preprocessing (no augmentation)
- **Supported Augmentations**: Horizontal/Vertical flip, Rotation, Brightness, Contrast, Gaussian Noise

## Examples

### Basic Preview
```bash
# Preview with preprocessing only
mg preview --config config.yaml

# Preview validation data (no augmentation)
mg preview --config config.yaml --split val
```

### Advanced Usage
```bash
# Preview with more samples
mg preview --config config.yaml --num-samples 12

# Save comparison plot
mg preview --config config.yaml --save --output data_processing_comparison.png
```

## Supported Data Loaders

### 1. CIFAR-10 NPZ Data Loader (`Custom_load_cifar10_npz_data`)
- Automatically loads data from configured NPZ file
- Displays CIFAR-10 class names (airplane, automobile, bird, etc.)
- Shows normalized images with proper class labels

### 2. Directory-based Image Data (`ImageDataLoader`, `Default`)
- Loads images from train/val directory structures
- Supports common image formats: PNG, JPG, JPEG, BMP, TIFF
- Automatically detects class directories
- Shows images with directory-based class names

## Output

The preview command provides:

1. **Terminal Output**: 
   - Configuration summary
   - Data loader information
   - Sample information with class names
   - File paths and statistics

2. **Visual Output**:
   - Grid layout of sample images
   - Class labels as titles
   - Professional plot formatting
   - Optional save to PNG file

## Example Output

```
🔍 Previewing data from configuration: config.yaml
📊 Data loader: Custom_load_cifar10_npz_data
🎯 Split: train
📈 Number of samples to preview: 8
📂 Loading data from: ./data/cifar10.npz
📊 Loaded data shape: (1000, 32, 32, 3)
🎯 Labels shape: (1000,)
🏷️ Unique classes: [0 1 2 3 4 5 6 7 8 9]
🎨 Creating visualization...
Sample 1: Class 6: frog
Sample 2: Class 0: airplane
Sample 3: Class 9: truck
Sample 4: Class 1: automobile
💾 Plot saved to: data_preview_train_Custom_load_cifar10_npz_data.png
✅ Data preview completed!
```

## Benefits

- **Data Verification**: Ensure your data is loading correctly
- **Class Distribution**: Visualize class balance and variety
- **Quality Check**: Spot potential issues with images or labels
- **Documentation**: Save preview plots for reports or presentations
- **Debugging**: Identify data loading problems early

## Future Enhancements

The preview command will be extended to support:
- Additional data loader types
- Custom visualization layouts
- Statistical summaries
- Batch processing insights
- Augmentation previews
