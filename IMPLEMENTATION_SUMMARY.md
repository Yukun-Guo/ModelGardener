# ModelGardener Parameter Tree Enhancements

## Summary

This document summarizes the implementation of two key enhancements to the ModelGardener parameter tree configuration:

1. **Task Type Parameter** - Added to Basic Configuration
2. **K-Fold Cross-Validation** - Added to Advanced Configuration

## 1. Task Type Parameter

### Location
- **Configuration Section**: Basic Configuration
- **Position**: Above the Data Configuration section
- **Parameter Name**: `task_type`

### Implementation Details

#### Available Task Types
The following 12 image-related computer vision tasks are available:

1. `image_classification` - Standard image classification tasks
2. `semantic_segmentation` - Pixel-level classification for image segmentation
3. `object_detection` - Detecting and localizing objects in images
4. `instance_segmentation` - Segmenting individual object instances
5. `image_generation` - Generating new images (GANs, VAEs, etc.)
6. `style_transfer` - Transferring artistic styles between images
7. `super_resolution` - Enhancing image resolution and quality
8. `image_denoising` - Removing noise from images
9. `depth_estimation` - Estimating depth information from images
10. `pose_estimation` - Detecting human poses and key points
11. `face_recognition` - Identifying and verifying faces
12. `optical_flow` - Estimating motion between image frames

#### UI Configuration
- **Type**: Dropdown list (parameter type: `list`)
- **Default Value**: `image_classification`
- **Tooltip**: "Type of computer vision task to perform (classification, segmentation, detection, etc.)"

#### Code Changes
- Added to `basic_config['task_type']` in `create_comprehensive_config()` method
- Added parameter handling in `dict_to_params()` method for dropdown display
- Added tooltip description in `get_parameter_tooltip()` method

## 2. K-Fold Cross-Validation Configuration

### Location
- **Configuration Section**: Advanced Configuration
- **Position**: Between Callbacks and Training Advanced sections
- **Parameter Group**: `cross_validation`

### Implementation Details

#### Available Parameters
The cross-validation configuration includes 10 parameters:

| Parameter | Type | Default | Range/Options | Description |
|-----------|------|---------|---------------|-------------|
| `enabled` | Boolean | `false` | true/false | Enable k-fold cross-validation |
| `k_folds` | Integer | `5` | 2-20 | Number of folds to divide dataset |
| `validation_split` | Float | `0.2` | 0.1-0.5 (step: 0.01) | Validation fraction per fold |
| `stratified` | Boolean | `true` | true/false | Preserve class distribution |
| `shuffle` | Boolean | `true` | true/false | Shuffle data before splitting |
| `random_seed` | Integer | `42` | 0-999999 | Random seed for reproducibility |
| `save_fold_models` | Boolean | `false` | true/false | Save individual fold models |
| `fold_models_dir` | Directory | `./fold_models` | Directory path | Folder for fold models |
| `aggregate_metrics` | Boolean | `true` | true/false | Calculate mean/std across folds |
| `fold_selection_metric` | List | `val_accuracy` | 8 options | Metric for best fold selection |

#### Fold Selection Metrics
Available options for `fold_selection_metric`:
- `val_accuracy` - Validation accuracy
- `val_loss` - Validation loss
- `accuracy` - Training accuracy  
- `loss` - Training loss
- `val_precision` - Validation precision
- `val_recall` - Validation recall
- `val_f1_score` - Validation F1 score
- `val_auc` - Validation Area Under Curve

#### UI Configuration
- **Group Type**: Parameter group with expandable children
- **Default State**: Collapsed (enabled = false)
- **Parameter Types**: Mixed (boolean, integer, float, directory, dropdown)

#### Code Changes
- Added `cross_validation` section to `advanced_config` in `create_comprehensive_config()` method
- Added parameter type handling in `dict_to_params()` method:
  - `k_folds` and `random_seed` as integer parameters with appropriate ranges
  - `validation_split` as float parameter with 0.1-0.5 range and 0.01 step
  - `fold_selection_metric` as dropdown list with 8 metric options
  - `fold_models_dir` as directory_only parameter
  - Boolean parameters handled automatically
- Added comprehensive tooltips for all parameters in `get_parameter_tooltip()` method

## Implementation Files Modified

### Primary File: `main_window.py`

#### 1. Configuration Structure (`create_comprehensive_config()` method)
- Added `task_type` to basic configuration (line ~1658)
- Added `cross_validation` section to advanced configuration (line ~1745)

#### 2. Parameter Tree Conversion (`dict_to_params()` method)
- Added `task_type` dropdown handling (line ~2225)
- Added `fold_selection_metric` dropdown handling (line ~2355)
- Added `k_folds` and `random_seed` integer parameter handling (line ~2370)
- Added `validation_split` float parameter handling (line ~2390)
- Added `fold_models_dir` directory parameter handling (line ~2218)

#### 3. Tooltip Definitions (`get_parameter_tooltip()` method)
- Added task type tooltip (line ~2057)
- Added 12 cross-validation parameter tooltips (lines 2058-2070)

## Example Configuration

### Basic Configuration with Task Type
```json
{
  "basic": {
    "task_type": "semantic_segmentation",
    "data": {
      "train_dir": "/path/to/training/data",
      "val_dir": "/path/to/validation/data"
    }
  }
}
```

### Advanced Configuration with K-Fold CV
```json
{
  "advanced": {
    "cross_validation": {
      "enabled": true,
      "k_folds": 10,
      "validation_split": 0.15,
      "stratified": true,
      "shuffle": true,
      "random_seed": 12345,
      "save_fold_models": true,
      "fold_models_dir": "./experiments/fold_models",
      "aggregate_metrics": true,
      "fold_selection_metric": "val_f1_score"
    }
  }
}
```

## Usage Instructions

### For Task Type Parameter
1. Navigate to **Basic Configuration** in the parameter tree
2. Locate **Task Type** parameter at the top (above Data section)
3. Select from the dropdown menu containing 12 computer vision tasks
4. Default selection is `image_classification`

### For K-Fold Cross-Validation
1. Navigate to **Advanced Configuration** in the parameter tree
2. Locate **Cross Validation** section (between Callbacks and Training Advanced)
3. Enable cross-validation by setting `enabled` to `true`
4. Configure the following key parameters:
   - Set `k_folds` (recommend 5-10 for most cases)
   - Adjust `validation_split` if needed (0.2 is typical)
   - Choose appropriate `fold_selection_metric` for your task
   - Optionally enable `save_fold_models` and set `fold_models_dir`

## Testing and Validation

The implementation has been thoroughly tested with:
- ✅ Configuration structure validation
- ✅ Parameter type compatibility
- ✅ UI component compatibility (dropdowns, ranges, directories)
- ✅ Tooltip coverage and accuracy
- ✅ Default value appropriateness
- ✅ Range and limit validation

All tests pass successfully, confirming the implementation is robust and ready for production use.

## Benefits

### Task Type Parameter
- **Improved Clarity**: Users immediately understand what type of task they're configuring
- **Better Organization**: Provides context for all subsequent configuration choices
- **Future Extensibility**: Easy to add new computer vision tasks as they become available

### K-Fold Cross-Validation
- **Robust Evaluation**: Provides more reliable performance estimates than single train/val splits
- **Comprehensive Configuration**: All key CV parameters are available for fine-tuning
- **Professional Features**: Includes advanced options like stratification, model saving, and metric aggregation
- **User-Friendly**: Intuitive parameter organization with helpful tooltips and appropriate UI controls
