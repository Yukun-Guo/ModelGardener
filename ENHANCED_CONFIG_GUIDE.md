# Enhanced Configuration System for ModelGardener

## Overview

The enhanced configuration system in ModelGardener provides a comprehensive solution for saving, loading, and sharing configurations with custom functions. This enables users to easily collaborate and share their complete ML setups, including custom data loaders, loss functions, augmentations, and more.

## Key Features

### 1. **Embedded Custom Functions**
- Custom function source code is embedded directly in configuration files
- Includes file checksums for integrity verification
- Automatic dependency detection and listing
- Parameter and type hint extraction

### 2. **Multiple Sharing Strategies**
- **File Paths Only** (Legacy): References to local file paths
- **File Paths with Content** (Enhanced): Embedded source code for complete portability
- **Shareable Packages**: Complete directories with all dependencies and setup instructions

### 3. **Automatic Function Restoration**
- Auto-extraction of embedded custom functions on load
- Intelligent file placement and organization
- Checksum verification to prevent conflicts
- User-friendly error reporting and guidance

## Configuration File Structure

### Enhanced Configuration Format

```json
{
  "configuration": {
    // Your regular ModelGardener configuration
    "data": { ... },
    "model": { ... },
    "training": { ... }
  },
  "metadata": {
    "version": "1.2",
    "sharing_strategy": "file_paths_with_content",
    "creation_date": "2025-08-20T14:30:00",
    "model_gardener_version": "1.0",
    "custom_functions": {
      "data_loaders": [...],
      "loss_functions": [...],
      "augmentations": [...],
      "callbacks": [...],
      "preprocessing": [...],
      "metrics": [...],
      "optimizers": [...]
    }
  }
}
```

### Custom Function Metadata Structure

Each custom function entry includes:

```json
{
  "name": "Custom_function_name",
  "file_path": "./path/to/function.py",
  "original_name": "function_name",
  "type": "function|class",
  "file_content": "# Complete source code...",
  "file_size": 1234,
  "file_checksum": "abc123...",
  "relative_file_path": "function.py",
  "sharing_enabled": true,
  "dependencies": ["tensorflow", "numpy"],
  "imports": ["tensorflow", "numpy", "typing.List"],
  "docstring": "Function description...",
  "parameters_info": [
    {"name": "param1", "type_hint": "str"},
    {"name": "param2", "type_hint": "int"}
  ]
}
```

## Usage Guide

### Saving Configurations

#### 1. Standard Save (With Custom Functions)
```python
# In ModelGardener UI:
# 1. Load your custom functions using "Load Custom..." buttons
# 2. Configure your model parameters
# 3. Click "Save JSON" or "Save YAML"
# → Creates enhanced config with embedded custom functions
```

#### 2. Export Shareable Package
```python
# In ModelGardener UI:
# 1. Configure everything as above
# 2. Click "Export Shareable Package"
# → Creates complete directory with:
#   - model_config.json (enhanced config)
#   - custom_functions/ (organized function files)
#   - custom_functions_manifest.json (metadata)
#   - setup_custom_functions.py (setup script)
#   - README.md (instructions)
```

### Loading Configurations

#### 1. Standard Load
```python
# In ModelGardener UI:
# 1. Click "Load Config"
# 2. Select enhanced configuration file
# 3. Choose "Yes" when prompted to auto-reload custom functions
# → Automatically extracts and loads custom functions
```

#### 2. Manual Auto-Reload
```python
# If you skipped auto-reload:
# 1. Click "Auto-Reload Custom Functions"
# → Reloads from last loaded configuration
```

#### 3. Package Import
```python
# For shared packages:
# 1. Extract package to desired location
# 2. Run: python setup_custom_functions.py (optional)
# 3. Load model_config.json in ModelGardener
# 4. Auto-reload when prompted
```

## Sharing Workflows

### For Package Creators

1. **Develop Custom Functions**
   - Create custom data loaders, loss functions, etc.
   - Load them into ModelGardener using "Load Custom..." buttons
   - Configure your model with desired settings

2. **Test Configuration**
   - Ensure everything works as expected
   - Document any special requirements

3. **Create Shareable Package**
   - Click "Export Shareable Package"
   - Choose destination directory
   - Package is automatically created with all files and documentation

4. **Share Package**
   - Zip the generated package directory
   - Share via email, cloud storage, or version control
   - Recipients get complete, self-contained setup

### For Package Users

1. **Receive Package**
   - Extract shared package to local directory
   - Review README.md for specific instructions

2. **Setup (Optional)**
   - Run `python setup_custom_functions.py` to check dependencies
   - Install any missing Python packages as indicated

3. **Import to ModelGardener**
   - Open ModelGardener
   - Load the `model_config.json` file
   - Accept auto-reload when prompted
   - Start training/experimentation immediately

## Advanced Features

### Automatic Dependency Detection
The system automatically detects and lists:
- Python package imports (`tensorflow`, `numpy`, etc.)
- Common ML library dependencies
- Custom parameter types and hints

### Checksum Verification
- Ensures file integrity during sharing
- Prevents conflicts when extracting files
- Warns about modified or corrupted functions

### Intelligent File Organization
- Creates organized directory structure
- Prevents naming conflicts
- Maintains relative path references
- Supports version management

### Error Handling and Recovery
- Graceful handling of missing files
- Clear error messages and suggestions
- Fallback to manual loading when auto-reload fails
- Recovery options for corrupted configurations

## Best Practices

### For Function Developers
1. **Document Functions Thoroughly**
   - Use comprehensive docstrings
   - Include parameter descriptions and types
   - Specify dependencies clearly

2. **Keep Functions Self-Contained**
   - Minimize external dependencies
   - Use standard library when possible
   - Handle edge cases gracefully

3. **Test Before Sharing**
   - Verify functions work in clean environments
   - Test the complete save/load cycle
   - Validate shared packages

### For Package Sharing
1. **Include Comprehensive Documentation**
   - Explain the model purpose and use case
   - Document any special requirements
   - Provide example usage or datasets

2. **Version Control**
   - Tag packages with version numbers
   - Document changes between versions
   - Maintain backward compatibility when possible

3. **Validate Dependencies**
   - Test on multiple environments
   - Document minimum package versions
   - Provide installation instructions

## Troubleshooting

### Common Issues

1. **Auto-Reload Fails**
   - Check file paths in configuration
   - Verify all dependencies are installed
   - Try manual loading with "Load Custom..." buttons

2. **Function Import Errors**
   - Ensure Python syntax is valid
   - Check for missing imports
   - Verify function signatures match expectations

3. **Package Extraction Issues**
   - Check write permissions in target directory
   - Verify configuration file integrity
   - Look for path separator issues on different OS

### Getting Help

1. Check the setup script output for dependency issues
2. Review README.md in shared packages
3. Use ModelGardener's built-in error messages and suggestions
4. Try loading functions individually to isolate problems

## Migration from Legacy Configs

Legacy configurations (simple JSON/YAML files) are automatically detected and supported. To upgrade:

1. Load legacy configuration in ModelGardener
2. Re-load any custom functions manually
3. Save as new enhanced configuration
4. Consider creating shareable package for future use

The enhanced system maintains full backward compatibility while providing powerful new sharing capabilities.
