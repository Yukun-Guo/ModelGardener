# Auto-Discovery Features

ModelGardener's CLI provides intelligent auto-discovery capabilities that eliminate the need for repetitive parameter specification, making workflows faster and more intuitive.

## Overview

Auto-discovery automatically locates commonly used files and directories, allowing you to run commands with minimal parameters. This feature is available across all major CLI commands: `evaluate`, `predict`, and `deploy`.

## How Auto-Discovery Works

### 🔍 File Discovery Priority

#### Configuration Files
1. **Current Directory**: Looks for `config.yaml` in the working directory
2. **Fallback**: Provides helpful error messages if not found

#### Model Files (in `logs/` directory)
1. **`final_model.keras`** - Primary trained model
2. **`model.keras`** - Standard model file
3. **`best_model.keras`** - Best performing checkpoint
4. **Latest Timestamped Model** - Most recent versioned model directory

#### Input Data (for prediction)
Searches for test directories in this order:
1. `test/`
2. `test_data/`
3. `test_images/`
4. `val/`
5. `data/test/`
6. `data/val/`
7. `prediction_data/`
8. Single image files in current directory

### 📁 Directory Structure Recognition

Auto-discovery works best with ModelGardener's standard project structure:

```
my_project/
├── config.yaml          # Auto-discovered config
├── logs/                 # Auto-discovered models
│   ├── model_version_01/
│   ├── model_version_02/
│   └── final_model.keras # Highest priority
├── test/                 # Auto-discovered test data
│   ├── class1/
│   └── class2/
├── evaluation/           # Auto-generated reports
├── predictions/          # Auto-generated results
└── deployed_models/      # Auto-generated deployments
```

## Command-Specific Auto-Discovery

### `mg evaluate` 
```bash
# Full auto-discovery
mg evaluate

# What it finds:
# ✓ config.yaml (in current directory)
# ✓ Latest model in logs/
# ✓ Evaluation data from config
```

**Auto-Discovery Benefits:**
- Finds latest trained model automatically
- Uses evaluation data path from configuration
- Creates timestamped reports in `evaluation/` folder
- Supports both JSON and YAML output formats

### `mg predict`
```bash
# Full auto-discovery
mg predict

# What it finds:
# ✓ config.yaml (in current directory)
# ✓ Latest model in logs/
# ✓ Test data in common directories
```

**Auto-Discovery Benefits:**
- Locates test images and directories automatically
- Supports multiple image formats (jpg, png, jpeg, bmp, tiff, webp)
- Creates comprehensive prediction reports with metadata
- Generates both JSON reports and CSV summaries

### `mg deploy`
```bash
# Full auto-discovery
mg deploy

# What it finds:
# ✓ config.yaml (in current directory)
# ✓ Latest model in logs/
# ✓ Uses default formats: onnx, tflite
```

**Auto-Discovery Benefits:**
- Automatically selects latest trained model
- Uses sensible default formats for deployment
- Creates organized deployment directories
- Supports all major model formats

## Selective Auto-Discovery

You can mix auto-discovery with explicit parameters:

### Partial Override Examples

```bash
# Auto-discover config and model, specify custom data
mg evaluate -d ./custom_test_data

# Auto-discover everything except output format
mg evaluate --output-format json

# Auto-discover config, specify custom model
mg predict -m ./models/specific_model.keras

# Auto-discover config and model, specify formats
mg deploy -f keras onnx
```

### Override Priority

When you specify a parameter explicitly, it takes precedence over auto-discovery:

1. **Explicit Parameters** (highest priority)
2. **Auto-Discovery**
3. **Default Values** (lowest priority)

## Short Parameters Support

All auto-discoverable commands support intuitive short parameters:

| Long Parameter | Short | Usage Example |
|---------------|-------|---------------|
| `--config` | `-c` | `mg evaluate -c custom_config.yaml` |
| `--model-path` | `-m` | `mg predict -m ./models/model.keras` |
| `--input` | `-i` | `mg predict -i ./test_images/` |
| `--output` | `-o` | `mg predict -o results.json` |
| `--data-path` | `-d` | `mg evaluate -d ./validation_data/` |
| `--formats` | `-f` | `mg deploy -f onnx tflite` |
| `--quantize` | `-q` | `mg deploy -f tflite -q` |
| `--encrypt` | `-e` | `mg deploy -f onnx -e` |

## Best Practices

### 1. Use Standard Project Structure
Organize your project using ModelGardener's recommended structure for optimal auto-discovery:

```bash
mg create my_project --interactive  # Creates proper structure
cd my_project
mg train                            # Uses auto-discovered config.yaml
mg evaluate                         # Uses auto-discovered model
mg predict                          # Uses auto-discovered test data
mg deploy                           # Uses auto-discovered model
```

### 2. Leverage Default Workflows
Take advantage of auto-discovery for common workflows:

```bash
# Complete ML pipeline with minimal typing
mg train && mg evaluate && mg predict && mg deploy
```

### 3. Mix Auto-Discovery with Custom Parameters
Use auto-discovery for common files, explicit parameters for custom needs:

```bash
# Auto-discover config and model, use custom test data
mg evaluate -d ./custom_validation_set/

# Auto-discover everything, just change output format
mg evaluate --output-format json

# Auto-discover config and model, deploy to custom directory
mg deploy -f onnx tflite -o production_models/
```

### 4. Understand Override Behavior
Know when to use explicit parameters vs. auto-discovery:

```bash
# Let auto-discovery handle everything (recommended for standard workflows)
mg evaluate

# Override specific aspects when needed
mg evaluate -m ./models/experimental_model.keras  # Test specific model
mg predict -i ./edge_cases/                       # Test specific data
mg deploy -f keras -o ./web_app/models/           # Deploy for specific use
```

## Error Handling

Auto-discovery provides helpful error messages when files can't be found:

```bash
$ mg evaluate
❌ No configuration file found. Please specify with -c or ensure config.yaml exists in current directory.

$ mg predict
🔍 Using auto-discovered config: config.yaml
❌ No model found in logs/ directory. Please specify with -m or train a model first.

$ mg deploy
🔍 Using auto-discovered config: config.yaml
🔍 Using auto-discovered model: logs/model_version_03/final_model.keras
✅ Deployment completed successfully!
```

## Troubleshooting

### Common Issues and Solutions

**Issue**: "No configuration file found"
```bash
# Solution: Ensure config.yaml exists or specify path
mg evaluate -c path/to/config.yaml
```

**Issue**: "No model found in logs/ directory"
```bash
# Solution: Train a model first or specify model path
mg train                                    # Train first
# OR
mg evaluate -m path/to/model.keras         # Specify model
```

**Issue**: "No input data found"
```bash
# Solution: Create test directory or specify input path
mkdir test && cp images/* test/             # Create test directory
# OR
mg predict -i path/to/test/images/         # Specify input
```

**Issue**: Auto-discovery finds wrong file
```bash
# Solution: Use explicit parameters to override
mg evaluate -m ./models/specific_model.keras  # Override model discovery
mg predict -i ./specific_test_data/           # Override input discovery
```

## Advanced Features

### Model Version Selection
Auto-discovery prioritizes model files in this order:
1. `final_model.keras` (production-ready model)
2. `model.keras` (standard model file)
3. `best_model.keras` (best checkpoint)
4. Latest versioned directory (e.g., `model_version_05/`)

### Multi-Directory Support
For input discovery, the system checks multiple common directory names to maximize compatibility with different project organizations.

### Graceful Fallbacks
When auto-discovery fails, the system provides clear guidance on how to resolve issues, whether by creating missing files or using explicit parameters.
