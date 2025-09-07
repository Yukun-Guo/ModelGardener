# ModelGardener AI Coding Instructions

## Architecture Overview

ModelGardener is a **modular ML framework** that auto-generates Python scripts from YAML configs and supports **custom function injection**. The core philosophy: users configure via YAML, framework generates executable Python scripts.

### Key Components (src/modelgardener/)
- **`enhanced_trainer.py`**: Main orchestrator - coordinates runtime, data pipeline, model building, training
- **`script_generator.py`**: Generates `train.py`, `evaluation.py`, `prediction.py`, `deploy.py` from config
- **`config_manager.py`**: Handles config saving/loading with custom function metadata
- **`scalable_dataset_loader.py`**: Creates optimized tf.data pipelines with custom data loaders
- **`enhanced_model_builder.py`**: Builds models (built-in or custom architectures)
- **`preprocessing_pipeline.py`**: Built-in preprocessing + custom preprocessing functions
- **`custom_functions_loader.py`**: Dynamic loading of user-defined functions

### Custom Functions Architecture
Users extend functionality via `custom_modules/` directory:
```
custom_modules/
├── custom_data_loaders.py    # Custom data loading logic
├── custom_models.py          # Custom model architectures  
├── custom_loss_functions.py  # Custom loss functions
├── custom_metrics.py         # Custom evaluation metrics
├── custom_callbacks.py       # Custom training callbacks
├── custom_preprocessing.py   # Custom data preprocessing
└── custom_augmentations.py   # Custom data augmentation
```

## Development Workflows

### Testing Enhanced Functions
```bash
# Configure Python environment first
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Test specific modules
python test_simple_validation.py  # Basic functionality test
python quick_validation.py        # Comprehensive validation
```

### Working with Custom Functions
- **Example functions**: Located in `src/modelgardener/example_funcs/` - enhanced versions support multi-input/output, 2D/3D data, multiple task types
- **Function loading order**: `./custom_modules/` → `example_funcs/` package → relative `example_funcs/` path
- **Function structure**: Each function category has specific signature requirements (see `CustomFunctionsLoader` class)

### CLI Development
```bash
# CLI entry point
mg --help                    # Main CLI interface
mg create --interactive      # Project creation wizard
mg config --interactive      # Config modification wizard
mg train --config config.yaml
```

## Critical Patterns

### Custom Function Integration
Functions must follow specific signatures:
```python
# Data loaders: return tf.data.Dataset or (train_ds, val_ds) tuple
def my_data_loader(data_dir, batch_size, **kwargs) -> tf.data.Dataset

# Models: return compiled Keras model
def my_model(input_shape, num_classes, **kwargs) -> tf.keras.Model

# Loss functions: return callable or loss value
def my_loss(y_true, y_pred, **kwargs) -> tf.Tensor
```

### Configuration-Driven Generation
Key insight: The framework **generates Python scripts**, not just trains models. When modifying:
1. Update `script_generator.py` templates for new script features
2. Update `config_manager.py` for new config structure
3. Update `enhanced_trainer.py` for new training capabilities

### Multi-Input/Output Support
Recent enhancement: All `example_funcs/` now support:
- **Multi-input**: Lists, dicts, or single tensors
- **2D/3D data**: Automatic dimension detection
- **Task types**: Classification, segmentation, object detection
- **Utilities**: `utils.py` provides helper functions for dimension detection, task inference

### Error Handling Strategy
- **Custom function loading**: Graceful fallbacks with warnings, don't crash
- **Auto-loading hierarchy**: Try multiple paths before failing
- **Validation**: Check function signatures before execution

## Integration Points

### EnhancedTrainer Pipeline
4-phase training: Runtime Setup → Data Pipeline → Model Building → Training Execution
- Each phase returns bool for success/failure
- Custom functions injected at each phase
- Supports standard training, cross-validation, custom loops

### Data Loading Strategy
- **Built-in loaders**: Standard image directory loading
- **Custom loaders**: User-defined functions with caching support
- **Optimization**: tf.data pipeline with prefetch, cache, parallel processing

### Preprocessing Pipeline
Order: Built-in sizing/normalization → Custom preprocessing functions
- Supports both 2D images and 3D volumes
- Custom functions can return wrapper functions or process directly

## File Navigation

### Core Architecture Files
- `enhanced_trainer.py:25-400` - Main training orchestration
- `script_generator.py:80-200` - Template generation logic
- `scalable_dataset_loader.py:40-150` - Dataset creation patterns

### Custom Function Examples
- `example_funcs/utils.py` - Multi-input/output utilities
- `example_funcs/example_custom_models.py` - Adaptive model architectures
- `example_funcs/example_custom_data_loaders.py` - Enhanced data loading

### CLI Implementation
- `modelgardener_cli.py:1-100` - Main CLI class structure
- `cli_config.py` - Interactive configuration wizards

When working on this codebase, prioritize understanding the **custom function loading mechanism** and **configuration-to-script generation flow** - these are the unique architectural decisions that differentiate ModelGardener from standard ML frameworks.

### Notes
python venv is .venv/bin/python， please use this python interpreter test the code.