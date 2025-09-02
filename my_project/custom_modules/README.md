# Custom Modules for ModelGardener

This directory contains templates for custom functions that can be used with ModelGardener.

## Available Templates

- `custom_models.py` - Custom model architectures
- `custom_data_loaders.py` - Custom data loading functions
- `custom_loss_functions.py` - Custom loss functions
- `custom_optimizers.py` - Custom optimizers
- `custom_metrics.py` - Custom metrics
- `custom_callbacks.py` - Custom training callbacks
- `custom_augmentations.py` - Custom data augmentation functions
- `custom_preprocessing.py` - Custom preprocessing functions
- `custom_training_loops.py` - Custom training loop strategies

## Usage

1. **Customize the Templates**: Edit the template files to implement your custom functions
2. **Update Configuration**: Add references to your custom functions in the configuration file
3. **Use in Training**: The generated scripts will automatically load and use your custom functions

## Example Configuration

```yaml
metadata:
  custom_functions:
    models:
      - name: "MyCustomModel"
        file_path: "./custom_modules/custom_models.py"
        function_name: "create_my_custom_model"
    loss_functions:
      - name: "MyCustomLoss"
        file_path: "./custom_modules/custom_loss_functions.py"  
        function_name: "my_custom_loss"
```

## Notes

- All custom functions should follow the patterns shown in the templates
- Make sure to install any additional dependencies required by your custom functions
- Test your custom functions independently before using them in training

## Support

Refer to the ModelGardener documentation for more details on custom functions.
