# Loss Function Configuration Improvements

## Overview

The loss function configuration in the ModelGardener CLI's `create` command with interactive mode has been significantly improved to provide a better user experience with automatic model analysis and intelligent configuration guidance.

## Key Improvements

### 1. Automatic Model Output Analysis
- **Before**: Users had to manually specify the number of outputs without guidance
- **After**: The system automatically analyzes the loaded model to detect the number of outputs and their names

### 2. Intelligent Output Name Detection
- **Before**: Generic output names like `output_1`, `output_2`
- **After**: Meaningful names like `main_output`, `aux_output`, or custom names from the model definition

### 3. Step-by-Step Guided Process
The new workflow follows a clear 4-step process:

#### Step 1: Model Analysis
- Automatically analyzes the configured model
- For custom models: Loads and inspects the actual model structure
- For built-in models: Uses naming conventions and heuristics
- Provides fallback source code analysis if model building fails

#### Step 2: Model Output Information Update
- Updates the configuration with detected output information
- Allows user to override if the detection is incorrect
- Shows clear summary of detected outputs and names

#### Step 3: Loss Strategy Selection
- **Single Output**: Automatically selects `single_loss_all_outputs`
- **Multiple Outputs**: Presents user with strategy choices:
  - `single_loss_all_outputs`: Use same loss function for all outputs
  - `different_loss_each_output`: Configure different loss functions for each output

#### Step 4: Loss Function Configuration
- Configures loss functions based on the selected strategy
- For multiple outputs with different losses: Iterates through each output with clear labels
- Supports both built-in and custom loss functions

### 4. Enhanced Custom Model Support
- **Dynamic Model Loading**: Actually loads and builds custom models to analyze their structure
- **Source Code Analysis**: Falls back to analyzing the source code if model building fails
- **Pattern Recognition**: Detects multiple output patterns in code (auxiliary outputs, named outputs, etc.)

### 5. Better Error Handling and Feedback
- Clear progress indicators at each step
- Meaningful error messages when analysis fails
- Graceful fallbacks when automatic detection isn't possible

## Technical Implementation

### Core Methods Added/Modified

1. **`analyze_model_outputs(config)`**
   - Main analysis method that coordinates the detection process
   - Handles both custom and built-in models
   - Returns number of outputs and their names

2. **`_analyze_custom_model_outputs(file_path, function_name, model_config)`**
   - Loads custom model functions/classes and builds them
   - Inspects the resulting Keras model structure
   - Extracts meaningful output names from tensor names

3. **`_analyze_model_source_code(model_func)`**
   - Fallback method for source code analysis
   - Uses regex patterns to detect multiple output patterns
   - Extracts output names from variable names and layer definitions

4. **Enhanced `configure_loss_functions(config)`**
   - Completely redesigned workflow with 4 clear steps
   - Automatic strategy selection for single outputs
   - Better user guidance and confirmations

5. **Updated `_configure_multiple_losses(num_outputs, output_names)`**
   - Now accepts detected output names as parameter
   - Uses meaningful names in user prompts
   - Maintains backward compatibility

### Configuration Structure

The resulting configuration follows this improved structure:

```json
{
  "Model Output Configuration": {
    "num_outputs": 2,
    "output_names": "main_output,aux_output_1",
    "loss_strategy": "different_loss_each_output"
  },
  "Loss Selection": {
    "main_output": {
      "selected_loss": "Categorical Crossentropy",
      "custom_loss_path": null,
      "parameters": {}
    },
    "aux_output_1": {
      "selected_loss": "Binary Crossentropy", 
      "custom_loss_path": null,
      "parameters": {}
    }
  }
}
```

## Example Workflows

### Single Output Model (ResNet-50)
```
🔍 Step 1: Analyzing model outputs...
   ✅ Detected 1 output (default)
   
📝 Step 2: Model Output Information
   Detected outputs: 1
   Output names: ['main_output']
   
⚙️ Step 3: Loss Strategy Selection
   Single output detected - using 'single_loss_all_outputs' strategy
   
🎯 Step 4: Loss Function Selection
   Strategy: single_loss_all_outputs
   [User selects loss function]
```

### Multi-Output Custom Model
```
🔍 Step 1: Analyzing model outputs...
   Custom model: create_simple_cnn_two_outputs
   ✅ Detected 2 output(s): ['main_output', 'aux_output_1']
   
📝 Step 2: Model Output Information
   Detected outputs: 2
   Output names: ['main_output', 'aux_output_1']
   
⚙️ Step 3: Loss Strategy Selection
   Multiple outputs detected (2) - please select strategy:
   [User chooses between single or different loss strategies]
   
🎯 Step 4: Loss Function Selection
   Strategy: different_loss_each_output
   Configuring loss function for 'main_output': [User selects]
   Configuring loss function for 'aux_output_1': [User selects]
```

## Benefits

1. **Improved User Experience**: Clear, step-by-step guidance with automatic detection
2. **Reduced Errors**: Automatic analysis reduces manual configuration mistakes
3. **Better Model Support**: Enhanced support for complex custom models
4. **Flexibility**: Users can override automatic detection when needed
5. **Consistency**: Standardized workflow regardless of model complexity

## Backward Compatibility

The improvements maintain full backward compatibility:
- Existing configuration files continue to work
- Legacy method signatures are preserved
- Default behaviors match previous functionality when automatic detection fails

## Testing

The implementation includes comprehensive tests covering:
- Single output models (built-in architectures)
- Multi-output custom models with various patterns
- Source code analysis fallback scenarios
- Different loss strategy configurations
- Error handling and edge cases
