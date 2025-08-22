# Model Parameters Verification Guide

## 🔍 How to Verify Model Parameters are Working

The ModelGardener application now includes comprehensive model-specific parameters based on real keras.applications and other model implementations. Here's how to verify they're working correctly:

## 1. Application Startup

When you run the application:
```bash
.venv/bin/python main.py
```

The application should:
- ✅ Start without errors
- ✅ Display the parameter tree on the left side
- ✅ Show "Basic Configuration" and "Advanced Configuration" tabs

## 2. Locating Model Parameters

To find the model parameters:

1. **Open Basic Configuration tab** (should be open by default)
2. **Expand "model" section**
3. **Look for "model_parameters"** - this should contain the actual model creation parameters

## 3. What You Should See

Under `Basic Configuration > model > model_parameters`, you should see:

### For ResNet50 (default):
- `input_shape` (group with height, width, channels)
- `include_top` (boolean)
- `weights` (dropdown: imagenet/None)  
- `pooling` (dropdown: None/avg/max)
- `classes` (integer: 1000)
- `classifier_activation` (dropdown: softmax/sigmoid/etc.)
- `load_custom_model` (button)

### For MobileNet (when selected):
- All the above ResNet parameters, PLUS:
- `alpha` (float: width multiplier)
- `depth_multiplier` (integer)  
- `dropout` (float)

### For EfficientNet (when selected):
- All the common parameters, PLUS:
- `drop_connect_rate` (float: stochastic depth)

## 4. Testing Dynamic Parameter Updates

1. **Change task_type**: 
   - Go to Basic Configuration
   - Change "task_type" from "image_classification" to "object_detection"
   - Watch the model_family options update
   - Model parameters should change to detection-specific parameters

2. **Change model_name**:
   - Select different models (ResNet50 → MobileNet → EfficientNetB0)
   - Model parameters should update automatically
   - Each model should show its specific parameters

3. **Check the log output**:
   - The bottom panel should show messages like:
   - "Updated model parameters for MobileNet (image_classification)"
   - "Forced UI refresh - X parameters displayed"

## 5. Expected Parameter Counts

| Model | Task | Parameter Count | Special Parameters |
|-------|------|----------------|------------------|
| ResNet50 | Classification | 7 | Standard keras.applications |
| MobileNet | Classification | 10 | alpha, depth_multiplier, dropout |
| EfficientNetB0 | Classification | 8 | drop_connect_rate |
| YOLOv8 | Detection | 10 | anchors, ultralytics_format |
| U-Net | Segmentation | 10 | filters, attention, deep_supervision |

## 6. Troubleshooting

If model parameters are not showing:

### Check 1: Verify ModelGroup is working
```bash
.venv/bin/python diagnostic_model_params.py
```
All tests should pass.

### Check 2: Look for error messages
- Check the log panel at the bottom of the application
- Look for error messages about model parameters

### Check 3: Manual refresh
- Try changing the model selection back and forth
- Each change should trigger a parameter update

### Check 4: Check parameter tree structure
The structure should be:
```
Basic Configuration
├── task_type
├── data
└── model
    ├── model_family
    ├── model_name
    └── model_parameters  ← HERE ARE THE REAL MODEL PARAMETERS
        ├── input_shape
        ├── include_top
        ├── weights
        ├── pooling
        ├── classes
        ├── classifier_activation
        └── [model-specific parameters]
```

## 7. What's Different Now

**BEFORE**: Only saw basic parameters (type, name, model_name, task_type)
**AFTER**: See actual keras model creation parameters that match real APIs

**Example - keras.applications.ResNet50() parameters:**
- `include_top=True` ✅ 
- `weights="imagenet"` ✅
- `input_shape=None` ✅ (as group)
- `pooling=None` ✅
- `classes=1000` ✅
- `classifier_activation="softmax"` ✅

## 8. Success Indicators

✅ **Model parameters section expands to show 6+ parameters**
✅ **Parameters change when switching models (ResNet→MobileNet)**
✅ **MobileNet shows alpha parameter with limits [0.25, 0.35, 0.5, 0.75, 1.0, 1.3, 1.4]**
✅ **EfficientNet shows drop_connect_rate parameter**
✅ **Detection models show IoU/confidence thresholds**
✅ **Log messages confirm parameter updates**

If you see all these indicators, the model parameters are working perfectly! 🎉

## Need More Help?

If the model parameters still aren't showing, try:
1. Restart the application
2. Change the model selection to trigger an update
3. Check that you're looking in the right place: `model > model_parameters`
4. Run the diagnostic script to verify the underlying system works
